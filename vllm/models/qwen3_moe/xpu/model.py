# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""

import torch
from torch import nn

import vllm.model_executor.models.qwen3_moe as _qwen3_moe
from vllm.compilation.decorators import ignore_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeMLP,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer as _Qwen3MoeDecoderLayer,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeForCausalLM as _Qwen3MoeForCausalLM,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeModel as _Qwen3MoeModel,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeSparseMoeBlock as _Qwen3MoeSparseMoeBlock,
)
from vllm.model_executor.models.utils import (
    extract_layer_index,
)
from vllm.models.common.ops.sequence_parallel import (
    is_sequence_parallel_active,
    sp_all_gather,
    sp_shard,
    suspend_sequence_parallel_boundary,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class Qwen3MoeSparseMoeBlock(_Qwen3MoeSparseMoeBlock):
    """Adds an `already_sequence_parallel` escape hatch, mirroring
    `Qwen3NextSparseMoeBlock`: when the caller (this file's
    `Qwen3MoeDecoderLayer`) already reduce_scattered `hidden_states` via
    its wrapped `o_proj`, skip the redundant internal `sp_shard` +
    `all_gather` and stay in the sharded `[N/tp, H]` domain -- the
    model's single final `sp_all_gather` handles gathering it back.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = sp_all_gather(final_hidden_states)
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class Qwen3MoeDecoderLayer(_Qwen3MoeDecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Reimplements `_Qwen3MoeDecoderLayer.__init__` (upstream)
        # in full -- rather than calling `super().__init__()` and then
        # monkeypatching the module-global `Qwen3MoeSparseMoeBlock` that
        # it would otherwise construct -- so `self.mlp` is directly built
        # as *our* SP-aware subclass (with the `already_sequence_parallel`
        # escape hatch) with no global-state trickery. Keep this in sync
        # with upstream's `__init__` if it changes.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.parallel_config = vllm_config.parallel_config
        # From inside attention this is AllGather+qkv_proj ... o_proj+
        # ReduceScatter. Across compute regions it is ReduceScatter ->
        # sharded residual/norm -> AllGather, replacing AllReduce+norm.
        suspend_sequence_parallel_boundary(
            suspend_before=self.self_attn.qkv_proj,
            resume_after=self.self_attn.o_proj,
            parallel_config=self.parallel_config,
        )

        if isinstance(self.mlp, Qwen3MoeMLP):
            suspend_sequence_parallel_boundary(
                suspend_before=self.mlp.gate_up_proj,
                resume_after=self.mlp.down_proj,
                parallel_config=self.parallel_config,
            )
        elif (
            self.parallel_config.use_sequence_parallel
            and not self.mlp.is_sequence_parallel
        ):
            # `is_sequence_parallel` (`use_sequence_parallel_moe`) governs
            # how `self.experts` is *built* -- sharded-token dispatch vs.
            # replicated-token + all_reduce -- not just how this block's
            # forward chunks/gathers around it. Without it, `self.experts`
            # would treat the sharded shard `o_proj` hands it as if every
            # rank held the same replicated tokens.
            raise ValueError(
                "Sequence parallelism for Qwen3MoE requires "
                "`use_sequence_parallel_moe` to also be enabled so the "
                "MoE experts are built to consume sharded (not "
                "replicated) tokens; got use_sequence_parallel=True but "
                "use_sequence_parallel_moe=False."
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sp_active = is_sequence_parallel_active(self.parallel_config)
        if residual is None:
            if sp_active:
                # This layer is about to seed the residual from raw
                # embeddings: chunk to the local per-rank shard
                # [N/tp, H]. No-op when sequence parallelism isn't
                # active for this call.
                hidden_states = sp_shard(hidden_states)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Attention's output boundary has reduce-scattered hidden_states, so
        # residual addition and norm run on local token shards. A dense MLP
        # gathers them at its input boundary; an SP MoE consumes them directly.
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if isinstance(self.mlp, Qwen3MoeSparseMoeBlock) and sp_active:
            # o_proj's wrapped row-linear already reduce_scattered
            # hidden_states into the local shard -- tell the MoE block
            # not to chunk it again; its output stays sharded too.
            hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@ignore_torch_compile
class Qwen3MoeModel(_Qwen3MoeModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[torch.nn.Module] = Qwen3MoeDecoderLayer,
    ):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )
        self.parallel_config = vllm_config.parallel_config

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        full_num_tokens = positions.shape[-1]
        result = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        if is_sequence_parallel_active(self.parallel_config):
            # The last decoder layer's o_proj (SP-marked) reduce_scatters
            # back down to a per-rank shard. All-gather it, then trim off
            # the padding `sp_shard` added to make the
            # token count divisible by tp_size, so the output length
            # matches the real input length exactly.
            if isinstance(result, tuple):
                hidden_states, aux_hidden_states = result
                hidden_states = sp_all_gather(hidden_states)
                return hidden_states[:full_num_tokens], aux_hidden_states
            hidden_states = sp_all_gather(result)
            return hidden_states[:full_num_tokens]
        else:
            return result


class Qwen3MoeForCausalLM(_Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # `_Qwen3MoeForCausalLM.__init__` (upstream) itself builds
        # `self.model = Qwen3MoeModel(...)`, resolving `Qwen3MoeModel`
        # from `qwen3_moe.py`'s own module globals -- i.e. upstream's
        # plain (non-SP) model, not our subclass -- and then derives
        # `self.moe_layers`/`self.lm_head` tied weights from it.
        # Temporarily monkeypatch that global so upstream's `__init__`
        # constructs *our* SP-aware `Qwen3MoeModel` in place, instead of
        # building an upstream one (registering attention layers under
        # the same prefix we'd reuse) and then rebuilding/discarding it.
        original_model_cls = _qwen3_moe.Qwen3MoeModel
        _qwen3_moe.Qwen3MoeModel = Qwen3MoeModel
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _qwen3_moe.Qwen3MoeModel = original_model_cls
