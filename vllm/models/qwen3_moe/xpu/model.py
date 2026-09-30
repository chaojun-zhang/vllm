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
    is_sp_active,
    mark_sp_region,
    sp_all_gather,
    sp_shard,
)

logger = init_logger(__name__)


class Qwen3MoeSparseMoeBlock(_Qwen3MoeSparseMoeBlock):
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
        # Reimplements upstream's `__init__` in full (instead of calling
        # `super().__init__()` and monkeypatching the MoE block it builds)
        # so `self.mlp` is built directly as our SP-aware subclass. Keep
        # this in sync with upstream's `__init__` if it changes.
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
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
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
        self.is_moe_mlp = isinstance(self.mlp, Qwen3MoeSparseMoeBlock)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # o_proj already reduce-scattered hidden_states into a local shard,
        # so a dense MLP needs to all-gather it first; an SP MoE block can
        # consume the shard directly.
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.is_moe_mlp and is_sp_active(self.parallel_config):
            # hidden_states is already a local shard (from o_proj); tell the
            # MoE block to skip re-sharding it. Its output stays sharded too.
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
        if self.parallel_config.use_sequence_parallel:
            layers = self.layers[self.start_layer : self.end_layer]
            for i, layer in enumerate(layers):
                if isinstance(layer.mlp, Qwen3MoeMLP):
                    # Dense MLP has its own column/row-linear boundary:
                    # o_proj -> gate_up_proj, then down_proj -> next qkv_proj.
                    mark_sp_region(
                        entry_after=layer.self_attn.o_proj,
                        exit_before=layer.mlp.gate_up_proj,
                        parallel_config=self.parallel_config,
                    )
                    if i + 1 < len(layers):
                        mark_sp_region(
                            entry_after=layer.mlp.down_proj,
                            exit_before=layers[i + 1].self_attn.qkv_proj,
                            parallel_config=self.parallel_config,
                        )
                else:
                    # Sparse MoE has no linear boundary of its own and can
                    # run directly on a local shard, so the region skips
                    # over it entirely: o_proj -> next layer's qkv_proj.
                    if i + 1 < len(layers):
                        mark_sp_region(
                            entry_after=layer.self_attn.o_proj,
                            exit_before=layers[i + 1].self_attn.qkv_proj,
                            parallel_config=self.parallel_config,
                        )


class Qwen3MoeForCausalLM(_Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Upstream's `__init__` builds `self.model = Qwen3MoeModel(...)` by
        # resolving the plain (non-SP) model from this module's globals.
        # Monkeypatch that global so it builds our SP-aware `Qwen3MoeModel`
        # instead, avoiding building and discarding an upstream one first.
        original_model_cls = _qwen3_moe.Qwen3MoeModel
        _qwen3_moe.Qwen3MoeModel = Qwen3MoeModel
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _qwen3_moe.Qwen3MoeModel = original_model_cls
