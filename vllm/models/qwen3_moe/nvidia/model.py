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

import vllm.envs as envs
import vllm.model_executor.models.qwen3_moe as _qwen3_moe
from vllm.compilation.decorators import ignore_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer as _Qwen3MoeDecoderLayer,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeForCausalLM as _Qwen3MoeForCausalLM,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeMLP,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeModel as _Qwen3MoeModel,
)
from vllm.models.common.ops.sequence_parallel import (
    is_sp_active,
    mark_sp_region,
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class Qwen3MoeDecoderLayer(_Qwen3MoeDecoderLayer):
    """Identical to the upstream layer -- this vendor fork exists so
    sequence parallelism and other NVIDIA-specific behavior can be layered
    on top without touching the upstream model."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.parallel_config = vllm_config.parallel_config

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not is_sp_active(self.parallel_config):
            return super().forward(positions, hidden_states, residual)

        # The residual stream is sharded here.
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if isinstance(self.mlp, Qwen3MoeMLP):
            # Dense MLP: gate_up_proj/down_proj are SP region boundaries.
            hidden_states = self.mlp(hidden_states)
        elif self.mlp.is_sequence_parallel:
            # MoE-SP experts already take one token shard per TP rank.
            hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
        else:
            # TP experts need full tokens and return a reduced output.
            hidden_states = sp_shard(self.mlp(sp_all_gather(hidden_states)))
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
        if not self.parallel_config.use_sequence_parallel:
            return
        assert get_pp_group().world_size == 1, (
            "Sequence parallelism is not supported with pipeline parallelism"
        )
        pc = self.parallel_config
        # Each layer has two SP regions: [prev down_proj -> qkv_proj] and
        # [o_proj -> gate_up_proj]. Sparse MoE blocks have no gate_up/down
        # boundary (the decoder layer handles them), so those sides are None.
        prev_down_proj = None  # first region starts at the sharded embeddings
        for layer in self.layers[self.start_layer : self.end_layer]:
            if isinstance(layer.mlp, Qwen3MoeMLP):
                gate_up_proj, down_proj = layer.mlp.gate_up_proj, layer.mlp.down_proj
            else:
                gate_up_proj, down_proj = None, None
            mark_sp_region(prev_down_proj, layer.self_attn.qkv_proj, pc)
            mark_sp_region(layer.self_attn.o_proj, gate_up_proj, pc)
            prev_down_proj = down_proj
        # Last region ends at the gathered final norm.
        mark_sp_region(prev_down_proj, None, pc)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if not is_sp_active(self.parallel_config):
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        if (
            self.parallel_config.use_sequence_parallel_moe
            and envs.VLLM_MOE_SKIP_PADDING
            and is_forward_context_available()
        ):
            # MoE-SP routers see the local shard, so shard the padding mask.
            forward_context = get_forward_context()
            forward_context.is_padding = sp_padding_mask(
                forward_context.is_padding, inputs_embeds
            )
        output = super().forward(
            input_ids, positions, intermediate_tensors, sp_shard(inputs_embeds)
        )
        num_tokens = positions.shape[-1]
        if isinstance(output, tuple):
            hidden_states, aux_hidden_states = output
            return sp_all_gather(hidden_states)[:num_tokens], [
                sp_all_gather(x)[:num_tokens] for x in aux_hidden_states
            ]
        return sp_all_gather(output)[:num_tokens]


class Qwen3MoeForCausalLM(_Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # `_Qwen3MoeForCausalLM.__init__` (upstream) itself builds
        # `self.model = Qwen3MoeModel(...)`, resolving `Qwen3MoeModel`
        # from `qwen3_moe.py`'s own module globals -- i.e. upstream's
        # plain model, not our subclass -- and then derives
        # `self.moe_layers`/`self.lm_head` tied weights from it.
        # Temporarily monkeypatch that global so upstream's `__init__`
        # constructs *our* `Qwen3MoeModel` in place, instead of
        # building an upstream one (registering attention layers under
        # the same prefix we'd reuse) and then rebuilding/discarding it.
        original_model_cls = _qwen3_moe.Qwen3MoeModel
        _qwen3_moe.Qwen3MoeModel = Qwen3MoeModel
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _qwen3_moe.Qwen3MoeModel = original_model_cls
