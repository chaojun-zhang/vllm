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
"""Inference-only Qwen3 model compatible with HuggingFace weights."""

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer as Qwen3DecoderLayerBase
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as Qwen3ForCausalLMBase
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import (
    is_sequence_parallel_active,
    sp_all_gather,
    sp_shard,
    suspend_sequence_parallel_boundary,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class Qwen3DecoderLayer(Qwen3DecoderLayerBase):
    """Identical to the upstream layer except:

    - Attention and MLP are SP compute regions. From each region's perspective,
      its input projection performs AllGather+GEMM and its output projection
      performs GEMM+ReduceScatter. Across regions, this is the standard
      ``ReduceScatter -> sharded residual/norm -> AllGather`` replacement for
      ``AllReduce -> residual/norm``.
    - ``forward`` shards the raw embeddings down to this rank's local
      token chunk right before the first layer runs (``residual is None``
      is only true for that one call) -- a no-op when SP isn't active for
      the in-flight call. The corresponding all-gather back up to full
      tokens happens once, at the model level, right after ``self.norm``
      (see ``Qwen3Model.forward``) -- not here, since gathering earlier
      would also require gathering ``residual`` just to keep it
      shape-matched for that fused add+norm, for no benefit.
    """

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        per_layer_sliding_window: int | None = None,
    ) -> None:
        super().__init__(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            per_layer_sliding_window=per_layer_sliding_window,
        )
        self.parallel_config = get_current_vllm_config().parallel_config
        if self.parallel_config.use_sequence_parallel:
            suspend_sequence_parallel_boundary(
                suspend_before=self.self_attn.qkv_proj,
                resume_after=self.self_attn.o_proj,
                parallel_config=self.parallel_config,
            )
            suspend_sequence_parallel_boundary(
                suspend_before=self.mlp.gate_up_proj,
                resume_after=self.mlp.down_proj,
                parallel_config=self.parallel_config,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None and is_sequence_parallel_active(self.parallel_config):
            hidden_states = sp_shard(hidden_states)
        return super().forward(positions, hidden_states, residual)


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )
        self.parallel_config = vllm_config.parallel_config
        self.use_sequence_parallel = self.parallel_config.use_sequence_parallel
        assert not (self.use_sequence_parallel and not get_pp_group().is_last_rank), (
            "Currently, SP is not supported with PP"
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        result = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        if not is_sequence_parallel_active(self.parallel_config):
            return result

        # `self.norm` (called inside `super().forward()`) already consumed
        # `residual` -- only the surviving `hidden_states` (and any aux
        # hidden states, which mirror it) need gathering back from this
        # rank's local shard to the full token count.
        full_num_tokens = positions.shape[0]
        if isinstance(result, tuple):
            hidden_states, aux_hidden_states = result
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
            return hidden_states, aux_hidden_states
        assert isinstance(result, torch.Tensor)
        return sp_all_gather(result)[:full_num_tokens]


class Qwen3ForCausalLM(Qwen3ForCausalLMBase):
    hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
