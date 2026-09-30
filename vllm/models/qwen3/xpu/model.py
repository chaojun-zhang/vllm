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

from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM as Qwen3ForCausalLMBase
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import mark_sp_region

logger = init_logger(__name__)


class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )
        self.parallel_config = vllm_config.parallel_config
        if self.parallel_config.use_sequence_parallel:
            layers = self.layers[self.start_layer : self.end_layer]
            for i, layer in enumerate(layers):
                # Region 1 -- after attention: o_proj -> post_attention_layernorm
                # -> gate_up_proj. ReduceScatter -> sharded residual/norm ->
                # AllGather, replacing AllReduce + norm.
                mark_sp_region(
                    entry_after=layer.self_attn.o_proj,
                    exit_before=layer.mlp.gate_up_proj,
                    parallel_config=self.parallel_config,
                )
                if i + 1 < len(layers):
                    # Region 2 -- after MLP: down_proj -> next layer's
                    # input_layernorm -> next layer's qkv_proj. The last
                    # layer has no next qkv_proj, so its down_proj and the
                    # final norm run on full tokens instead.
                    mark_sp_region(
                        entry_after=layer.mlp.down_proj,
                        exit_before=layers[i + 1].self_attn.qkv_proj,
                        parallel_config=self.parallel_config,
                    )


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
