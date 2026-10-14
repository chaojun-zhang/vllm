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

import vllm.model_executor.models.qwen3_moe as _qwen3_moe
from vllm.compilation.decorators import ignore_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer as _Qwen3MoeDecoderLayer,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeForCausalLM as _Qwen3MoeForCausalLM,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeModel as _Qwen3MoeModel,
)

logger = init_logger(__name__)


class Qwen3MoeDecoderLayer(_Qwen3MoeDecoderLayer):
    """Identical to the upstream layer -- this vendor fork exists so
    sequence parallelism and other XPU-specific behavior can be layered
    on top without touching the upstream model."""


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
