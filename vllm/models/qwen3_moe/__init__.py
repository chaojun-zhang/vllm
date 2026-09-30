# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3MoE model — vendor-specific entry point.

Dispatches to ``xpu/`` based on the current platform, otherwise falls
back to the upstream model in vllm.model_executor.models.
"""

from vllm.platforms import current_platform

if current_platform.is_xpu():
    from .xpu.model import Qwen3MoeForCausalLM  # type: ignore[assignment]
else:
    from vllm.model_executor.models.qwen3_moe import (  # type: ignore[assignment]
        Qwen3MoeForCausalLM,
    )

__all__ = [
    "Qwen3MoeForCausalLM",
]
