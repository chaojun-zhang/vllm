# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Comm+GEMM kernels for FP8-quantized linear layers, split by
backend: ``xpu`` (oneDNN ``_xpu_C`` ops, falling back to ``torch._scaled_mm``
where supported) and ``cuda`` (CUDA/ROCm, via ``torch._scaled_mm``-only
``aten::_scaled_mm`` symm_mem fused ops)."""

from vllm.model_executor.kernels.fused_comm.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
)
from vllm.model_executor.kernels.fused_comm.fp8 import flashinfer
from vllm.model_executor.kernels.fused_comm.fp8.utils import resolve_fp8_linear_kernel
from vllm.platforms import PlatformEnum

POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS: dict[
    PlatformEnum, list[type[FusedAllGatherGemmKernel]]
] = {
    PlatformEnum.CUDA: [
        flashinfer.FlashInferAllGatherScaledMM,
    ],
}

POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS: dict[
    PlatformEnum, list[type[FusedGemmReduceScatterKernel]]
] = {
    PlatformEnum.CUDA: [
        flashinfer.FlashInferScaledMMReduceScatter,
    ],
}

__all__ = [
    "POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS",
    "POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS",
    "resolve_fp8_linear_kernel",
]
