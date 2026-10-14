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
from vllm.model_executor.kernels.fused_comm.fp8 import (
    cutlass,
    flashinfer,
    pytorch,
    triton,
    xpu,
)
from vllm.model_executor.kernels.fused_comm.fp8.utils import resolve_fp8_linear_kernel
from vllm.platforms import PlatformEnum

POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS: dict[
    PlatformEnum, list[type[FusedAllGatherGemmKernel]]
] = {
    PlatformEnum.XPU: [
        xpu.XPUAllGatherScaledMM,
        xpu.XPUAllGatherBlockScaledMM,
        pytorch.TorchAllGatherScaledMM,
        pytorch.TorchAllGatherBlockScaledMM,
    ],
    PlatformEnum.CUDA: [
        flashinfer.FlashInferAllGatherScaledMM,
        cutlass.CutlassAllGatherScaledMM,
        cutlass.CutlassAllGatherBlockScaledMM,
        triton.TritonAllGatherBlockScaledMM,
        pytorch.TorchAllGatherScaledMM,
        pytorch.TorchAllGatherBlockScaledMM,
    ],
    PlatformEnum.ROCM: [
        pytorch.TorchAllGatherScaledMM,
        pytorch.TorchAllGatherBlockScaledMM,
    ],
}

POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS: dict[
    PlatformEnum, list[type[FusedGemmReduceScatterKernel]]
] = {
    PlatformEnum.XPU: [
        xpu.XPUScaledMMReduceScatter,
        xpu.XPUBlockScaledMMReduceScatter,
        pytorch.TorchScaledMMReduceScatter,
        pytorch.TorchBlockScaledMMReduceScatter,
    ],
    PlatformEnum.CUDA: [
        flashinfer.FlashInferScaledMMReduceScatter,
        cutlass.CutlassScaledMMReduceScatter,
        cutlass.CutlassBlockScaledMMReduceScatter,
        triton.TritonBlockScaledMMReduceScatter,
        pytorch.TorchScaledMMReduceScatter,
        pytorch.TorchBlockScaledMMReduceScatter,
    ],
    PlatformEnum.ROCM: [
        pytorch.TorchScaledMMReduceScatter,
        pytorch.TorchBlockScaledMMReduceScatter,
    ],
}

POSSIBLE_MXFP8_ALL_GATHER_GEMM_KERNELS: dict[
    PlatformEnum, list[type[FusedAllGatherGemmKernel]]
] = {
    PlatformEnum.XPU: [xpu.XPUAllGatherMXFP8],
}

POSSIBLE_MXFP8_GEMM_REDUCE_SCATTER_KERNELS: dict[
    PlatformEnum, list[type[FusedGemmReduceScatterKernel]]
] = {
    PlatformEnum.XPU: [xpu.XPUMXFP8ReduceScatter],
}


__all__ = [
    "POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS",
    "POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS",
    "POSSIBLE_MXFP8_ALL_GATHER_GEMM_KERNELS",
    "POSSIBLE_MXFP8_GEMM_REDUCE_SCATTER_KERNELS",
    "resolve_fp8_linear_kernel",
]
