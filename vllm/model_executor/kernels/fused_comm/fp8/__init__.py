# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Comm+GEMM kernels for FP8-quantized linear layers, split by
backend: ``xpu`` (oneDNN ``_xpu_C`` ops, falling back to ``torch._scaled_mm``
where supported) and ``cuda`` (CUDA/ROCm, via ``torch._scaled_mm``-only
``aten::_scaled_mm`` symm_mem fused ops)."""

from vllm.model_executor.kernels.fused_comm.base import (
    AllGatherGemmKernel,
    GemmReduceScatterKernel,
)
from vllm.model_executor.kernels.fused_comm.fp8 import torch, xpu
from vllm.platforms import PlatformEnum

POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS: dict[
    PlatformEnum, list[type[AllGatherGemmKernel]]
] = {
    PlatformEnum.XPU: [
        xpu.XPUAllGatherScaledMM,
        xpu.XPUAllGatherBlockScaledMM,
        torch.TorchAllGatherScaledMM,
        torch.TorchAllGatherBlockScaledMM,
    ],
    PlatformEnum.CUDA: [
        torch.TorchAllGatherScaledMM,
        torch.TorchAllGatherBlockScaledMM,
    ],
    PlatformEnum.ROCM: [
        torch.TorchAllGatherScaledMM,
        torch.TorchAllGatherBlockScaledMM,
    ],
}

POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS: dict[
    PlatformEnum, list[type[GemmReduceScatterKernel]]
] = {
    PlatformEnum.XPU: [
        xpu.XPUScaledMMReduceScatter,
        xpu.XPUBlockScaledMMReduceScatter,
        torch.TorchScaledMMReduceScatter,
        torch.TorchBlockScaledMMReduceScatter,
    ],
    PlatformEnum.CUDA: [
        torch.TorchScaledMMReduceScatter,
        torch.TorchBlockScaledMMReduceScatter,
    ],
    PlatformEnum.ROCM: [
        torch.TorchScaledMMReduceScatter,
        torch.TorchBlockScaledMMReduceScatter,
    ],
}

POSSIBLE_MXFP8_ALL_GATHER_GEMM_KERNELS: dict[
    PlatformEnum, list[type[AllGatherGemmKernel]]
] = {
    PlatformEnum.XPU: [xpu.XPUAllGatherMXFP8],
}

POSSIBLE_MXFP8_GEMM_REDUCE_SCATTER_KERNELS: dict[
    PlatformEnum, list[type[GemmReduceScatterKernel]]
] = {
    PlatformEnum.XPU: [xpu.XPUMXFP8ReduceScatter],
}
