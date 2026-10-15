# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.kernels.sequence_parallel.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
)
from vllm.model_executor.kernels.sequence_parallel.fp8 import (
    cutlass,
    flashinfer,
    triton,
)
from vllm.model_executor.kernels.sequence_parallel.fp8.utils import (
    resolve_fp8_linear_kernel,
)
from vllm.platforms import PlatformEnum

POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS: dict[
    PlatformEnum, list[type[FusedAllGatherGemmKernel]]
] = {
    PlatformEnum.CUDA: [
        flashinfer.FlashInferAllGatherScaledMM,
        cutlass.CutlassAllGatherScaledMM,
        cutlass.CutlassAllGatherBlockScaledMM,
        triton.TritonAllGatherBlockScaledMM,
    ],
}

POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS: dict[
    PlatformEnum, list[type[FusedGemmReduceScatterKernel]]
] = {
    PlatformEnum.CUDA: [
        flashinfer.FlashInferScaledMMReduceScatter,
        cutlass.CutlassScaledMMReduceScatter,
        cutlass.CutlassBlockScaledMMReduceScatter,
    ],
}

__all__ = [
    "POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS",
    "POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS",
    "resolve_fp8_linear_kernel",
]
