# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.model_executor.kernels.fused_comm.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
    LinearCommKernel,
    UnfusedAllGatherGemmKernel,
    UnfusedGemmReduceScatterKernel,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)


def _try_kernels(
    linear: ColumnParallelLinear | RowParallelLinear,
    kernels: list[type[LinearCommKernel]],
) -> LinearCommKernel | None:
    for kernel_cls in kernels:
        supported, reason = kernel_cls.is_supported(linear)
        if supported:
            return kernel_cls(linear)
        logger.debug("%s not available: %s", kernel_cls.__name__, reason)
    return None


def _try_fused_kernels(
    linear: ColumnParallelLinear | RowParallelLinear,
    ag_kernels: dict[PlatformEnum, list[type[FusedAllGatherGemmKernel]]],
    rs_kernels: dict[PlatformEnum, list[type[FusedGemmReduceScatterKernel]]],
) -> LinearCommKernel | None:
    kernels = rs_kernels if isinstance(linear, RowParallelLinear) else ag_kernels
    return _try_kernels(linear, list(kernels.get(current_platform._enum, [])))


def _init_unquantized_fused_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel | None:
    if not isinstance(linear.quant_method, UnquantizedLinearMethod):
        return None
    from vllm.model_executor.kernels.fused_comm.unquantized import (
        UnquantizedAllGatherGemm,
        UnquantizedGemmReduceScatter,
    )

    if isinstance(linear, RowParallelLinear):
        return UnquantizedGemmReduceScatter(linear)
    return UnquantizedAllGatherGemm(linear)


def _init_fp8_fused_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel | None:
    from vllm.model_executor.kernels.fused_comm.fp8 import (
        POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS,
        POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS,
    )

    return _try_fused_kernels(
        linear,
        POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS,
        POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS,
    )


def _init_mxfp8_fused_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel | None:
    from vllm.model_executor.kernels.fused_comm.fp8 import (
        POSSIBLE_MXFP8_ALL_GATHER_GEMM_KERNELS,
        POSSIBLE_MXFP8_GEMM_REDUCE_SCATTER_KERNELS,
    )
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
    )

    if not isinstance(linear.quant_method, Mxfp8OnlineLinearMethod):
        return None
    return _try_fused_kernels(
        linear,
        POSSIBLE_MXFP8_ALL_GATHER_GEMM_KERNELS,
        POSSIBLE_MXFP8_GEMM_REDUCE_SCATTER_KERNELS,
    )


def init_fused_comm_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel | None:
    """Return a fused (symmetric-memory) comm+GEMM kernel, or ``None``."""
    for init_kernel_fn in (
        _init_unquantized_fused_kernel,
        _init_fp8_fused_kernel,
        _init_mxfp8_fused_kernel,
    ):
        kernel = init_kernel_fn(linear)
        if kernel is not None:
            return kernel
    logger.debug("Fused comm+GEMM not available for %s", type(linear).__name__)
    return None


def init_unfused_comm_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> LinearCommKernel:
    """Return a comm+GEMM kernel built on plain collectives."""
    if isinstance(linear, RowParallelLinear):
        return UnfusedGemmReduceScatterKernel(linear)

    from vllm.model_executor.kernels.fused_comm.fp8.unfused import (
        UnfusedBlockScaledMMAllGather,
    )

    kernel = _try_kernels(linear, [UnfusedBlockScaledMMAllGather])
    return kernel or UnfusedAllGatherGemmKernel(linear)


__all__ = [
    "LinearCommKernel",
    "FusedAllGatherGemmKernel",
    "FusedGemmReduceScatterKernel",
    "UnfusedAllGatherGemmKernel",
    "UnfusedGemmReduceScatterKernel",
    "init_fused_comm_kernel",
    "init_unfused_comm_kernel",
]
