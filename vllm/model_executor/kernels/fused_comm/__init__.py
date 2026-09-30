# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TypeVar

from vllm.logger import init_logger
from vllm.model_executor.kernels.fused_comm import unquantized
from vllm.model_executor.kernels.fused_comm.base import (
    AllGatherGemmKernel,
    GemmReduceScatterKernel,
)
from vllm.model_executor.kernels.fused_comm.fp8 import (
    POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS,
    POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS,
    POSSIBLE_MXFP8_ALL_GATHER_GEMM_KERNELS,
    POSSIBLE_MXFP8_GEMM_REDUCE_SCATTER_KERNELS,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
)
from vllm.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)

_AllGatherKernelT = TypeVar("_AllGatherKernelT", bound=AllGatherGemmKernel)
_ReduceScatterKernelT = TypeVar("_ReduceScatterKernelT", bound=GemmReduceScatterKernel)

POSSIBLE_UNQUANTIZED_ALL_GATHER_GEMM_KERNELS: dict[
    PlatformEnum, list[type[AllGatherGemmKernel]]
] = {
    PlatformEnum.XPU: [unquantized.AllGatherGemm],
}

POSSIBLE_UNQUANTIZED_GEMM_REDUCE_SCATTER_KERNELS: dict[
    PlatformEnum, list[type[GemmReduceScatterKernel]]
] = {
    PlatformEnum.XPU: [unquantized.GemmReduceScatter],
}


def _choose_fused_kernel(
    possible_kernels: dict[PlatformEnum, list[type[_AllGatherKernelT]]]
    | dict[PlatformEnum, list[type[_ReduceScatterKernelT]]],
    quant_method: QuantizeMethodBase,
    *extra_args: object,
) -> _AllGatherKernelT | _ReduceScatterKernelT | None:
    """Try each candidate registered for the current platform, in order,
    and construct (and return) the first one whose `is_supported` accepts
    `quant_method`. Mirrors `choose_scaled_mm_linear_kernel`'s pattern for
    selecting an underlying GEMM kernel.
    """
    for kernel_cls in possible_kernels.get(current_platform._enum, []):
        supported, reason = kernel_cls.is_supported(quant_method)
        if supported:
            return kernel_cls(quant_method, *extra_args)
        logger.debug("%s not available: %s", kernel_cls.__name__, reason)
    return None


def init_all_gather_gemm_kernel(
    linear: ColumnParallelLinear,
) -> AllGatherGemmKernel | None:
    """Build the fused AllGather+GEMM kernel for `linear`'s quant method,
    or return `None` if no backend supports its quant method + platform.

    The returned kernel has already installed itself onto its quant
    method's GEMM hook (see `AllGatherGemmKernel`); callers only ever
    need `set_fuse_gemm_comms`.
    """
    quant_method = linear.quant_method
    if isinstance(quant_method, UnquantizedLinearMethod):
        kernel = _choose_fused_kernel(
            POSSIBLE_UNQUANTIZED_ALL_GATHER_GEMM_KERNELS, quant_method
        )
    elif isinstance(quant_method, Fp8LinearMethod):
        kernel = _choose_fused_kernel(
            POSSIBLE_FP8_ALL_GATHER_GEMM_KERNELS, quant_method
        )
    elif isinstance(quant_method, Mxfp8OnlineLinearMethod):
        kernel = _choose_fused_kernel(
            POSSIBLE_MXFP8_ALL_GATHER_GEMM_KERNELS, quant_method
        )
    else:
        kernel = None

    if kernel is None:
        logger.debug("Fused AllGather+GEMM not available for %s", type(linear).__name__)
    return kernel


def init_gemm_reduce_scatter_kernel(
    linear: RowParallelLinear,
) -> GemmReduceScatterKernel | None:
    """Build the fused GEMM+ReduceScatter kernel for `linear`'s quant
    method, or return `None` if no backend supports its quant method +
    platform.

    The returned kernel has already installed itself onto its quant
    method's GEMM hook (see `GemmReduceScatterKernel`); callers only
    ever need `set_fuse_gemm_comms`.
    """
    quant_method = linear.quant_method
    if isinstance(quant_method, UnquantizedLinearMethod):
        kernel = _choose_fused_kernel(
            POSSIBLE_UNQUANTIZED_GEMM_REDUCE_SCATTER_KERNELS, quant_method, linear
        )
    elif isinstance(quant_method, Fp8LinearMethod):
        kernel = _choose_fused_kernel(
            POSSIBLE_FP8_GEMM_REDUCE_SCATTER_KERNELS, quant_method, linear
        )
    elif isinstance(quant_method, Mxfp8OnlineLinearMethod):
        kernel = _choose_fused_kernel(
            POSSIBLE_MXFP8_GEMM_REDUCE_SCATTER_KERNELS, quant_method, linear
        )
    else:
        kernel = None

    if kernel is None:
        logger.debug(
            "Fused GEMM+ReduceScatter not available for %s", type(linear).__name__
        )
    return kernel


__all__ = [
    "AllGatherGemmKernel",
    "GemmReduceScatterKernel",
    "init_all_gather_gemm_kernel",
    "init_gemm_reduce_scatter_kernel",
]
