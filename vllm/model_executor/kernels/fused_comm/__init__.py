# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8.xpu import XPUMxFp8LinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    BlockWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUFp8BlockScaledMMKernel,
    XPUW8A8FP8LinearKernel,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
)

from vllm.model_executor.kernels.fused_comm.base import (
    FusedAllGatherGemmKernel,
    FusedCommLinearKernel,
    FusedGemmReduceScatterKernel,
)

logger = init_logger(__name__)


def _init_ag_gemm_kernel(
    linear: ColumnParallelLinear,
) -> FusedAllGatherGemmKernel | None:
    """Build the fused AllGather+GEMM kernel for `linear`'s quant method,
    or return `None` if none is supported."""
    from vllm.model_executor.kernels.fused_comm import unquantized
    from vllm.model_executor.kernels.fused_comm.fp8 import torch as fp8_torch
    from vllm.model_executor.kernels.fused_comm.fp8 import xpu as fp8_xpu

    quant_method = linear.quant_method
    if isinstance(quant_method, UnquantizedLinearMethod):
        return unquantized.SymmMemAllGatherGemm(linear)
    if isinstance(quant_method, Fp8LinearMethod) and isinstance(
        quant_method.fp8_linear, XPUW8A8FP8LinearKernel
    ):
        return fp8_xpu.XPUSymmMemAllGatherScaledMM(linear, quant_method.fp8_linear)
    if isinstance(quant_method, Fp8LinearMethod) and isinstance(
        quant_method.fp8_linear,
        (PerTensorTorchFP8ScaledMMLinearKernel, RowWiseTorchFP8ScaledMMLinearKernel),
    ):
        return fp8_torch.TorchSymmMemAllGatherScaledMM(linear, quant_method.fp8_linear)
    if isinstance(quant_method, Fp8LinearMethod) and isinstance(
        quant_method.fp8_linear, BlockWiseTorchFP8ScaledMMLinearKernel
    ):
        return fp8_torch.TorchSymmMemAllGatherBlockScaledMM(
            linear, quant_method.fp8_linear
        )
    if isinstance(quant_method, Fp8LinearMethod) and isinstance(
        quant_method.fp8_linear, XPUFp8BlockScaledMMKernel
    ):
        return fp8_xpu.XPUSymmMemAllGatherBlockScaledMM(linear, quant_method.fp8_linear)
    if isinstance(quant_method, Mxfp8OnlineLinearMethod) and isinstance(
        quant_method.kernel, XPUMxFp8LinearKernel
    ):
        return fp8_xpu.XPUSymmMemAllGatherMXFP8(linear, quant_method.kernel)
    return None


def _init_gemm_rs_kernel(
    linear: RowParallelLinear,
) -> FusedGemmReduceScatterKernel | None:
    """Build the fused GEMM+ReduceScatter kernel for `linear`'s quant
    method, or return `None` if none is supported."""
    from vllm.model_executor.kernels.fused_comm import unquantized
    from vllm.model_executor.kernels.fused_comm.fp8 import torch as fp8_torch
    from vllm.model_executor.kernels.fused_comm.fp8 import xpu as fp8_xpu

    quant_method = linear.quant_method
    if isinstance(quant_method, UnquantizedLinearMethod):
        return unquantized.SymmMemGemmReduceScatter(linear)
    if isinstance(quant_method, Fp8LinearMethod) and isinstance(
        quant_method.fp8_linear, XPUW8A8FP8LinearKernel
    ):
        return fp8_xpu.XPUSymmMemScaledMMReduceScatter(linear, quant_method.fp8_linear)
    if isinstance(quant_method, Fp8LinearMethod) and isinstance(
        quant_method.fp8_linear,
        (PerTensorTorchFP8ScaledMMLinearKernel, RowWiseTorchFP8ScaledMMLinearKernel),
    ):
        return fp8_torch.TorchSymmMemScaledMMReduceScatter(
            linear, quant_method.fp8_linear
        )
    if isinstance(quant_method, Fp8LinearMethod) and isinstance(
        quant_method.fp8_linear, BlockWiseTorchFP8ScaledMMLinearKernel
    ):
        return fp8_torch.TorchSymmMemBlockScaledMMReduceScatter(
            linear, quant_method.fp8_linear
        )
    if isinstance(quant_method, Fp8LinearMethod) and isinstance(
        quant_method.fp8_linear, XPUFp8BlockScaledMMKernel
    ):
        return fp8_xpu.XPUSymmMemBlockScaledMMReduceScatter(
            linear, quant_method.fp8_linear
        )
    if isinstance(quant_method, Mxfp8OnlineLinearMethod) and isinstance(
        quant_method.kernel, XPUMxFp8LinearKernel
    ):
        return fp8_xpu.XPUSymmMemMXFP8ReduceScatter(linear, quant_method.kernel)
    return None


def init_fused_comm_kernel(
    linear: ColumnParallelLinear | RowParallelLinear,
) -> FusedCommLinearKernel | None:
    """Build a fused Comm+GEMM kernel for `linear`, or return `None` if no
    backend supports its boundary direction + quant method + platform.

    Each direction's quant method dispatch is a plain if-chain in
    `_init_ag_gemm_kernel` / `_init_gemm_rs_kernel` -- adding a new quant
    method or backend means adding a branch there.
    """
    if isinstance(linear, ColumnParallelLinear):
        kernel = _init_ag_gemm_kernel(linear)
    elif isinstance(linear, RowParallelLinear):
        kernel = _init_gemm_rs_kernel(linear)
    else:
        kernel = None

    if kernel is None:
        logger.debug(
            "Fused Comms+GEMM not available for %s",
            type(linear).__name__,
        )
        return None

    supported, reason = kernel.is_supported()
    if not supported:
        logger.debug("Fused Comms+GEMM not available: %s", reason)
        return None
    return kernel


__all__ = [
    "FusedCommLinearKernel",
    "init_fused_comm_kernel",
]
