# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    xpu_mxfp8_quantize as quant_mxfp8,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class XPUMxFp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM on XPU."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUMxFp8 only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_scale = layer.weight_scale.view(torch.float8_e8m0fnu)
        # Keep weight_scale in [N, K/32] format (not transposed).
        # The oneDNN fp8_gemm kernel expects weight_scale with shape [N, K/32],
        # matching activation_scale shape [M, K/32] on the K/32 dimension.
        # Store weight as C-contiguous [K, N] so its data_ptr is GPU-allocator
        # aligned (64-byte).  Without .contiguous() the stored tensor is a
        # Fortran-order view sharing the checkpoint's potentially-misaligned
        # storage; that triggers oneDNN alignment checks during inference.
        # This one-time copy at load time avoids any copy at inference time.
        replace_parameter(layer, "weight", layer.weight.t().contiguous())
        replace_parameter(layer, "weight_scale", weight_scale.contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        x_fp8, x_scale = quant_mxfp8(x)
        return torch.ops._xpu_C.fp8_gemm(
            x_fp8,
            layer.weight,
            out_dtype,
            x_scale,
            layer.weight_scale,
            bias,
        )
