# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (  # noqa: E501
    CompressedTensorsW8A8Fp8,
)


def resolve_fp8_linear_kernel(
    layer: LinearBase,
) -> FP8ScaledMMLinearKernel | None:
    """Return `layer.quant_method`'s per-layer FP8 GEMM kernel, or `None`
    if it isn't a quant method this module knows how to find one on (e.g.
    a non-FP8 compressed-tensors scheme)."""
    quant_method = layer.quant_method
    if isinstance(quant_method, CompressedTensorsLinearMethod):
        scheme = getattr(layer, "scheme", None)
        if isinstance(scheme, CompressedTensorsW8A8Fp8):
            return scheme.fp8_linear
        return None
    return getattr(quant_method, "fp8_linear", None)


def as_scaled_mm_scales(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    tensorwise: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay out `As`/`Bs` for the `aten::_scaled_mm` inside the symm_mem
    fused ops, which accepts only both-tensor-wise or both-row-wise scales.

    Tensor-wise is kept only when `tensorwise` and both scales are
    singletons; everything else (mixed granularity, or a dynamic per-tensor
    activation scale that differs across the gathered shards) is expanded
    to row-wise `As [M, 1]` / `Bs [1, N]`.
    """
    if tensorwise and As.numel() == 1 and Bs.numel() == 1:
        return As.reshape(1), Bs.reshape(1)
    As = As.reshape(-1, 1).expand(A.shape[0], 1).contiguous()
    Bs = Bs.reshape(1, -1).expand(1, B.shape[1]).contiguous()
    return As, Bs
