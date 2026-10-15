# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.sequence_parallel.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
)
from vllm.model_executor.kernels.sequence_parallel.fp8.utils import (
    as_scaled_mm_scales,
    resolve_fp8_linear_kernel,
)
from vllm.model_executor.layers.linear import LinearBase, RowParallelLinear


class FlashInferAllGatherScaledMM(FusedAllGatherGemmKernel):
    """Fused AllGather+GEMM for ``FlashInferFP8ScaledMMLinearKernel``.

    Patches ``apply_scaled_mm`` directly (same reasoning as
    ``CutlassAllGatherScaledMM``: for this kernel ``apply_weights`` is a
    thin wrapper that returns ``apply_scaled_mm``'s result unchanged).
    Per-tensor scaling has no block-shape constraint for a backend-native
    GEMM to avoid, so this reuses the same generic
    ``aten::_scaled_mm.out``-backed symm_mem op as ``TorchAllGatherScaledMM``
    -- only ``is_supported`` differs, to also match when FlashInfer's own
    per-tensor kernel is the one selected.

    Restricted to a static activation scale: it is identical (loaded from
    the checkpoint) on every rank, so the same ``As`` can be reused for
    every gathered shard regardless of which rank produced it.
    """

    def __init__(self, layer: LinearBase) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert isinstance(fp8_linear, FlashInferFP8ScaledMMLinearKernel)
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_scaled_mm

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if not isinstance(fp8_linear, FlashInferFP8ScaledMMLinearKernel):
            return False, "requires FlashInferFP8ScaledMMLinearKernel"
        if not fp8_linear.config.activation_quant_key.scale.static:
            return False, "requires a static (per-tensor) activation scale"
        return True, None

    def target_gemm_method(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        A = A[: output_shape[0]]
        As, Bs = as_scaled_mm_scales(A, B, As, Bs, tensorwise=True)
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            A,
            [B],
            As,
            [Bs],
            gather_dim=0,
            biases=[bias],
            result_scales=[None],
            out_dtypes=[out_dtype],
            use_fast_accum=[False],
            group_name=self._group_name,
        )
        return mm_outputs[0]


class FlashInferScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """Fused GEMM+ReduceScatter counterpart of
    ``FlashInferAllGatherScaledMM``.

    Reuses the same generic ``aten::_scaled_mm.out``-backed symm_mem op as
    ``TorchScaledMMReduceScatter`` for the same reason: per-tensor scaling
    has no block-shape constraint for a backend-native GEMM to avoid.
    """

    def __init__(self, layer: RowParallelLinear) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert isinstance(fp8_linear, FlashInferFP8ScaledMMLinearKernel)
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_scaled_mm

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if not isinstance(fp8_linear, FlashInferFP8ScaledMMLinearKernel):
            return False, "requires FlashInferFP8ScaledMMLinearKernel"
        if not fp8_linear.config.activation_quant_key.scale.static:
            return False, "requires a static (per-tensor) activation scale"
        return True, None

    def target_gemm_method(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        A = A[: output_shape[0]]
        As, Bs = as_scaled_mm_scales(A, B, As, Bs, tensorwise=True)
        output = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
            A,
            B,
            As,
            Bs,
            "sum",
            0,
            0,
            self._group_name,
            output_shape,
            None,
            None,
            out_dtype,
            False,
        )
        return output
