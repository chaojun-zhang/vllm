# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed._symmetric_memory import (
    _pipelined_multi_all_gather_and_consume,
    enable_symm_mem_for_group,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
    CutlassFP8ScaledMMLinearKernel,
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


class CutlassAllGatherBlockScaledMM(FusedAllGatherGemmKernel):
    """Fused AllGather+GEMM for ``CutlassFp8BlockScaledMMKernel``.

    Pipelines the AllGather with a CUTLASS group-scaled GEMM per shard, the
    same way ``TorchAllGatherBlockScaledMM``/``XPUAllGatherBlockScaledMM``
    pipeline it with ``aten::_scaled_mm``. The quantized activation scale
    arrives column-major (as CUTLASS requires), but
    ``_pipelined_multi_all_gather_and_consume`` writes each shard into a
    plain contiguous buffer, so the column-major layout is rebuilt per shard
    before the GEMM.
    """

    def __init__(self, layer: LinearBase) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert isinstance(fp8_linear, CutlassFp8BlockScaledMMKernel)
        self._fp8_kernel = fp8_linear
        self._group = get_tp_group()
        enable_symm_mem_for_group(self._group.device_group.group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if not isinstance(fp8_linear, CutlassFp8BlockScaledMMKernel):
            return False, "requires CutlassFp8BlockScaledMMKernel"
        return True, None

    def target_gemm_method(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        fp8_kernel = self._fp8_kernel
        params = fp8_kernel._get_layer_params(layer)
        x_2d = x.view(-1, x.shape[-1])
        x_q, a_scale = fp8_kernel.quant_fp8(
            x_2d, params.input_scale, params.input_scale_ub
        )
        out_dtype = fp8_kernel.config.out_dtype

        world_size = self._group.world_size
        output = x_q.new_empty(
            x_q.shape[0] * world_size, params.weight.shape[0], dtype=out_dtype
        )
        output_shards = output.chunk(world_size)
        A = x_q.new_empty(x_q.shape[0] * world_size, x_q.shape[1])
        A_scale = a_scale.new_empty(a_scale.shape[0] * world_size, a_scale.shape[1])

        def shard_consumer(shards: list[torch.Tensor], rank: int) -> None:
            a, a_s = shards
            # Rebuild the column-major scale layout CUTLASS requires; the
            # gathered shard is a plain row-major view into A_scale.
            a_s = a_s.t().contiguous().t()
            out = fp8_kernel.apply_block_scaled_mm(
                A=a, B=params.weight, As=a_s, Bs=params.block_scale
            )
            output_shards[rank].copy_(out)

        _pipelined_multi_all_gather_and_consume(
            [x_q, a_scale.contiguous()],
            shard_consumer,
            [A, A_scale],
            self._group.device_group.group_name,
            False,
        )
        if bias is not None:
            output = output + bias
        return output


class CutlassBlockScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """Fused GEMM+ReduceScatter counterpart of
    ``CutlassAllGatherBlockScaledMM``.

    Like ``TorchBlockScaledMMReduceScatter``/``XPUBlockScaledMMReduceScatter``,
    this goes through the public ``fused_scaled_matmul_reduce_scatter`` symm_mem
    op (which pipelines ``aten::_scaled_mm`` internally -- there is no public
    pipelined-reduce-scatter primitive that plugs in a custom GEMM op the way
    ``_pipelined_multi_all_gather_and_consume`` does for AllGather). The op only
    consumes the raw quantized activation/scale tensors, so it is correct
    regardless of which block-scaled kernel produced them.
    """

    def __init__(self, layer: RowParallelLinear) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert isinstance(fp8_linear, CutlassFp8BlockScaledMMKernel)
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if not isinstance(fp8_linear, CutlassFp8BlockScaledMMKernel):
            return False, "requires CutlassFp8BlockScaledMMKernel"
        return True, None

    def target_gemm_method(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        fp8_kernel = self._fp8_kernel
        x_2d = x.view(-1, x.shape[-1])
        params = fp8_kernel._get_layer_params(layer)
        w = params.weight.t()
        w_scale = params.block_scale.t()
        x_q, a_scale = fp8_kernel.quant_fp8(
            x_2d, params.input_scale, params.input_scale_ub
        )
        out_dtype = fp8_kernel.config.out_dtype

        output_shape = [*x.shape[:-1], params.weight.shape[0]]
        output = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
            x_q,
            w,
            a_scale,
            w_scale,
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


class CutlassAllGatherScaledMM(FusedAllGatherGemmKernel):
    """Fused AllGather+GEMM for ``CutlassFP8ScaledMMLinearKernel`` (plain
    static per-tensor W8A8 FP8 -- not block-quantized).

    Patches ``apply_scaled_mm`` directly rather than the outer
    ``apply_weights``: for this kernel ``apply_weights`` does nothing after
    calling ``apply_scaled_mm`` but return its result, so the two are
    equivalent patch points (unlike the block-scaled kernels, where
    ``apply_weights`` does its own post-GEMM reshape/bias-add). Per-tensor
    scaling has none of the 128x128/column-major block-scale constraints
    that motivate a backend-native GEMM for the block-scaled kernels above,
    so this reuses the same generic ``aten::_scaled_mm.out``-backed
    symm_mem op as ``TorchAllGatherScaledMM`` -- only ``is_supported``
    differs, to also match when CUTLASS's own per-tensor kernel is the one
    selected.

    Restricted to a static activation scale: it is identical (loaded from
    the checkpoint) on every rank, so the same ``As`` can be reused for
    every gathered shard regardless of which rank produced it.
    """

    def __init__(self, layer: LinearBase) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert isinstance(fp8_linear, CutlassFP8ScaledMMLinearKernel)
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_scaled_mm

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if not isinstance(fp8_linear, CutlassFP8ScaledMMLinearKernel):
            return False, "requires CutlassFP8ScaledMMLinearKernel"
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


class CutlassScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """Fused GEMM+ReduceScatter counterpart of ``CutlassAllGatherScaledMM``.

    Reuses the same generic ``aten::_scaled_mm.out``-backed symm_mem op as
    ``TorchScaledMMReduceScatter`` for the same reason: per-tensor scaling
    has no block-shape constraint for a backend-native GEMM to avoid.
    """

    def __init__(self, layer: RowParallelLinear) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert isinstance(fp8_linear, CutlassFP8ScaledMMLinearKernel)
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_scaled_mm

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if not isinstance(fp8_linear, CutlassFP8ScaledMMLinearKernel):
            return False, "requires CutlassFP8ScaledMMLinearKernel"
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
