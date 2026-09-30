# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed._symmetric_memory import (
    _pipelined_multi_all_gather_and_consume,
    enable_symm_mem_for_group,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.linear.mxfp8.xpu import XPUMxFp8LinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUFp8BlockScaledMMKernel,
    XPUW8A8FP8LinearKernel,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    xpu_mxfp8_quantize as quant_mxfp8,
)
from vllm.platforms import current_platform

from vllm.model_executor.kernels.fused_comm.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
)


def _as_scaled_mm_weight_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """Reshape a per-channel weight scale to the ``[1, N]`` row-major layout
    required by ``aten::_scaled_mm`` (used internally by the ``symm_mem``
    fused ops).

    vLLM stores per-channel FP8 weight scales as ``[N, 1]``
    (``ChannelQuantScaleParameter``, ``output_dim=0``), whereas
    ``aten::_scaled_mm``'s rowwise-scaling path requires ``scale_b`` to be a
    contiguous ``[1, N]`` tensor. ``reshape`` (not ``.t()``) is required here:
    the data is already a contiguous flat buffer of ``N`` scales, so
    ``reshape(1, -1)`` yields a genuinely contiguous ``[1, N]`` view, while
    ``.t()`` would only produce a non-contiguous ``[1, N]`` tensor that fails
    ``aten::_scaled_mm``'s ``scale_b must be contiguous`` check.

    Per-tensor scales (``numel() == 1``) are returned unchanged, since the
    tensorwise-scaling path has no shape requirement beyond ``numel() == 1``.
    """
    if weight_scale.numel() == 1:
        return weight_scale
    return weight_scale.reshape(1, -1)


def _to_scaled_mm_scales(
    m: int, n: int, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize ``_xpu_C.fp8_gemm``'s scale shapes to what
    ``torch._scaled_mm`` expects.

    ``fp8_gemm`` accepts ``scale_a`` as a singleton (per-tensor), ``[M, 1]``
    (per-token), or ``[M, k_blocks]`` (block-quantized); ``scale_b`` as a
    singleton, ``[N]``/``[1, N]`` (per-channel), or ``[k_blocks, n_blocks]``
    (block-quantized). Block-quantized scales are already in the
    ``[M, k_blocks]`` / ``[k_blocks, n_blocks]`` layout ``torch._scaled_mm``'s
    block-wise mode expects, so they pass through as-is. ``torch._scaled_mm``
    additionally requires tensor-wise scales to be singletons on *both*
    sides, so a per-tensor scale on one operand is broadcast to match a
    per-token/per-channel scale on the other.
    """
    a_blockwise = scale_a.dim() >= 2 and scale_a.shape[-1] > 1
    b_blockwise = scale_b.dim() >= 2 and scale_b.shape[0] > 1
    if a_blockwise or b_blockwise:
        return scale_a, scale_b

    a_scalar = scale_a.numel() == 1
    b_scalar = scale_b.numel() == 1
    if a_scalar and b_scalar:
        return scale_a.reshape(1), scale_b.reshape(1, 1)
    # expand() produces a stride-0 view, which torch._scaled_mm rejects for
    # rowwise scales ("both should be contiguous"), so .contiguous() here is
    # required, unlike the pre-contiguous block scales above.
    scale_a = (
        scale_a.reshape(-1, 1) if not a_scalar else scale_a.expand(m, 1).contiguous()
    )
    scale_b = (
        scale_b.reshape(1, -1)
        if not b_scalar
        else scale_b.expand(n).reshape(1, n).contiguous()
    )
    return scale_a, scale_b


def _xpu_fp8_mm_out(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    """FP8 GEMM used by the XPU fused Comm+GEMM kernels below.

    Thin wrapper around ``torch._scaled_mm``, which on XPU dispatches to
    tensor-wise, row-wise, or block-wise (including MXFP8's
    ``float8_e8m0fnu``/32-element groups) internally based on scale
    shape/dtype; ``_to_scaled_mm_scales`` only normalizes the scale shapes
    ``fp8_gemm``-style callers pass in to what it expects.
    """
    scale_a_mm, scale_b_mm = _to_scaled_mm_scales(
        A.shape[0], B.shape[1], scale_a, scale_b
    )
    torch._scaled_mm(
        A,
        B,
        scale_a=scale_a_mm,
        scale_b=scale_b_mm,
        bias=bias,
        out_dtype=out.dtype,
        out=out,
    )


class XPUSymmMemAllGatherScaledMM(FusedAllGatherGemmKernel):
    """FP8 W8A8 fused AllGather+GEMM using torch symmetric memory.

    Requires the underlying FP8 kernel to be ``XPUW8A8FP8LinearKernel``:
    ``aten::_scaled_mm`` requires its ``mat2`` (weight) operand to be
    column-major ``[K, N]``, and ``XPUW8A8FP8LinearKernel
    .process_weights_after_loading`` happens to already produce weight in
    this layout (a transposed view of the ``[N, K]`` checkpoint tensor), so
    no extra transpose is needed here. The per-channel weight scale,
    however, is stored as ``[N, 1]`` and must be reshaped to ``[1, N]`` (see
    ``_as_scaled_mm_weight_scale``) to satisfy ``aten::_scaled_mm``'s
    rowwise-scaling shape/contiguity requirements.
    """

    def __init__(
        self, layer: ColumnParallelLinear, fp8_linear: XPUW8A8FP8LinearKernel
    ) -> None:
        super().__init__(layer)
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)

    def is_supported(self) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "symmetric memory requires XPU"
        return True, None

    def apply(
        self,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x_shard.view(-1, x_shard.shape[-1])
        w, w_scale, x_scale, x_scale_ub = self._fp8_kernel._get_layer_params(self.layer)
        w_scale = _as_scaled_mm_weight_scale(w_scale)
        x_q, a_scale = self._fp8_kernel.quant_fp8(x_2d, x_scale, x_scale_ub)
        out_dtype = self._fp8_kernel.config.out_dtype

        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            x_q,
            [w],
            a_scale,
            [w_scale],
            gather_dim=0,
            biases=[bias],
            result_scales=[None],
            out_dtypes=[out_dtype],
            use_fast_accum=[False],
            group_name=self._group_name,
        )
        return mm_outputs[0]


class XPUSymmMemScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """FP8 W8A8 fused GEMM+ReduceScatter using torch symmetric memory.

    See ``XPUSymmMemAllGatherScaledMM`` for the weight/scale layout
    requirements shared by both directions.

    ReduceScatter sums every rank's partial GEMM contribution for a given
    output shard, so bias must *not* be passed into the GEMM (it would be
    added once per summed contribution, i.e. ``world_size`` times total);
    the GEMM always runs with ``bias=None`` and the real bias is added to
    the output once, afterwards.
    """

    def __init__(
        self, layer: RowParallelLinear, fp8_linear: XPUW8A8FP8LinearKernel
    ) -> None:
        super().__init__(layer)
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)

    def is_supported(self) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "symmetric memory requires XPU"
        return True, None

    def apply(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[-1])
        w, w_scale, x_scale, x_scale_ub = self._fp8_kernel._get_layer_params(self.layer)
        w_scale = _as_scaled_mm_weight_scale(w_scale)
        x_q, a_scale = self._fp8_kernel.quant_fp8(x_2d, x_scale, x_scale_ub)
        out_dtype = self._fp8_kernel.config.out_dtype

        output_shape = [*x.shape[:-1], w.shape[1]]
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
        if bias is not None:
            output = output + bias
        return output


class XPUSymmMemAllGatherBlockScaledMM(FusedAllGatherGemmKernel):
    """FP8 block-scaled fused AllGather+GEMM for ``XPUFp8BlockScaledMMKernel``.

    Block-FP8 activation quantization is dynamic (per-token-group, computed
    from each rank's own local shard -- see
    ``Fp8BlockScaledMMLinearKernel.can_implement``), so both the quantized
    activation *and* its scale differ per rank and must be gathered
    together, not just the activation. This uses
    ``_pipelined_multi_all_gather_and_consume`` directly (the same private
    primitive the public ``fused_all_gather_scaled_matmul`` op is built on)
    to jointly gather ``(x_q, a_scale)`` and run one local ``_xpu_fp8_mm_out``
    per peer shard as it arrives, giving the same compute/communication
    overlap as the other symm_mem-based kernels in this package.

    ``XPUFp8BlockScaledMMKernel.apply_block_scaled_mm`` always applies bias
    *outside* the GEMM (passing an empty placeholder tensor into
    ``_xpu_C.fp8_gemm``), so this kernel matches that convention: the GEMM
    itself always runs with ``bias=None``, and the real bias is added to
    the output afterwards.
    """

    def __init__(
        self,
        layer: ColumnParallelLinear,
        fp8_linear: XPUFp8BlockScaledMMKernel,
    ) -> None:
        super().__init__(layer)
        self._fp8_kernel = fp8_linear
        self._group = get_tp_group()
        enable_symm_mem_for_group(self._group.device_group.group_name)

    def is_supported(self) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "requires XPU"
        return True, None

    def apply(
        self,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x_shard.view(-1, x_shard.shape[-1])
        params = self._fp8_kernel._get_layer_params(self.layer)
        x_q, a_scale = self._fp8_kernel.quant_fp8(
            x_2d, params.input_scale, params.input_scale_ub
        )
        weight_t = params.weight.t()  # [N, K] -> [K, N] view
        weight_scale_t = params.block_scale.t()  # [n_blocks, k_blocks] view
        out_dtype = self._fp8_kernel.config.out_dtype

        world_size = self._group.world_size
        output = x_q.new_empty(
            x_q.shape[0] * world_size, weight_t.shape[1], dtype=out_dtype
        )
        output_shards = output.chunk(world_size)
        A = x_q.new_empty(x_q.shape[0] * world_size, x_q.shape[1])
        A_scale = a_scale.new_empty(a_scale.shape[0] * world_size, a_scale.shape[1])

        def shard_consumer(shards: list[torch.Tensor], rank: int) -> None:
            _xpu_fp8_mm_out(
                shards[0],
                weight_t,
                scale_a=shards[1],
                scale_b=weight_scale_t,
                out=output_shards[rank],
                bias=None,
            )

        _pipelined_multi_all_gather_and_consume(
            [x_q, a_scale],
            shard_consumer,
            [A, A_scale],
            self._group.device_group.group_name,
            False,
        )
        if bias is not None:
            output = output + bias
        return output


class XPUSymmMemBlockScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """FP8 block-scaled fused GEMM+ReduceScatter counterpart of
    ``XPUSymmMemAllGatherBlockScaledMM``.

    Unlike the AllGather direction, the public
    ``fused_scaled_matmul_reduce_scatter`` op's scale check only requires
    ``A_scale.shape[:-1] == A.shape[:-1]`` (unconstrained trailing dim),
    which the block scale's ``[M, ceil(K/128)]`` layout already satisfies.
    Since ``torch._scaled_mm`` natively handles block-wise scales on XPU (no
    oneDNN fallback needed), this can call the public
    ``patched_fused_scaled_matmul_reduce_scatter`` wrapper directly, just
    like ``TorchSymmMemBlockScaledMMReduceScatter`` -- no private primitive
    or custom ``mm_out_op`` required.
    """

    def __init__(
        self,
        layer: RowParallelLinear,
        fp8_linear: XPUFp8BlockScaledMMKernel,
    ) -> None:
        super().__init__(layer)
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)

    def is_supported(self) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "requires XPU"
        return True, None

    def apply(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[-1])
        params = self._fp8_kernel._get_layer_params(self.layer)
        x_q, a_scale = self._fp8_kernel.quant_fp8(
            x_2d, params.input_scale, params.input_scale_ub
        )
        weight_t = params.weight.t()
        weight_scale_t = params.block_scale.t()
        out_dtype = self._fp8_kernel.config.out_dtype

        output_shape = [*x.shape[:-1], params.weight.shape[0]]
        output = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
            x_q,
            weight_t,
            a_scale,
            weight_scale_t,
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
        if bias is not None:
            output = output + bias
        return output


class XPUSymmMemAllGatherMXFP8(FusedAllGatherGemmKernel):
    """MXFP8 fused AllGather+GEMM for ``XPUMxFp8LinearKernel``.

    Structurally identical to ``XPUSymmMemAllGatherBlockScaledMM`` (MXFP8
    activation quantization is also dynamic/per-rank), with two differences
    driven by ``XPUMxFp8LinearKernel.apply_weights``: the bias is passed
    straight into the GEMM (not added afterwards), and the ``e8m0`` scale
    dtype must be bitcast to ``uint8`` around the collective -- the
    collective backend (e.g. XCCL) does not support ``float8_e8m0fnu``
    directly, so the scale is gathered as its bitwise-identical unsigned
    view and viewed back before the GEMM.
    """

    def __init__(
        self,
        layer: ColumnParallelLinear,
        fp8_linear: XPUMxFp8LinearKernel,
    ) -> None:
        super().__init__(layer)
        self._fp8_kernel = fp8_linear
        self._group = get_tp_group()
        enable_symm_mem_for_group(self._group.device_group.group_name)

    def is_supported(self) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "requires XPU"
        return True, None

    def apply(
        self,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x_shard.view(-1, x_shard.shape[-1])
        out_dtype = x_2d.dtype
        x_q, a_scale = quant_mxfp8(x_2d)
        weight_t = self.layer.weight.t()  # [N, K] -> [K, N] view
        weight_scale_t = self.layer.weight_scale.t()  # [k_blocks, n_blocks] view

        scale_dtype = a_scale.dtype
        a_scale_u8 = a_scale.view(torch.uint8)

        world_size = self._group.world_size
        output = x_q.new_empty(
            x_q.shape[0] * world_size, weight_t.shape[1], dtype=out_dtype
        )
        output_shards = output.chunk(world_size)
        A = x_q.new_empty(x_q.shape[0] * world_size, x_q.shape[1])
        A_scale = a_scale_u8.new_empty(
            a_scale_u8.shape[0] * world_size, a_scale_u8.shape[1]
        )

        def shard_consumer(shards: list[torch.Tensor], rank: int) -> None:
            _xpu_fp8_mm_out(
                shards[0],
                weight_t,
                scale_a=shards[1].view(scale_dtype),
                scale_b=weight_scale_t,
                out=output_shards[rank],
                bias=bias,
            )

        _pipelined_multi_all_gather_and_consume(
            [x_q, a_scale_u8],
            shard_consumer,
            [A, A_scale],
            self._group.device_group.group_name,
            False,
        )
        return output


class XPUSymmMemMXFP8ReduceScatter(FusedGemmReduceScatterKernel):
    """MXFP8 fused GEMM+ReduceScatter counterpart of
    ``XPUSymmMemAllGatherMXFP8``.

    As with the block-FP8 ReduceScatter direction, the public
    ``fused_scaled_matmul_reduce_scatter`` op's scale check has no
    restriction on the scale's trailing dim, and its scale handling is
    dtype-agnostic (shape-based only), so the ``float8_e8m0fnu`` scale can
    be passed straight through without the AllGather direction's bitcast --
    no collective ever touches the scale here, it stays local to this rank.

    Unlike the AllGather direction, ReduceScatter sums every rank's partial
    GEMM contribution for a given output shard, so (unlike
    ``XPUMxFp8LinearKernel.apply_weights``'s bias-inside-the-GEMM
    convention) bias must *not* be passed into the GEMM here -- it would be
    added once per summed contribution; the GEMM always runs with
    ``bias=None`` and the real bias is added to the output once,
    afterwards.
    """

    def __init__(
        self,
        layer: RowParallelLinear,
        fp8_linear: XPUMxFp8LinearKernel,
    ) -> None:
        super().__init__(layer)
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)

    def is_supported(self) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "requires XPU"
        return True, None

    def apply(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[-1])
        out_dtype = x_2d.dtype
        x_q, a_scale = quant_mxfp8(x_2d)
        weight_t = self.layer.weight.t()
        weight_scale_t = self.layer.weight_scale.t()

        output_shape = [*x.shape[:-1], self.layer.weight.shape[0]]
        output = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
            x_q,
            weight_t,
            a_scale,
            weight_scale_t,
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
        if bias is not None:
            output = output + bias
        return output
