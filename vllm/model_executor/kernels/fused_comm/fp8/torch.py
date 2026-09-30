# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed._symmetric_memory import (
    _pipelined_multi_all_gather_and_consume,
    enable_symm_mem_for_group,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    BlockWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
    TorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.platforms import current_platform

from vllm.model_executor.kernels.fused_comm.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
)


def _shape_scales_for_torch_scaled_mm(
    fp8_kernel: TorchFP8ScaledMMLinearKernel,
    a_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape activation/weight scales into the layout ``aten::_scaled_mm``
    requires, mirroring the corresponding ``apply_scaled_mm`` of
    ``fp8_kernel`` (see ``scaled_mm/pytorch.py``). The fused symm_mem ops
    call ``aten::_scaled_mm`` internally and cannot run ``apply_scaled_mm``
    themselves, so this logic has to be duplicated here.
    """
    if isinstance(fp8_kernel, PerTensorTorchFP8ScaledMMLinearKernel):
        if a_scale.dim() == 0:
            a_scale = a_scale.view(1)
        if w_scale.dim() == 0:
            w_scale = w_scale.view(1)
        return a_scale, w_scale
    if isinstance(fp8_kernel, RowWiseTorchFP8ScaledMMLinearKernel):
        w_scale = w_scale.view(1, -1) if w_scale.dim() == 1 else w_scale.t()
        if a_scale.dim() == 1:
            a_scale = a_scale.view(-1, 1)
        return a_scale, w_scale
    raise TypeError(
        f"Unsupported TorchFP8ScaledMMLinearKernel subclass for fused "
        f"Comm+GEMM: {type(fp8_kernel).__name__}"
    )


def _make_unpadded_quant_fp8(fp8_kernel: TorchFP8ScaledMMLinearKernel) -> QuantFP8:
    """Build a ``QuantFP8`` matching ``fp8_kernel.quant_fp8``'s activation
    quant config, but with ``num_token_padding=None``.

    ``TorchFP8ScaledMMLinearKernel.get_output_padding`` pads the local
    (standalone, non-fused) ``torch._scaled_mm`` call's token dim up to a
    minimum of 17 rows for perf. The fused ``symm_mem`` ops instead gather
    (or reduce-scatter) each rank's quantized shard directly: padding rows
    added independently per rank would get woven into the result at each
    rank's shard boundary instead of trimmed from the end, corrupting the
    output. This path doesn't need the padding's perf benefit either, since
    the actual GEMM runs on the gathered (larger) tensor, not the small
    per-rank shard -- so it's simplest to just not pad here.
    """
    act_scale_descriptor = fp8_kernel.config.activation_quant_key.scale
    return QuantFP8(
        static=act_scale_descriptor.static,
        group_shape=act_scale_descriptor.group_shape,
        num_token_padding=None,
    )


class TorchSymmMemAllGatherScaledMM(FusedAllGatherGemmKernel):
    """FP8 fused AllGather+GEMM using torch symmetric memory, for the
    ``TorchFP8ScaledMMLinearKernel`` subclasses that feed real (not
    dequantized) scales straight into ``aten::_scaled_mm``:
    ``PerTensorTorchFP8ScaledMMLinearKernel`` and
    ``RowWiseTorchFP8ScaledMMLinearKernel`` (ROCm rowwise).

    ``ChannelWiseTorchFP8ScaledMMLinearKernel`` is excluded: it dequantizes
    the GEMM output with the real scales applied afterwards (an "unfused DQ"
    workaround for platforms without native rowwise ``_scaled_mm`` support),
    which the fused symm_mem op cannot replicate since it calls
    ``aten::_scaled_mm`` with the scales we pass it directly.
    """

    def __init__(
        self,
        layer: ColumnParallelLinear,
        fp8_linear: TorchFP8ScaledMMLinearKernel,
    ) -> None:
        super().__init__(layer)
        self._fp8_kernel = fp8_linear
        self._quant_fp8 = _make_unpadded_quant_fp8(fp8_linear)
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)

    def is_supported(self) -> tuple[bool, str | None]:
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "symmetric memory requires CUDA or ROCm/XPU"
        return True, None

    def apply(
        self,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x_shard.view(-1, x_shard.shape[-1])
        w, w_scale, x_scale, x_scale_ub = self._fp8_kernel._get_layer_params(self.layer)
        x_q, a_scale = self._quant_fp8(x_2d, x_scale, x_scale_ub)
        a_scale, w_scale = _shape_scales_for_torch_scaled_mm(
            self._fp8_kernel, a_scale, w_scale
        )
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


class TorchSymmMemScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """FP8 fused GEMM+ReduceScatter counterpart of
    ``TorchSymmMemAllGatherScaledMM``; see its docstring for the supported
    ``TorchFP8ScaledMMLinearKernel`` subclasses and their layout
    requirements.

    Unlike the AllGather direction (which concatenates distinct per-rank
    shards and so can add bias once per shard inside the GEMM), ReduceScatter
    *sums* every rank's partial GEMM contribution for a given output shard.
    Passing ``bias`` into the underlying per-rank ``aten::_scaled_mm`` calls
    would add it once per summed contribution (``world_size`` times total),
    so the GEMM always runs with ``bias=None`` and the real bias is added to
    the output once, afterwards.
    """

    def __init__(
        self,
        layer: RowParallelLinear,
        fp8_linear: TorchFP8ScaledMMLinearKernel,
    ) -> None:
        super().__init__(layer)
        self._fp8_kernel = fp8_linear
        self._quant_fp8 = _make_unpadded_quant_fp8(fp8_linear)
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)

    def is_supported(self) -> tuple[bool, str | None]:
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "symmetric memory requires CUDA or ROCm/XPU"
        return True, None

    def apply(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[-1])
        w, w_scale, x_scale, x_scale_ub = self._fp8_kernel._get_layer_params(self.layer)
        x_q, a_scale = self._quant_fp8(x_2d, x_scale, x_scale_ub)
        a_scale, w_scale = _shape_scales_for_torch_scaled_mm(
            self._fp8_kernel, a_scale, w_scale
        )
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


class TorchSymmMemAllGatherBlockScaledMM(FusedAllGatherGemmKernel):
    """FP8 block-scaled (DeepSeek-style 1x128/128x128) fused AllGather+GEMM
    using torch symmetric memory, for
    ``BlockWiseTorchFP8ScaledMMLinearKernel``.

    Unlike the per-tensor/row-wise AG kernels above, this cannot use the
    public ``fused_all_gather_scaled_matmul`` op: its scale-shape classifier
    (``_check_and_verify_fp8_all_gather_scale_mode``) only recognizes
    per-tensor (``numel() == 1``) and per-row (last dim ``== 1``) scales, and
    raises ``ValueError`` for any other shape -- including the block scale's
    ``[M, ceil(K/128)]`` layout (confirmed with a real 2-rank run on XPU
    whenever ``K > 128``, i.e. essentially always). So this instead calls
    ``_pipelined_multi_all_gather_and_consume`` directly (the same private
    primitive the public op is built on) to jointly gather ``(x_q, a_scale)``
    and run one local ``aten::_scaled_mm`` per peer shard as it arrives, with
    no scale-shape restriction.

    Known limitation: on CUDA, the standalone kernel additionally pads `M`
    to a multiple of 4 before a single local ``_scaled_mm`` call and slices
    it back off immediately after (a cuBLASLt requirement) -- entirely
    inside that one call. The fused op gathers+matmuls as one atomic step
    with no hook to pad only the local shard without corrupting the gather,
    so this relies on the *gathered* (full) token count already being a
    multiple of 4; if not, ``_scaled_mm`` raises explicitly rather than
    silently miscomputing.

    ``BlockWiseTorchFP8ScaledMMLinearKernel`` always applies bias *outside*
    the GEMM (see ``BlockScaledMMLinearKernel.apply_weights``), so this
    kernel matches that convention: the GEMM itself always runs with
    ``bias=None``, and the real bias is added to the output afterwards.
    """

    def __init__(
        self,
        layer: ColumnParallelLinear,
        fp8_linear: BlockWiseTorchFP8ScaledMMLinearKernel,
    ) -> None:
        super().__init__(layer)
        self._fp8_kernel = fp8_linear
        self._group = get_tp_group()
        enable_symm_mem_for_group(self._group.device_group.group_name)

    def is_supported(self) -> tuple[bool, str | None]:
        # Mirror the wrapped kernel's own support check (e.g. the SM90
        # requirement for DeepSeek-style block scaling on CUDA) so this
        # fused kernel never claims support in a case the non-fused kernel
        # itself wouldn't allow.
        supported, reason = type(self._fp8_kernel).is_supported()
        if not supported:
            return False, reason
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "symmetric memory requires CUDA or ROCm/XPU"
        return True, None

    def apply(
        self,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x_shard.view(-1, x_shard.shape[-1])
        params = self._fp8_kernel._get_layer_params(self.layer)
        w = params.weight.t()
        w_scale = params.block_scale.t()
        x_q, a_scale = self._fp8_kernel.quant_fp8(
            x_2d, params.input_scale, params.input_scale_ub
        )
        out_dtype = self._fp8_kernel.config.out_dtype

        world_size = self._group.world_size
        output = x_q.new_empty(x_q.shape[0] * world_size, w.shape[1], dtype=out_dtype)
        output_shards = output.chunk(world_size)
        A = x_q.new_empty(x_q.shape[0] * world_size, x_q.shape[1])
        A_scale = a_scale.new_empty(a_scale.shape[0] * world_size, a_scale.shape[1])

        def shard_consumer(shards: list[torch.Tensor], rank: int) -> None:
            torch.ops.aten._scaled_mm.out(
                shards[0],
                w,
                scale_a=shards[1],
                scale_b=w_scale,
                out_dtype=out_dtype,
                out=output_shards[rank],
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


class TorchSymmMemBlockScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """FP8 block-scaled fused GEMM+ReduceScatter counterpart of
    ``TorchSymmMemAllGatherBlockScaledMM``; see its docstring for the
    CUDA ``M % 4`` caveat.

    Unlike the AllGather direction, the public
    ``fused_scaled_matmul_reduce_scatter`` op's scale check is generic: it
    only requires ``A_scale.shape[:-1] == A.shape[:-1]`` (the scale's
    trailing dim is unconstrained), which the block scale's
    ``[M, ceil(K/128)]`` layout already satisfies. So this can safely keep
    using the public ``patched_fused_scaled_matmul_reduce_scatter`` wrapper
    directly, with no need for the private pipelined primitive the AllGather
    direction requires.

    As with ``TorchSymmMemScaledMMReduceScatter``, ReduceScatter sums every
    rank's partial GEMM contribution, so bias must *not* be passed into the
    GEMM (it would be added once per summed contribution); the GEMM runs
    with ``bias=None`` and the real bias is added to the output once,
    afterwards -- matching ``BlockScaledMMLinearKernel.apply_weights``'s
    bias-outside-the-GEMM convention anyway.
    """

    def __init__(
        self,
        layer: RowParallelLinear,
        fp8_linear: BlockWiseTorchFP8ScaledMMLinearKernel,
    ) -> None:
        super().__init__(layer)
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)

    def is_supported(self) -> tuple[bool, str | None]:
        # Mirror the wrapped kernel's own support check (e.g. the SM90
        # requirement for DeepSeek-style block scaling on CUDA) so this
        # fused kernel never claims support in a case the non-fused kernel
        # itself wouldn't allow.
        supported, reason = type(self._fp8_kernel).is_supported()
        if not supported:
            return False, reason
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "symmetric memory requires CUDA or ROCm/XPU"
        return True, None

    def apply(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[-1])
        params = self._fp8_kernel._get_layer_params(self.layer)
        w = params.weight.t()
        w_scale = params.block_scale.t()
        x_q, a_scale = self._fp8_kernel.quant_fp8(
            x_2d, params.input_scale, params.input_scale_ub
        )
        out_dtype = self._fp8_kernel.config.out_dtype

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
        if bias is not None:
            output = output + bias
        return output
