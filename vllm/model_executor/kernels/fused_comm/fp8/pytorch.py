# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed._symmetric_memory import (
    _pipelined_multi_all_gather_and_consume,
    enable_symm_mem_for_group,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.fused_comm.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
)
from vllm.model_executor.kernels.fused_comm.fp8.utils import (
    as_scaled_mm_scales,
    resolve_fp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    BlockWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import LinearBase, RowParallelLinear
from vllm.platforms import current_platform


class TorchAllGatherScaledMM(FusedAllGatherGemmKernel):
    """Fused AllGather+GEMM for a per-tensor/row-wise FP8 ``quant_method``
    (vLLM's native ``Fp8LinearMethod`` or a compressed-tensors FP8
    scheme)."""

    def __init__(self, layer: LinearBase) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert fp8_linear is not None
        self._fp8_kernel = fp8_linear
        # Only a static activation scale is identical on every rank, so only
        # then can the gathered shards share one tensor-wise scale.
        self._static_act_scale = fp8_linear.config.activation_quant_key.scale.static
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_scaled_mm

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if fp8_linear is None:
            return False, (
                "requires Fp8LinearMethod or a compressed-tensors FP8 scheme"
            )
        if not isinstance(
            fp8_linear,
            (
                PerTensorTorchFP8ScaledMMLinearKernel,
                RowWiseTorchFP8ScaledMMLinearKernel,
            ),
        ):
            return False, (
                "requires PerTensorTorchFP8ScaledMMLinearKernel or "
                "RowWiseTorchFP8ScaledMMLinearKernel"
            )
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
        As, Bs = as_scaled_mm_scales(A, B, As, Bs, tensorwise=self._static_act_scale)
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


class TorchScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """Fused GEMM+ReduceScatter counterpart of ``TorchAllGatherScaledMM``."""

    def __init__(self, layer: RowParallelLinear) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert fp8_linear is not None
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_scaled_mm

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if fp8_linear is None:
            return False, (
                "requires Fp8LinearMethod or a compressed-tensors FP8 scheme"
            )
        if not isinstance(
            fp8_linear,
            (
                PerTensorTorchFP8ScaledMMLinearKernel,
                RowWiseTorchFP8ScaledMMLinearKernel,
            ),
        ):
            return False, (
                "requires PerTensorTorchFP8ScaledMMLinearKernel or "
                "RowWiseTorchFP8ScaledMMLinearKernel"
            )
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


class TorchAllGatherBlockScaledMM(FusedAllGatherGemmKernel):
    """Fused AllGather+GEMM for a block-quantized FP8 ``quant_method``
    (vLLM's native ``Fp8LinearMethod`` or a compressed-tensors block FP8
    scheme)."""

    def __init__(self, layer: LinearBase) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert fp8_linear is not None
        self._fp8_kernel = fp8_linear
        self._group = get_tp_group()
        enable_symm_mem_for_group(self._group.device_group.group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if fp8_linear is None:
            return False, (
                "requires Fp8LinearMethod or a compressed-tensors FP8 scheme"
            )
        if not isinstance(fp8_linear, BlockWiseTorchFP8ScaledMMLinearKernel):
            return False, "requires BlockWiseTorchFP8ScaledMMLinearKernel"

        return True, None

    def target_gemm_method(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[-1])
        params = self._fp8_kernel._get_layer_params(layer)
        w = params.weight.t()
        w_scale = params.block_scale.t()
        x_q, a_scale = self._fp8_kernel.quant_fp8(
            x_2d, params.input_scale, params.input_scale_ub
        )
        out_dtype = self._fp8_kernel.config.out_dtype
        column_major_scales = self._fp8_kernel.quant_fp8.column_major_scales

        world_size = self._group.world_size
        output = x_q.new_empty(x_q.shape[0] * world_size, w.shape[1], dtype=out_dtype)
        output_shards = output.chunk(world_size)
        A = x_q.new_empty(x_q.shape[0] * world_size, x_q.shape[1])
        A_scale = a_scale.new_empty(a_scale.shape[0] * world_size, a_scale.shape[1])

        def shard_consumer(shards: list[torch.Tensor], rank: int) -> None:
            # Mirror BlockWiseTorchFP8ScaledMMLinearKernel.apply_block_scaled_mm:
            # cuBLASLt needs M % 4 == 0 and a column-major activation scale.
            a, a_s = shards
            out = output_shards[rank]
            M = a.shape[0]
            pad = (-M) % 4 if current_platform.is_cuda() else 0
            if pad:
                a_p = a.new_empty((M + pad, a.shape[1]))
                a_p[:M] = a
                a = a_p
                out = out.new_empty((M + pad, out.shape[1]))
            if column_major_scales:
                a_s_p = a_s.new_empty((a_s.shape[1], M + pad)).t()
                a_s_p[:M] = a_s
                a_s = a_s_p
            torch.ops.aten._scaled_mm.out(
                a,
                w,
                scale_a=a_s,
                scale_b=w_scale,
                out_dtype=out_dtype,
                out=out,
            )
            if pad:
                output_shards[rank].copy_(out[:M])

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


class TorchBlockScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """Fused GEMM+ReduceScatter counterpart of ``TorchAllGatherBlockScaledMM``."""

    # TODO: only validated on XPU. On CUDA this lacks the M % 4 padding of
    # BlockWiseTorchFP8ScaledMMLinearKernel, and chunking the column-major
    # activation scale per rank is unverified.

    def __init__(self, layer: RowParallelLinear) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert fp8_linear is not None
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if fp8_linear is None:
            return False, (
                "requires Fp8LinearMethod or a compressed-tensors FP8 scheme"
            )
        if not isinstance(fp8_linear, BlockWiseTorchFP8ScaledMMLinearKernel):
            return False, "requires BlockWiseTorchFP8ScaledMMLinearKernel"

        return True, None

    def target_gemm_method(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[-1])
        params = self._fp8_kernel._get_layer_params(layer)
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
        return output
