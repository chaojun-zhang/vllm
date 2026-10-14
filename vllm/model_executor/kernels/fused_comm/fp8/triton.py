# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Comm+GEMM kernels for ``TritonFp8BlockScaledMMKernel`` (the
block-scaled FP8 fallback used wherever CUTLASS/DeepGEMM/FlashInfer's
Hopper+-only kernels aren't available, e.g. Ada Lovelace GPUs like L20/L40
on ``--linear-backend triton`` or by default kernel-priority fallback)."""

import torch
from torch.distributed._symmetric_memory import (
    _pipelined_multi_all_gather_and_consume,
    _pipelined_produce_and_all2all,
    enable_symm_mem_for_group,
    reduce_partials,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.fused_comm.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
)
from vllm.model_executor.kernels.fused_comm.fp8.utils import resolve_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.linear import LinearBase, RowParallelLinear


class TritonAllGatherBlockScaledMM(FusedAllGatherGemmKernel):
    """Fused AllGather+GEMM for ``TritonFp8BlockScaledMMKernel``.

    Pipelines the AllGather with the Triton block-scaled GEMM per shard,
    the same way ``CutlassAllGatherBlockScaledMM``/
    ``TorchAllGatherBlockScaledMM`` pipeline it with their own GEMM.
    Triton's kernel reads the activation scale via explicit strides, so
    (unlike the CUTLASS/DeepGEMM kernels) no column-major layout needs to
    be rebuilt after the gather.
    """

    def __init__(self, layer: LinearBase) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert isinstance(fp8_linear, TritonFp8BlockScaledMMKernel)
        self._fp8_kernel = fp8_linear
        self._group = get_tp_group()
        enable_symm_mem_for_group(self._group.device_group.group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if not isinstance(fp8_linear, TritonFp8BlockScaledMMKernel):
            return False, "requires TritonFp8BlockScaledMMKernel"
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
            x_2d,
            params.input_scale,
            params.input_scale_ub,
            use_triton=fp8_kernel.use_triton,
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


class TritonBlockScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """Fused GEMM+ReduceScatter counterpart of
    ``TritonAllGatherBlockScaledMM``.

    Unlike ``TorchBlockScaledMMReduceScatter``/``CutlassBlockScaledMMReduceScatter``
    (which go through the public ``fused_scaled_matmul_reduce_scatter`` symm_mem
    op, itself hardcoded to ``aten::_scaled_mm`` and its 128x128-only/column-major
    block-scale requirements), this pipelines the same lower-level primitives
    that op is built from -- ``_pipelined_produce_and_all2all`` +
    ``reduce_partials`` -- with Triton's own ``apply_block_scaled_mm`` as the
    per-shard GEMM. This supports any ``weight_group_shape`` (not just
    128x128) and needs no column-major scale layout, since Triton's GEMM
    reads scales via explicit strides.
    """

    def __init__(self, layer: RowParallelLinear) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert isinstance(fp8_linear, TritonFp8BlockScaledMMKernel)
        self._fp8_kernel = fp8_linear
        self._group = get_tp_group()
        enable_symm_mem_for_group(self._group.device_group.group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if not isinstance(fp8_linear, TritonFp8BlockScaledMMKernel):
            return False, "requires TritonFp8BlockScaledMMKernel"
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
            x_2d,
            params.input_scale,
            params.input_scale_ub,
            use_triton=fp8_kernel.use_triton,
        )
        out_dtype = fp8_kernel.config.out_dtype
        group_name = self._group.device_group.group_name
        world_size = self._group.world_size
        n_out = params.weight.shape[0]

        # Each rank's full (unsharded) M rows, split into the world_size
        # destination chunks produce_and_all2all will ship to each peer.
        A_shards = x_q.chunk(world_size)
        A_scale_shards = a_scale.chunk(world_size)

        def chunk_producer(rank: int, out: torch.Tensor) -> None:
            out.copy_(
                fp8_kernel.apply_block_scaled_mm(
                    A=A_shards[rank],
                    B=params.weight,
                    As=A_scale_shards[rank],
                    Bs=params.block_scale,
                )
            )

        stacked_partials = x_q.new_empty(x_q.shape[0], n_out, dtype=out_dtype)
        _pipelined_produce_and_all2all(chunk_producer, stacked_partials, group_name)

        # Each rank now holds `world_size` unreduced partial contributions
        # (one per peer's local K-shard) for its own M-shard; sum them.
        output = reduce_partials(
            stacked_partials.view(world_size, -1, n_out),
            dim=0,
            reduce_op="sum",
            output_dtype=out_dtype,
            group_size=world_size,
        )
        return output
