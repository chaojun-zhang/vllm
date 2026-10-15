# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.distributed._symmetric_memory import (
    _pipelined_multi_all_gather_and_consume,
    enable_symm_mem_for_group,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.sequence_parallel.base import (
    FusedAllGatherGemmKernel,
)
from vllm.model_executor.kernels.sequence_parallel.fp8.utils import (
    resolve_fp8_linear_kernel,
)
from vllm.model_executor.layers.linear import LinearBase


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
