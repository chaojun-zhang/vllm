# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed._symmetric_memory import (
    _pipelined_multi_all_gather_and_consume,
    enable_symm_mem_for_group,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.fused_comm.base import (
    AllGatherGemmKernel,
    GemmReduceScatterKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    BlockWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
    TorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.platforms import current_platform


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


class TorchAllGatherScaledMM(AllGatherGemmKernel):
    def __init__(self, quant_method: Fp8LinearMethod) -> None:
        self._fp8_kernel = quant_method.fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(quant_method, quant_method.fp8_linear.apply_scaled_mm)

    @classmethod
    def is_supported(cls, quant_method: Fp8LinearMethod) -> tuple[bool, str | None]:
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "symmetric memory requires CUDA or ROCm/XPU"
        if not isinstance(
            quant_method.fp8_linear,
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

    def apply_fused(
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
        As, Bs = _shape_scales_for_torch_scaled_mm(self._fp8_kernel, As, Bs)

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


class TorchScaledMMReduceScatter(GemmReduceScatterKernel):
    def __init__(
        self,
        quant_method: Fp8LinearMethod,
        layer: RowParallelLinear,
    ) -> None:
        self._fp8_kernel = quant_method.fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(quant_method, layer, quant_method.fp8_linear.apply_scaled_mm)

    @classmethod
    def is_supported(cls, quant_method: Fp8LinearMethod) -> tuple[bool, str | None]:
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "symmetric memory requires CUDA or ROCm/XPU"
        if not isinstance(
            quant_method.fp8_linear,
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

    def apply_fused(
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
        As, Bs = _shape_scales_for_torch_scaled_mm(self._fp8_kernel, As, Bs)

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
        real_bias = self.layer.bias if not self.layer.skip_bias_add else None
        if real_bias is not None:
            output = output + real_bias
        return output


class TorchAllGatherBlockScaledMM(AllGatherGemmKernel):
    def __init__(self, quant_method: Fp8LinearMethod) -> None:
        self._fp8_kernel = quant_method.fp8_linear
        self._group = get_tp_group()
        enable_symm_mem_for_group(self._group.device_group.group_name)
        super().__init__(quant_method, quant_method.fp8_linear.apply_weights)

    @classmethod
    def is_supported(cls, quant_method: Fp8LinearMethod) -> tuple[bool, str | None]:
        if not isinstance(
            quant_method.fp8_linear, BlockWiseTorchFP8ScaledMMLinearKernel
        ):
            return False, "requires BlockWiseTorchFP8ScaledMMLinearKernel"

        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "symmetric memory requires CUDA or ROCm/XPU"
        return True, None

    def apply_fused(
        self,
        layer: torch.nn.Module,
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


class TorchBlockScaledMMReduceScatter(GemmReduceScatterKernel):
    def __init__(
        self,
        quant_method: Fp8LinearMethod,
        layer: RowParallelLinear,
    ) -> None:
        self._fp8_kernel = quant_method.fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(quant_method, layer, quant_method.fp8_linear.apply_weights)

    @classmethod
    def is_supported(cls, quant_method: Fp8LinearMethod) -> tuple[bool, str | None]:
        if not isinstance(
            quant_method.fp8_linear, BlockWiseTorchFP8ScaledMMLinearKernel
        ):
            return False, "requires BlockWiseTorchFP8ScaledMMLinearKernel"

        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "symmetric memory requires CUDA or ROCm/XPU"
        return True, None

    def apply_fused(
        self,
        layer: torch.nn.Module,
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
        real_bias = self.layer.bias if not self.layer.skip_bias_add else None
        if real_bias is not None:
            output = output + real_bias
        return output
