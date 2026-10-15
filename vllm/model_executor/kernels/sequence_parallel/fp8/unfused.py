# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.sequence_parallel.base import FusedAllGatherGemmKernel
from vllm.model_executor.kernels.sequence_parallel.fp8.utils import (
    resolve_fp8_linear_kernel,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.models.common.ops.sequence_parallel import sp_all_gather


class UnfusedBlockScaledMMAllGather(FusedAllGatherGemmKernel):
    """All-gathers the quantized activation (not the raw input) via plain
    collectives, then a plain ``apply_block_scaled_mm``.

    ``sp_all_gather`` concatenates each rank's shard into a fresh
    contiguous tensor, so a column-major activation scale (as produced by
    e.g. the CUTLASS and DeepGEMM block kernels' ``QuantFP8``) loses its
    layout in the process; it is rebuilt below when required.
    """

    def __init__(self, layer: LinearBase) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert isinstance(fp8_linear, Fp8BlockScaledMMLinearKernel)
        self._fp8_kernel = fp8_linear
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if not isinstance(fp8_linear, Fp8BlockScaledMMLinearKernel):
            return False, "requires Fp8BlockScaledMMLinearKernel"
        if not fp8_linear.apply_input_quant:
            return False, "kernel takes unquantized input"
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
        x_q, x_scale = fp8_kernel.quant_fp8(
            x_2d,
            params.input_scale,
            params.input_scale_ub,
            use_triton=fp8_kernel.use_triton,
        )
        x_q_full = sp_all_gather(x_q)
        x_scale_full = sp_all_gather(x_scale)
        if fp8_kernel.quant_fp8.column_major_scales:
            x_scale_full = x_scale_full.t().contiguous().t()

        output = fp8_kernel.apply_block_scaled_mm(
            A=x_q_full, B=params.weight, As=x_scale_full, Bs=params.block_scale
        )
        if bias is not None:
            output = output + bias
        return output.to(dtype=fp8_kernel.config.out_dtype).view(
            x_q_full.shape[0], params.weight.shape[0]
        )
