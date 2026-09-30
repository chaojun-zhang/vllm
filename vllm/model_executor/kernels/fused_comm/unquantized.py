# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed._symmetric_memory import (
    enable_symm_mem_for_group,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.fused_comm.base import (
    AllGatherGemmKernel,
    GemmReduceScatterKernel,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.platforms import current_platform


class AllGatherGemm(AllGatherGemmKernel):
    @classmethod
    def is_supported(
        cls, quant_method: UnquantizedLinearMethod
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "symmetric memory requires XPU"
        return True, None

    def __init__(self, quant_method: UnquantizedLinearMethod) -> None:
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(quant_method, quant_method.apply)

    def apply_fused(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        W = layer.weight.t()
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x, [W], gather_dim=0, group_name=self._group_name
        )
        output = mm_outputs[0]
        if bias is not None:
            output = output + bias
        return output


class GemmReduceScatter(GemmReduceScatterKernel):
    @classmethod
    def is_supported(
        cls, quant_method: UnquantizedLinearMethod
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "symmetric memory requires XPU"
        return True, None

    def __init__(
        self, quant_method: UnquantizedLinearMethod, layer: torch.nn.Module
    ) -> None:
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(quant_method, layer, quant_method.apply)

    def apply_fused(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        W = layer.weight.t()
        output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x, W, "sum", scatter_dim=0, group_name=self._group_name
        )
        real_bias = layer.bias if not layer.skip_bias_add else None
        if real_bias is not None:
            output = output + real_bias
        return output
