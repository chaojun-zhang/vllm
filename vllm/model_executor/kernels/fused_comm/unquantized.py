# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed._symmetric_memory import (
    enable_symm_mem_for_group,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform

from vllm.model_executor.kernels.fused_comm.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
)


class SymmMemAllGatherGemm(FusedAllGatherGemmKernel):
    """Unquantized fused AllGather+GEMM using torch symmetric memory."""

    def __init__(self, layer: ColumnParallelLinear) -> None:
        super().__init__(layer)
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
        W = self.layer.weight.t()
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard, [W], gather_dim=0, group_name=self._group_name
        )
        output = mm_outputs[0]
        if bias is not None:
            output = output + bias
        return output


class SymmMemGemmReduceScatter(FusedGemmReduceScatterKernel):
    """Unquantized fused GEMM+ReduceScatter using torch symmetric memory."""

    def __init__(self, layer: RowParallelLinear) -> None:
        super().__init__(layer)
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
        W = self.layer.weight.t()
        output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x, W, "sum", scatter_dim=0, group_name=self._group_name
        )
        if bias is not None:
            output = output + bias
        return output
