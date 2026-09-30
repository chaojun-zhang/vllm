# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )


class FusedCommLinearKernel(ABC):
    """Common base for a fused Comm+GEMM kernel bound to one linear layer's
    boundary in eager sequence parallelism.

    Composed onto the layer the same way ``QuantizeMethodBase`` is (see
    ``layer.quant_method``): a delegate object selected once per layer and
    invoked on every call. The method does not own or modify weights -- it
    relies on the underlying quant method's
    ``process_weights_after_loading`` having already set the correct
    weight layout.
    """

    def __init__(self, layer: torch.nn.Module) -> None:
        self.layer = layer

    def is_supported(self) -> tuple[bool, str | None]:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the fused comm+GEMM op for this kernel's direction."""


class FusedAllGatherGemmKernel(FusedCommLinearKernel):
    """Fused AllGather + GEMM method for a ``ColumnParallelLinear``'s entry
    boundary. Peer to ``FusedGemmReduceScatterKernel`` -- a given backend
    may support one, both, or neither independently, since the two are
    never used by the same layer instance.

    Each instance is bound to a specific ``ColumnParallelLinear`` after
    weight loading.
    """

    def __init__(self, layer: "ColumnParallelLinear") -> None:
        super().__init__(layer)

    @abstractmethod
    def apply(
        self,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """AllGather + GEMM: [N/tp, H_in] -> [N, H_out_partition]"""


class FusedGemmReduceScatterKernel(FusedCommLinearKernel):
    """Fused GEMM + ReduceScatter method for a ``RowParallelLinear``'s exit
    boundary. Peer to ``FusedAllGatherGemmKernel`` -- a given backend may
    support one, both, or neither independently, since the two are never
    used by the same layer instance.

    Each instance is bound to a specific ``RowParallelLinear`` after
    weight loading.
    """

    def __init__(self, layer: "RowParallelLinear") -> None:
        super().__init__(layer)

    @abstractmethod
    def apply(
        self,
        x_shard: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """GEMM + ReduceScatter: [N, H_in_partition] -> [N/tp, H_out]"""
