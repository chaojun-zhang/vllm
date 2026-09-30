# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase


class AllGatherGemmKernel(ABC):
    """Fuses AllGather + GEMM for a ``ColumnParallelLinear``'s entry boundary.

    At construction, replaces ``hook`` -- the quant method's per-layer GEMM
    call (e.g. ``apply_scaled_mm``/``apply_weights``, or plain
    ``LinearMethodBase.apply``) -- with a dispatcher. ``set_fuse_gemm_comms``
    toggles that dispatcher: on, it runs ``apply_fused`` (the collective
    fused into the GEMM); off, it calls the original hook unchanged.
    ``mark_sp_region`` flips this switch once per forward, right before
    ``linear.forward`` runs -- which itself is never touched.

    Peer to ``GemmReduceScatterKernel`` -- a backend may support one, both,
    or neither, since the two never run on the same layer.
    """

    def __init__(self, quant_method: QuantizeMethodBase, hook: Callable) -> None:
        self.quant_method = quant_method
        self._fuse_gemm_comms = False
        self._unfused_fn = hook
        # Replace hook.__self__'s method so every caller (incl.
        # linear.forward) transparently goes through our dispatcher.
        setattr(hook.__self__, hook.__name__, self._dispatch)

    @classmethod
    @abstractmethod
    def is_supported(cls, quant_method: QuantizeMethodBase) -> tuple[bool, str | None]:
        """Whether this kernel can handle `quant_method` (and the current
        platform). Called on the class, before construction, so a backend
        can be probed without any side effects."""
        raise NotImplementedError

    def set_fuse_gemm_comms(self, enabled: bool) -> None:
        """Switch this call on/off the fused path. Called once per forward
        by `mark_sp_region`, ahead of invoking `linear.forward`."""
        self._fuse_gemm_comms = enabled

    def _dispatch(self, *args, **kwargs):
        if self._fuse_gemm_comms:
            return self.apply_fused(*args, **kwargs)
        return self._unfused_fn(*args, **kwargs)

    @abstractmethod
    def apply_fused(self, *args, **kwargs):
        """Run the fused comm+GEMM op in place of the patched hook."""


class GemmReduceScatterKernel(ABC):
    """Fuses GEMM + ReduceScatter for a ``RowParallelLinear``'s exit boundary.

    Same install/dispatch mechanism as ``AllGatherGemmKernel`` (see there).

    Also stores ``layer``: kernels hooking ``apply_scaled_mm`` (which
    doesn't get ``layer`` as an argument) need it to recompute the real
    bias. Kernels hooking ``apply_weights``/``apply`` get ``layer``
    directly as a call argument and ignore this stored copy.

    Peer to ``AllGatherGemmKernel`` -- a backend may support one, both,
    or neither, since the two never run on the same layer.
    """

    def __init__(
        self, quant_method: QuantizeMethodBase, layer: object, hook: Callable
    ) -> None:
        self.quant_method = quant_method
        self.layer = layer
        self._fuse_gemm_comms = False
        self._unfused_fn = hook
        # Replace hook.__self__'s method so every caller (incl.
        # linear.forward) transparently goes through our dispatcher.
        setattr(hook.__self__, hook.__name__, self._dispatch)

    @classmethod
    @abstractmethod
    def is_supported(cls, quant_method: QuantizeMethodBase) -> tuple[bool, str | None]:
        """Whether this kernel can handle `quant_method` (and the current
        platform). Called on the class, before construction, so a backend
        can be probed without any side effects."""
        raise NotImplementedError

    def set_fuse_gemm_comms(self, enabled: bool) -> None:
        """Switch this call on/off the fused path. Called once per forward
        by `mark_sp_region`, ahead of invoking `linear.forward`."""
        self._fuse_gemm_comms = enabled

    def _dispatch(self, *args, **kwargs):
        if self._fuse_gemm_comms:
            return self.apply_fused(*args, **kwargs)
        return self._unfused_fn(*args, **kwargs)

    @abstractmethod
    def apply_fused(self, *args, **kwargs):
        """Run the fused comm+GEMM op in place of the patched hook."""
