# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.kernels.linear import (  # noqa: E501
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    _SUPPORTED_ACT_QUANT_KEYS = {
        kFp8DynamicTensorSym,
        kFp8DynamicTokenSym,
        kFp8StaticTensorSym,
        kFp8StaticTokenSym,
    }
    _SUPPORTED_WEIGHT_QUANT_KEYS = {
        kFp8StaticChannelSym,
        kFp8StaticTensorSym,
    }

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in cls._SUPPORTED_WEIGHT_QUANT_KEYS:
            return (
                False,
                "XPUFP8ScaledMM only support per-channel and per-tensor quantization",
            )
        if c.activation_quant_key not in cls._SUPPORTED_ACT_QUANT_KEYS:
            return (
                False,
                "XPUFP8ScaledMM only support per-tensor and per-token activation "
                "quantization",
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        if c.activation_quant_key.dtype not in {
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        }:
            return False, "XPUFP8ScaledMM only support FP8 activation dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)

    def _ensure_kn_weight_layout(self, layer: torch.nn.Module) -> None:
        """Ensure weight is stored as C-contiguous [K, N] (KN layout).

        Checkpoints store weight as [N, K] (out_channels × in_channels).
        fp8_gemm requires [K, N] (in_channels × out_channels), C-contiguous.

        Fp8LinearMethod.process_weights_after_loading (fp8.py) calls
        weight.t() before calling this method, so we may receive:
          • [N, K] C-contiguous     ← direct from checkpoint (no prior .t())
          • [K, N] Fortran-order    ← after fp8.py's weight.t() (production)
          • [K, N] C-contiguous     ← already done (no-op)

        Square weights (K == N):
          shape is identical for [K,N] and [N,K].  We distinguish them by
          contiguity: Fortran-order means fp8.py already transposed → make
          contiguous.  C-contiguous means not yet transposed → transpose.
        """
        K = getattr(layer, "input_size_per_partition", self.config.weight_shape[1])
        N = getattr(layer, "output_size_per_partition", self.config.weight_shape[0])
        w = layer.weight

        if K != N:
            # Non-square: shape uniquely identifies the layout.
            if w.shape == (K, N):
                if not w.is_contiguous():
                    replace_parameter(layer, "weight", w.contiguous())
                return
            if w.shape == (N, K):
                replace_parameter(layer, "weight", w.t().contiguous())
                return
        else:
            # Square (K == N): use contiguity to distinguish.
            #   Fortran-order  → fp8.py already transposed [N,K]→[K,N]; just align.
            #   C-contiguous   → still in [N,K] checkpoint format; transpose.
            if w.shape == (K, N):
                if not w.is_contiguous():
                    # Production path: fp8.py did .t(), weight is [K,N] Fortran.
                    replace_parameter(layer, "weight", w.contiguous())
                else:
                    # Direct checkpoint path: weight is [N,K] C-contiguous.
                    replace_parameter(layer, "weight", w.t().contiguous())
                return

        raise ValueError(
            f"XPUFP8ScaledMM expects weight shape ({K},{N}) or ({N},{K}), "
            f"but got {tuple(w.shape)}"
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._ensure_kn_weight_layout(layer)
        # fp8_gemm routes on scale dtype (float32), not shape:
        #   per-tensor weight_scale [1]  → keep as [1]  (numel==1 branch)
        #   per-channel weight_scale [N] → keep as [N]  (per-channel branch)
        ws = layer.weight_scale
        if ws.numel() == 1:
            replace_parameter(layer, "weight_scale", ws.reshape(1))

    def apply_scaled_mm(
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
        # B is C-contiguous [K, N] from process_weights_after_loading.
        # fp8_gemm routes on scale dtype (float32) and numel:
        #   As [1]   → per-tensor  (numel==1 branch)
        #   As [M,1] → per-token   (group={1,K} branch, broadcast across K)
        #   Bs [1]   → per-tensor
        #   Bs [N]   → per-channel (mask=bit1 branch)
        # No shape manipulation needed here.
        output = torch.ops._xpu_C.fp8_gemm(A, B, out_dtype, As, Bs, bias)
        return output.view(*output_shape)


class XPUW8A16FP8LinearKernel(XPUFP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
            cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in {kFp8StaticChannelSym, kFp8StaticTensorSym}:
            return (
                False,
                "XPUFP8ScaledMM only support per-channel and per-tensor quantization",
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        return True, None

    def __init__(
            self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # fp8_gemm_w8a16 expects weight in [in, out] layout.
        # Transpose if weight is still in [out, in] layout.
        # For square matrices, use contiguity as tie-breaker:
        # checkpoint weights are contiguous, .t() views are not.
        weight = layer.weight
        out_features, in_features = self.config.weight_shape

        if weight.shape == (out_features, in_features) and (
            in_features != out_features or weight.is_contiguous()
        ):
            replace_parameter(layer, "weight", weight.data.t())
        # else: already in [in, out] layout — no-op

        weight_scale = layer.weight_scale.t().contiguous()
        replace_parameter(layer, "weight_scale", weight_scale.data)

    def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

    def apply_scaled_mm(
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
        pass
