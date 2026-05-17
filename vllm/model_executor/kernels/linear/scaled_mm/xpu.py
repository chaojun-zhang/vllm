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
        expected_shape = (
            getattr(layer, "input_size_per_partition", self.config.weight_shape[1]),
            getattr(layer, "output_size_per_partition", self.config.weight_shape[0]),
        )
        if layer.weight.shape == expected_shape:
            return
        if layer.weight.shape == expected_shape[::-1]:
            replace_parameter(layer, "weight", layer.weight.data.t())
            return
        raise ValueError(
            "XPUFP8ScaledMM expects weight shape "
            f"{expected_shape} or {expected_shape[::-1]}, "
            f"but got {tuple(layer.weight.shape)}"
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._ensure_kn_weight_layout(layer)

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
        mat2 = B.t().contiguous().t()
        scale_a = As.reshape(-1, 1).expand(A.shape[0], 1).contiguous()
        scale_b = Bs.reshape(1, -1).expand(1, mat2.shape[1]).contiguous()
        output = torch._scaled_mm(
            A,
            mat2,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )
        if type(output) is tuple and len(output) == 2:
            output = output[0]
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
        w, w_s, x_s, _ = self._get_layer_params(layer)
        out_dtype = x.dtype if self.config.out_dtype is None else self.config.out_dtype
        output_shape = [*x.shape[:-1], w.shape[1]]
        return self.apply_scaled_mm(
            A=x.view(-1, x.shape[-1]),
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )
        replace_parameter(layer, "weight", layer.weight.data.t())

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
