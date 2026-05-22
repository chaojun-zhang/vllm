# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUW8A8FP8LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU FP8 GEMM is only available on XPU",
)


def _require_fp8_gemm() -> None:
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")
    if not hasattr(torch.ops._xpu_C, "fp8_gemm"):
        pytest.skip("_xpu_C.fp8_gemm is not available")


def _broadcast_scale(scale: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    if scale.numel() == 1:
        return scale
    if (
        scale.ndim == 2
        and scale.shape[0] in (1, tensor.shape[0])
        and scale.shape[1] in (1, tensor.shape[1])
    ):
        return scale
    if scale.numel() == tensor.shape[0]:
        return scale.reshape(-1, 1)
    if scale.numel() == tensor.shape[1]:
        return scale.reshape(1, -1)
    return scale


def _reference_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    output = (A.float() * _broadcast_scale(As, A)) @ (
        B.float() * _broadcast_scale(Bs, B)
    )
    if bias is not None:
        output = output + bias.float()
    return output.to(out_dtype)


def _scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    # torch._scaled_mm requires the second matrix to use col-major strides.
    B_col_major = B.t().contiguous().t()
    return torch._scaled_mm(
        A,
        B_col_major,
        _broadcast_scale(As, A).contiguous(),
        _broadcast_scale(Bs, B).contiguous(),
        bias,
        None,
        out_dtype,
        False,
    )


def _make_inputs(
    *,
    M: int,
    K: int,
    N: int,
    per_token_activation: bool,
    per_channel_weight: bool,
    with_bias: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    torch.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)
    device = torch.device("xpu")
    dtype = torch.bfloat16

    A = torch.randn(M, K, device=device, dtype=dtype) * 0.5
    B = torch.randn(K, N, device=device, dtype=dtype) * 0.5

    A_fp8, As = ops.scaled_fp8_quant(
        A, None, use_per_token_if_dynamic=per_token_activation
    )
    if per_channel_weight:
        B_t_fp8, Bs = ops.scaled_fp8_quant(
            B.t().contiguous(), None, use_per_token_if_dynamic=True
        )
        B_fp8 = B_t_fp8.t().contiguous()
        Bs = Bs.reshape(1, -1)
    else:
        B_fp8, Bs = ops.scaled_fp8_quant(B, None, use_per_token_if_dynamic=False)

    bias = torch.randn(N, device=device, dtype=dtype) * 0.1 if with_bias else None
    return A_fp8, B_fp8, As, Bs, bias


def _quantize_activation_for_key(
    x: torch.Tensor,
    activation_quant_key: QuantKey,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if activation_quant_key == kFp8DynamicTensorSym:
        x_fp8, x_scale = ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=False)
        return x_fp8, x_scale, None
    if activation_quant_key == kFp8DynamicTokenSym:
        x_fp8, x_scale = ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
        return x_fp8, x_scale, None
    if activation_quant_key == kFp8StaticTensorSym:
        input_scale = torch.tensor([0.0045], device=x.device, dtype=torch.float32)
        x_fp8, x_scale = ops.scaled_fp8_quant(x, input_scale)
        return x_fp8, x_scale, input_scale
    if activation_quant_key == kFp8StaticTokenSym:
        _, input_scale = ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
        x_fp8, x_scale = ops.scaled_fp8_quant(
            x,
            input_scale,
            group_shape=(GroupShape.PER_TOKEN.row, GroupShape.PER_TOKEN.col),
        )
        return x_fp8, x_scale, input_scale
    raise AssertionError(f"unexpected activation quant key: {activation_quant_key}")


def _quantize_weight_for_key(
    weight: torch.Tensor,
    weight_quant_key: QuantKey,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight_quant_key == kFp8StaticTensorSym:
        weight_scale = torch.tensor([0.0035], device=weight.device, dtype=torch.float32)
        return ops.scaled_fp8_quant(weight, weight_scale)
    if weight_quant_key == kFp8StaticChannelSym:
        weight_t_fp8, weight_scale = ops.scaled_fp8_quant(
            weight.t().contiguous(), None, use_per_token_if_dynamic=True
        )
        return weight_t_fp8.t().contiguous(), weight_scale.reshape(1, -1)
    raise AssertionError(f"unexpected weight quant key: {weight_quant_key}")


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize(
    ("per_token_activation", "per_channel_weight"),
    [
        pytest.param(False, False, id="per-tensor-act-per-tensor-weight"),
        pytest.param(True, True, id="per-token-act-per-channel-weight"),
    ],
)
def test_xpu_fp8_gemm_matches_torch_scaled_mm_and_reference(
    per_token_activation: bool,
    per_channel_weight: bool,
    with_bias: bool,
) -> None:
    _require_fp8_gemm()
    A, B, As, Bs, bias = _make_inputs(
        M=16,
        K=32,
        N=32,
        per_token_activation=per_token_activation,
        per_channel_weight=per_channel_weight,
        with_bias=with_bias,
        seed=0,
    )

    actual = torch.ops._xpu_C.fp8_gemm(A, B, torch.bfloat16, As, Bs, bias)
    expected_scaled_mm = _scaled_mm(A, B, As, Bs, bias, torch.bfloat16)
    expected_reference = _reference_scaled_mm(A, B, As, Bs, bias, torch.bfloat16)

    torch.testing.assert_close(actual, expected_scaled_mm, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual, expected_reference, atol=2e-2, rtol=2e-2)


def test_xpu_fp8_gemm_supports_non_tile_shapes_against_reference() -> None:
    _require_fp8_gemm()
    A, B, As, Bs, bias = _make_inputs(
        M=7,
        K=19,
        N=13,
        per_token_activation=True,
        per_channel_weight=True,
        with_bias=True,
        seed=1,
    )

    actual = torch.ops._xpu_C.fp8_gemm(A, B, torch.bfloat16, As, Bs, bias)
    expected = _reference_scaled_mm(A, B, As, Bs, bias, torch.bfloat16)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "activation_quant_key",
    [
        pytest.param(kFp8DynamicTensorSym, id="dynamic-tensor-act"),
        pytest.param(kFp8DynamicTokenSym, id="dynamic-token-act"),
        pytest.param(kFp8StaticTensorSym, id="static-tensor-act"),
        pytest.param(kFp8StaticTokenSym, id="static-token-act"),
    ],
)
@pytest.mark.parametrize(
    "weight_quant_key",
    [
        pytest.param(kFp8StaticTensorSym, id="static-tensor-weight"),
        pytest.param(kFp8StaticChannelSym, id="static-channel-weight"),
    ],
)
def test_xpu_scaled_mm_kernel_apply_weights_matches_reference_for_supported_keys(
    default_vllm_config: object,
    activation_quant_key: QuantKey,
    weight_quant_key: QuantKey,
) -> None:
    assert default_vllm_config is not None
    _require_fp8_gemm()
    torch.manual_seed(2)
    torch.xpu.manual_seed_all(2)
    M, K, N = 16, 32, 48
    x = torch.randn(M, K, device="xpu", dtype=torch.bfloat16) * 0.5
    weight = torch.randn(K, N, device="xpu", dtype=torch.bfloat16) * 0.5
    bias = torch.randn(N, device="xpu", dtype=torch.bfloat16) * 0.1
    x_fp8, x_scale, input_scale = _quantize_activation_for_key(x, activation_quant_key)
    weight_fp8, weight_scale = _quantize_weight_for_key(weight, weight_quant_key)
    kernel = _make_xpu_fp8_kernel(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        weight_shape=(K, N),
    )

    layer = torch.nn.Module()
    layer.input_size_per_partition = K
    layer.output_size_per_partition = N
    layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
    layer.weight_scale = weight_scale
    layer.input_scale = input_scale
    layer.input_scale_ub = None

    actual = kernel.apply_weights(layer, x.view(2, M // 2, K), bias)
    expected = _reference_scaled_mm(
        x_fp8, weight_fp8, x_scale, weight_scale, bias, torch.bfloat16
    ).view(2, M // 2, N)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def _make_xpu_fp8_kernel(
    weight_quant_key: QuantKey,
    activation_quant_key: QuantKey,
    weight_shape: tuple[int, int],
) -> XPUW8A8FP8LinearKernel:
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        weight_shape=weight_shape,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    return XPUW8A8FP8LinearKernel(
        config,
        layer_param_names=["weight", "weight_scale", "input_scale", "input_scale_ub"],
    )


def test_xpu_scaled_mm_kernel_layout_and_apply_weights(
    default_vllm_config: object,
) -> None:
    assert default_vllm_config is not None
    _require_fp8_gemm()
    M, K, N = 16, 32, 48
    A, B, As, Bs, bias = _make_inputs(
        M=M,
        K=K,
        N=N,
        per_token_activation=True,
        per_channel_weight=True,
        with_bias=True,
        seed=2,
    )
    kernel = _make_xpu_fp8_kernel(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=(K, N),
    )

    layer = torch.nn.Module()
    layer.input_size_per_partition = K
    layer.output_size_per_partition = N
    layer.weight = torch.nn.Parameter(B.t().clone(), requires_grad=False)
    layer.weight_scale = Bs
    layer.input_scale = As
    layer.input_scale_ub = None

    kernel.process_weights_after_loading(layer)

    assert layer.weight.shape == (K, N)
    torch.testing.assert_close(layer.weight.float(), B.float(), atol=0, rtol=0)

    actual = kernel.apply_weights(layer, A.view(2, M // 2, K), bias)
    expected = _reference_scaled_mm(A, B, As, Bs, bias, torch.bfloat16).view(
        2, M // 2, N
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_xpu_scaled_mm_kernel_layout_falls_back_to_config_shape(
    default_vllm_config: object,
) -> None:
    assert default_vllm_config is not None
    _require_fp8_gemm()
    K, N = 32, 48
    _, weight, _, weight_scale, _ = _make_inputs(
        M=16,
        K=K,
        N=N,
        per_token_activation=True,
        per_channel_weight=True,
        with_bias=False,
        seed=3,
    )
    kernel = _make_xpu_fp8_kernel(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=(N, K),
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.t().clone(), requires_grad=False)
    layer.weight_scale = weight_scale
    layer.input_scale = None
    layer.input_scale_ub = None

    kernel.process_weights_after_loading(layer)

    assert layer.weight.shape == (K, N)
    torch.testing.assert_close(layer.weight.float(), weight.float(), atol=0, rtol=0)


def test_xpu_scaled_mm_kernel_can_implement_expected_fp8_quant_keys() -> None:
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticTensorSym,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=(32, 32),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )

    supported, reason = XPUW8A8FP8LinearKernel.can_implement(config)

    assert supported, reason
