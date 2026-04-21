# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

# This registers op implementations
import vllm.kernels  # noqa: F401
from tests.kernels.allclose_default import get_default_rtol
from vllm import ir
from vllm.platforms import current_platform


def rms_norm_inputs(n_tokens: int, hidden_size: int, dtype: torch.dtype):
    x = torch.randn(n_tokens, hidden_size, dtype=dtype)
    weight = torch.rand(hidden_size, dtype=dtype)
    return x, weight


def mixer2_rms_norm_gated_inputs(n_tokens: int, hidden_size: int, dtype: torch.dtype):
    x = torch.randn(n_tokens, hidden_size, dtype=dtype)
    gate = torch.randn(n_tokens, hidden_size, dtype=dtype)
    weight = torch.rand(hidden_size, dtype=dtype)
    return x, gate, weight


rms_norm_native = ir.ops.rms_norm.impls["native"].impl_fn
mixer2_rms_norm_gated_native = ir.ops.mixer2_rms_norm_gated.impls["native"].impl_fn


def mixer2_rms_norm_gated_reference(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    group_size: int | None = None,
) -> torch.Tensor:
    input_dtype = x.dtype
    x = x * F.silu(gate.to(torch.float32))
    if group_size is None:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + epsilon)
    else:
        *prefix_dims, hidden_size = x.shape
        x_grouped = x.reshape(*prefix_dims, hidden_size // group_size, group_size)
        variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
        x = (x_grouped * torch.rsqrt(variance + epsilon)).reshape(
            *prefix_dims, hidden_size
        )

    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(input_dtype)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_rms_norm_registration():
    expected = {
        "native": True,
        "vllm_c": current_platform.is_cuda_alike(),
        "aiter": current_platform.is_rocm(),
        "oink": False,
        "xpu_kernels": current_platform.is_xpu(),
    }

    actual = {
        provider: impl.supported for provider, impl in ir.ops.rms_norm.impls.items()
    }

    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 17])
@pytest.mark.parametrize("hidden_size", [16, 4096, 8192])
@pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestRMSNorm:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, hidden_size, epsilon):
        x, weight = rms_norm_inputs(4, 8, dtype)
        out = rms_norm_native(x, weight, epsilon=epsilon)

        # Check shape, dtype, device
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # Check the scaling property of rms norm
        out2 = rms_norm_native(x * 2.0, weight, epsilon=epsilon)
        torch.testing.assert_close(out2, out, rtol=get_default_rtol(out), atol=1e-3)

        # Check behavior with and without weight
        weight1 = torch.ones_like(weight)
        out3 = rms_norm_native(x, weight1, epsilon=epsilon)
        out4 = rms_norm_native(x, None, epsilon=epsilon)
        torch.testing.assert_close(out3, out4)

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter", "xpu_kernels"])
    def test_impls(self, dtype, n_tokens, hidden_size, epsilon, provider):
        impl = ir.ops.rms_norm.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x, weight = rms_norm_inputs(n_tokens, hidden_size, dtype)
        args = (x, weight, epsilon, None)

        assert impl.supported

        if provider == "aiter" and dtype not in [torch.float16, torch.bfloat16]:
            assert not impl.supports_args(*args)
            return

        assert impl.supports_args(*args)

        out_impl = impl.impl_fn(*args)
        out_native = rms_norm_native(*args)

        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )

        # check that dispatched call matches direct call
        with ir.ops.rms_norm.set_priority([provider, "native"]):
            out_impl2 = ir.ops.rms_norm(*args)

        # exact match
        torch.testing.assert_close(out_impl2, out_impl, rtol=0.0, atol=0.0)

        # none of these support variance_size override
        assert not impl.supports_args(x, weight, epsilon, 4)
        assert not impl.supports_args(x, weight, epsilon, variance_size=4)

        # test weight=None behavior
        out_impl_no_weight = impl.impl_fn(x, None, epsilon)
        out_impl_unit_weight = impl.impl_fn(x, torch.ones_like(weight), epsilon)
        torch.testing.assert_close(
            out_impl_no_weight,
            out_impl_unit_weight,
            rtol=get_default_rtol(out_impl_no_weight),
            atol=2e-4,
        )

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter", "xpu_kernels", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, epsilon, provider):
        if not ir.ops.rms_norm.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x, weight = rms_norm_inputs(n_tokens, hidden_size, dtype)
        args = (x, weight, epsilon, None)

        # When checking the torch op, we have to set priority and use dispatch
        with ir.ops.rms_norm.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.rms_norm, args)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_mixer2_rms_norm_gated_registration():
    expected = {
        "native": True,
        "triton": current_platform.is_cuda_alike() or current_platform.is_xpu(),
    }

    actual = {
        provider: impl.supported
        for provider, impl in ir.ops.mixer2_rms_norm_gated.impls.items()
    }

    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 17])
@pytest.mark.parametrize("hidden_size", [16, 4096])
@pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
@pytest.mark.parametrize("group_size", [None, 4])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestMixer2RMSNormGated:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, hidden_size, epsilon, group_size):
        x, gate, weight = mixer2_rms_norm_gated_inputs(n_tokens, hidden_size, dtype)
        out = mixer2_rms_norm_gated_native(
            x, gate, weight, epsilon=epsilon, group_size=group_size
        )
        ref = mixer2_rms_norm_gated_reference(
            x, gate, weight, epsilon=epsilon, group_size=group_size
        )

        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        torch.testing.assert_close(out, ref, rtol=get_default_rtol(out), atol=1e-3)

        weight1 = torch.ones_like(weight)
        out3 = mixer2_rms_norm_gated_native(
            x, gate, weight1, epsilon=epsilon, group_size=group_size
        )
        out4 = mixer2_rms_norm_gated_native(
            x, gate, None, epsilon=epsilon, group_size=group_size
        )
        torch.testing.assert_close(out3, out4)

        out_full = mixer2_rms_norm_gated_native(
            x, gate, weight, epsilon=epsilon, group_size=None
        )
        out_all_grouped = mixer2_rms_norm_gated_native(
            x, gate, weight, epsilon=epsilon, group_size=hidden_size
        )
        torch.testing.assert_close(
            out_all_grouped, out_full, rtol=get_default_rtol(out_full), atol=1e-3
        )

    @pytest.mark.parametrize("provider", ["triton"])
    def test_impls(self, dtype, n_tokens, hidden_size, epsilon, group_size, provider):
        impl = ir.ops.mixer2_rms_norm_gated.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x, gate, weight = mixer2_rms_norm_gated_inputs(n_tokens, hidden_size, dtype)
        args = (x, gate, weight, epsilon, group_size)

        assert impl.supported
        assert impl.supports_args(*args)

        out_impl = impl.impl_fn(*args)
        out_native = mixer2_rms_norm_gated_native(*args)

        torch.testing.assert_close(
            out_impl, out_native, rtol=get_default_rtol(out_impl), atol=1e-3
        )

        with ir.ops.mixer2_rms_norm_gated.set_priority([provider, "native"]):
            out_impl2 = ir.ops.mixer2_rms_norm_gated(*args)

        torch.testing.assert_close(out_impl2, out_impl, rtol=0.0, atol=0.0)

        assert not impl.supports_args(x, gate, None, epsilon, group_size)
        assert not impl.supports_args(x, gate, None, epsilon, group_size=group_size)

        out_native_no_weight = mixer2_rms_norm_gated_native(
            x, gate, None, epsilon, group_size
        )
        out_native_unit_weight = mixer2_rms_norm_gated_native(
            x, gate, torch.ones_like(weight), epsilon, group_size
        )
        torch.testing.assert_close(
            out_native_no_weight,
            out_native_unit_weight,
            rtol=get_default_rtol(out_native_no_weight),
            atol=2e-4,
        )

    @pytest.mark.parametrize("provider", ["triton", "native"])
    def test_torch_opcheck(
        self, dtype, n_tokens, hidden_size, epsilon, group_size, provider
    ):
        if not ir.ops.mixer2_rms_norm_gated.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x, gate, weight = mixer2_rms_norm_gated_inputs(n_tokens, hidden_size, dtype)
        args = (x, gate, weight, epsilon, group_size)

        with ir.ops.mixer2_rms_norm_gated.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.mixer2_rms_norm_gated, args)
