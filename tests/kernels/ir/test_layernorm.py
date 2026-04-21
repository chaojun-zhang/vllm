# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

# This registers op implementations
import vllm.kernels  # noqa: F401
from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm import ir
from vllm.platforms import current_platform

GPGPU_DEVICE = current_platform.is_cuda_alike() or current_platform.is_xpu()


def rms_norm_inputs(n_tokens: int, hidden_size: int, dtype: torch.dtype):
    x = torch.randn(n_tokens, hidden_size, dtype=dtype)
    weight = torch.rand(hidden_size, dtype=dtype)
    return x, weight


rms_norm_native = ir.ops.rms_norm.impls["native"].impl_fn


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


# ============= rms_norm_gated tests =============


def rms_norm_gated_inputs(n_tokens: int, hidden_size: int, dtype: torch.dtype):
    x = torch.randn(n_tokens, hidden_size, dtype=dtype)
    weight = torch.rand(hidden_size, dtype=dtype) + 0.5
    bias = torch.randn(hidden_size, dtype=dtype) * 0.1
    z = torch.randn(n_tokens, hidden_size, dtype=dtype)
    return x, weight, bias, z


def rms_norm_gated_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor | None,
    epsilon: float,
    group_size: int | None = None,
    norm_before_gate: bool = False,
    activation: str = "swish",
) -> torch.Tensor:
    """Independent PyTorch reference for rms_norm_gated."""
    orig_dtype = x.dtype
    x = x.float()
    w = weight.float()
    b = bias.float() if bias is not None else None
    g = z.float() if z is not None else None

    def gate(inp, gate_val):
        if activation in ("swish", "silu"):
            return inp * torch.nn.functional.silu(gate_val)
        elif activation == "sigmoid":
            return inp * torch.sigmoid(gate_val)
        return inp

    if g is not None and not norm_before_gate:
        x = gate(x, g)

    if group_size is None:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
        out = x / rms * w
    else:
        x_g = x.reshape(*x.shape[:-1], -1, group_size)
        rms = torch.sqrt(x_g.pow(2).mean(dim=-1, keepdim=True) + epsilon)
        out = (x_g / rms).reshape(x.shape) * w

    if b is not None:
        out = out + b

    if g is not None and norm_before_gate:
        out = gate(out, g)

    return out.to(orig_dtype)


rms_norm_gated_native = ir.ops.rms_norm_gated.impls["native"].impl_fn


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_rms_norm_gated_registration():
    expected = {
        "native": True,
        "triton": GPGPU_DEVICE,
    }

    actual = {
        provider: impl.supported
        for provider, impl in ir.ops.rms_norm_gated.impls.items()
    }

    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 17])
@pytest.mark.parametrize("hidden_size", [16, 4096])
@pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
@pytest.mark.parametrize("group_size", [None, 4])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestRMSNormGated:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, hidden_size, epsilon, group_size):
        x, weight, bias, z = rms_norm_gated_inputs(4, 8, dtype)

        # Basic call
        out = rms_norm_gated_native(x, weight, None, z, epsilon)

        # Shape, dtype, device
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # Scale invariance: rms_norm_gated(c*x, w, None, z, eps)
        # ≈ rms_norm_gated(x, w, None, z, eps) for c > 0
        # (approximate due to epsilon; gating amplifies the effect)
        out2 = rms_norm_gated_native(x * 2.0, weight, None, z, epsilon)
        torch.testing.assert_close(out2, out, rtol=1e-2, atol=5e-3)

        # Compare against independent reference
        ref = rms_norm_gated_reference(x, weight, None, z, epsilon)
        torch.testing.assert_close(
            out, ref, rtol=get_default_rtol(out), atol=get_default_atol(out)
        )

        # bias=None should be equivalent to bias=zeros
        out_no_bias = rms_norm_gated_native(x, weight, None, z, epsilon)
        out_zero_bias = rms_norm_gated_native(
            x, weight, torch.zeros_like(weight), z, epsilon
        )
        torch.testing.assert_close(out_no_bias, out_zero_bias)

        # z=None should match rms_norm(x, weight, eps)
        out_no_z = rms_norm_gated_native(x, weight, None, None, epsilon)
        out_rms = rms_norm_native(x, weight, epsilon)
        torch.testing.assert_close(
            out_no_z, out_rms, rtol=get_default_rtol(out_rms), atol=1e-3
        )

        # norm_before_gate=True vs False should differ when z is given
        out_before = rms_norm_gated_native(
            x, weight, None, z, epsilon, None, True, "swish"
        )
        ref_before = rms_norm_gated_reference(
            x, weight, None, z, epsilon, None, True, "swish"
        )
        torch.testing.assert_close(
            out_before,
            ref_before,
            rtol=get_default_rtol(out_before),
            atol=get_default_atol(out_before),
        )

        # activation="sigmoid"
        out_sig = rms_norm_gated_native(
            x, weight, None, z, epsilon, None, False, "sigmoid"
        )
        ref_sig = rms_norm_gated_reference(
            x, weight, None, z, epsilon, None, False, "sigmoid"
        )
        torch.testing.assert_close(
            out_sig,
            ref_sig,
            rtol=get_default_rtol(out_sig),
            atol=get_default_atol(out_sig),
        )

        # group_size=hidden_size should be equivalent to group_size=None
        hs = x.shape[-1]
        out_gs_none = rms_norm_gated_native(x, weight, None, z, epsilon, None)
        out_gs_full = rms_norm_gated_native(x, weight, None, z, epsilon, hs)
        torch.testing.assert_close(
            out_gs_none,
            out_gs_full,
            rtol=get_default_rtol(out_gs_none),
            atol=get_default_atol(out_gs_none),
        )

    @pytest.mark.parametrize("provider", ["triton"])
    def test_impls(self, dtype, n_tokens, hidden_size, epsilon, group_size, provider):
        impl = ir.ops.rms_norm_gated.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x, weight, bias, z = rms_norm_gated_inputs(n_tokens, hidden_size, dtype)
        args = (x, weight, bias, z, epsilon, group_size, False, "swish")

        assert impl.supports_args(*args)

        out_impl = impl.impl_fn(*args)
        out_native = rms_norm_gated_native(*args)

        torch.testing.assert_close(
            out_impl,
            out_native,
            rtol=get_default_rtol(out_impl),
            atol=get_default_atol(out_impl),
        )

        # Check dispatched call matches direct call
        with ir.ops.rms_norm_gated.set_priority([provider, "native"]):
            out_dispatch = ir.ops.rms_norm_gated(*args)
        torch.testing.assert_close(out_dispatch, out_impl, rtol=0.0, atol=0.0)

        # Test z=None path
        args_no_z = (x, weight, None, None, epsilon, group_size, False, "swish")
        out_impl_no_z = impl.impl_fn(*args_no_z)
        out_native_no_z = rms_norm_gated_native(*args_no_z)
        torch.testing.assert_close(
            out_impl_no_z,
            out_native_no_z,
            rtol=get_default_rtol(out_impl_no_z),
            atol=get_default_atol(out_impl_no_z),
        )

        # Test activation="sigmoid"
        args_sig = (x, weight, bias, z, epsilon, group_size, False, "sigmoid")
        out_impl_sig = impl.impl_fn(*args_sig)
        out_native_sig = rms_norm_gated_native(*args_sig)
        torch.testing.assert_close(
            out_impl_sig,
            out_native_sig,
            rtol=get_default_rtol(out_impl_sig),
            atol=get_default_atol(out_impl_sig),
        )

    @pytest.mark.parametrize("provider", ["triton", "native"])
    def test_torch_opcheck(
        self, dtype, n_tokens, hidden_size, epsilon, group_size, provider
    ):
        if not ir.ops.rms_norm_gated.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x, weight, bias, z = rms_norm_gated_inputs(n_tokens, hidden_size, dtype)
        args = (x, weight, bias, z, epsilon, group_size, False, "swish")

        with ir.ops.rms_norm_gated.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.rms_norm_gated, args)
