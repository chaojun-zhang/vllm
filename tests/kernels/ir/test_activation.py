# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness tests for vLLM IR GELU activation ops.

Per-op correctness, semantics, and opcheck tests mirroring the
pattern established in tests/kernels/ir/test_layernorm.py.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import vllm.kernels  # noqa: F401 — registers provider implementations
from tests.ir.ir_test_utils import (
    COMMON_HIDDEN_SIZES,
    NUM_TOKENS,
    assert_close,
    clone_args,
    supported_providers,
)
from vllm import ir
from vllm.platforms import current_platform

GPGPU_DEVICE = current_platform.is_cuda_alike() or current_platform.is_xpu()

_C_GELU_NEW = math.sqrt(2.0 / math.pi)


def ref_gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return F.gelu(x, approximate=approximate)


def ref_gelu_and_mul(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate=approximate) * x[..., d:]


def ref_gelu_new(x: torch.Tensor) -> torch.Tensor:
    return (
        0.5 * x * (1.0 + torch.tanh(_C_GELU_NEW * (x + 0.044715 * torch.pow(x, 3.0))))
    )


def ref_gelu_fast(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def ref_quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_gelu_registration():
    expected = {"native": True}
    actual = {provider: impl.supported for provider, impl in ir.ops.gelu.impls.items()}
    assert actual == expected


@pytest.mark.skipif(
    not GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_gelu_and_mul_registration():
    expected = {
        "native": True,
        "vllm_c": GPGPU_DEVICE,
    }
    actual = {
        provider: impl.supported for provider, impl in ir.ops.gelu_and_mul.impls.items()
    }
    assert actual == expected


@pytest.mark.skipif(
    not GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_gelu_new_registration():
    expected = {
        "native": True,
        "vllm_c": GPGPU_DEVICE,
    }
    actual = {
        provider: impl.supported for provider, impl in ir.ops.gelu_new.impls.items()
    }
    assert actual == expected


@pytest.mark.skipif(
    not GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_gelu_fast_registration():
    expected = {
        "native": True,
        "vllm_c": GPGPU_DEVICE,
    }
    actual = {
        provider: impl.supported for provider, impl in ir.ops.gelu_fast.impls.items()
    }
    assert actual == expected


@pytest.mark.skipif(
    not GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_quick_gelu_registration():
    expected = {
        "native": True,
        "vllm_c": GPGPU_DEVICE,
    }
    actual = {
        provider: impl.supported for provider, impl in ir.ops.quick_gelu.impls.items()
    }
    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
@pytest.mark.skipif(
    not GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestGelu:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, hidden_size):
        (x,) = ir.ops.gelu.generate_inputs(num_tokens=4, hidden_size=8, dtype=dtype)
        native = ir.ops.gelu.impls["native"].impl_fn

        out = native(x)

        # Shape, dtype, device preserved
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # GELU(0) == 0
        zeros = torch.zeros_like(x)
        assert torch.all(native(zeros) == 0.0)

        # GELU is close to identity for large positive values
        large = torch.full_like(x, 20.0)
        torch.testing.assert_close(native(large), large, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.gelu))
    def test_impls(self, dtype, n_tokens, hidden_size, provider):
        impl = ir.ops.gelu.impls[provider]
        (x,) = ir.ops.gelu.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )
        args = (x,)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref = ref_gelu(*clone_args(args))
        out = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.gelu, out, ref)

        # dispatched call must match direct call exactly
        with ir.ops.gelu.set_priority([provider, "native"]):
            out_dispatched = ir.ops.gelu(*args)
        out_direct = impl.impl_fn(*args)
        torch.testing.assert_close(out_dispatched, out_direct, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["native"])
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, provider):
        if not ir.ops.gelu.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.gelu.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )

        with ir.ops.gelu.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.gelu, args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
@pytest.mark.skipif(
    not GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestGeluAndMul:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, hidden_size):
        # Use even hidden_size so gelu_and_mul can split at d = hidden_size // 2
        even_size = 8
        (x,) = ir.ops.gelu_and_mul.generate_inputs(
            num_tokens=4, hidden_size=even_size, dtype=dtype
        )
        native = ir.ops.gelu_and_mul.impls["native"].impl_fn
        d = x.shape[-1] // 2

        out = native(x)

        # Output shape is (n_tokens, d), dtype and device preserved
        assert out.shape == (*x.shape[:-1], d)
        assert out.dtype == x.dtype
        assert out.device == x.device

        # When the gate (second half) is all-ones, gelu_and_mul(x) == gelu(x[:d])
        gate_ones = torch.cat([x[..., :d], torch.ones_like(x[..., :d])], dim=-1)
        out_ones = native(gate_ones)
        expected = F.gelu(gate_ones[..., :d])
        assert_close(ir.ops.gelu_and_mul, out_ones, expected)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.gelu_and_mul))
    def test_impls(self, dtype, n_tokens, hidden_size, provider):
        impl = ir.ops.gelu_and_mul.impls[provider]
        (x,) = ir.ops.gelu_and_mul.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )
        args = (x,)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref = ref_gelu_and_mul(*clone_args(args))
        out = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.gelu_and_mul, out, ref)

        with ir.ops.gelu_and_mul.set_priority([provider, "native"]):
            out_dispatched = ir.ops.gelu_and_mul(*args)
        out_direct = impl.impl_fn(*args)
        torch.testing.assert_close(out_dispatched, out_direct, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, provider):
        if not ir.ops.gelu_and_mul.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.gelu_and_mul.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )

        with ir.ops.gelu_and_mul.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.gelu_and_mul, args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
@pytest.mark.skipif(
    not GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestGeluNew:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, hidden_size):
        (x,) = ir.ops.gelu_new.generate_inputs(num_tokens=4, hidden_size=8, dtype=dtype)
        native = ir.ops.gelu_new.impls["native"].impl_fn

        out = native(x)

        # Shape, dtype, device preserved
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # gelu_new(0) == 0 (tanh(0) == 0 so the formula evaluates to 0)
        zeros = torch.zeros_like(x)
        assert torch.all(native(zeros) == 0.0)

        # Large positive values converge to identity
        large = torch.full_like(x, 20.0)
        torch.testing.assert_close(native(large), large, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.gelu_new))
    def test_impls(self, dtype, n_tokens, hidden_size, provider):
        impl = ir.ops.gelu_new.impls[provider]
        (x,) = ir.ops.gelu_new.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )
        args = (x,)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref = ref_gelu_new(*clone_args(args))
        out = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.gelu_new, out, ref)

        with ir.ops.gelu_new.set_priority([provider, "native"]):
            out_dispatched = ir.ops.gelu_new(*args)
        out_direct = impl.impl_fn(*args)
        torch.testing.assert_close(out_dispatched, out_direct, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, provider):
        if not ir.ops.gelu_new.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.gelu_new.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )

        with ir.ops.gelu_new.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.gelu_new, args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
@pytest.mark.skipif(
    not GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestGeluFast:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, hidden_size):
        (x,) = ir.ops.gelu_fast.generate_inputs(
            num_tokens=4, hidden_size=8, dtype=dtype
        )
        native = ir.ops.gelu_fast.impls["native"].impl_fn

        out = native(x)

        # Shape, dtype, device preserved
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # gelu_fast(0) == 0
        zeros = torch.zeros_like(x)
        assert torch.all(native(zeros) == 0.0)

        # Large positive values converge to identity
        large = torch.full_like(x, 20.0)
        torch.testing.assert_close(native(large), large, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.gelu_fast))
    def test_impls(self, dtype, n_tokens, hidden_size, provider):
        impl = ir.ops.gelu_fast.impls[provider]
        (x,) = ir.ops.gelu_fast.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )
        args = (x,)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref = ref_gelu_fast(*clone_args(args))
        out = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.gelu_fast, out, ref)

        with ir.ops.gelu_fast.set_priority([provider, "native"]):
            out_dispatched = ir.ops.gelu_fast(*args)
        out_direct = impl.impl_fn(*args)
        torch.testing.assert_close(out_dispatched, out_direct, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, provider):
        if not ir.ops.gelu_fast.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.gelu_fast.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )

        with ir.ops.gelu_fast.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.gelu_fast, args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
@pytest.mark.skipif(
    not GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestQuickGelu:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, n_tokens, hidden_size):
        (x,) = ir.ops.quick_gelu.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )
        native = ir.ops.quick_gelu.impls["native"].impl_fn

        out = native(x)

        # Shape, dtype, device preserved
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # quick_gelu(0) = 0 * sigmoid(0) == 0
        zeros = torch.zeros_like(x)
        assert torch.all(native(zeros) == 0.0)

        # Large positive values: x * sigmoid(1.702 * x) ≈ x for large x
        large = torch.full_like(x, 20.0)
        torch.testing.assert_close(native(large), large, rtol=1e-3, atol=1e-2)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.quick_gelu))
    def test_impls(self, dtype, n_tokens, hidden_size, provider):
        impl = ir.ops.quick_gelu.impls[provider]
        (x,) = ir.ops.quick_gelu.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )
        args = (x,)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref = ref_quick_gelu(*clone_args(args))
        out = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.quick_gelu, out, ref)

        with ir.ops.quick_gelu.set_priority([provider, "native"]):
            out_dispatched = ir.ops.quick_gelu(*args)
        out_direct = impl.impl_fn(*args)
        torch.testing.assert_close(out_dispatched, out_direct, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "native"])
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, provider):
        if not ir.ops.quick_gelu.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.quick_gelu.generate_inputs(
            num_tokens=n_tokens, hidden_size=hidden_size, dtype=dtype
        )

        with ir.ops.quick_gelu.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.quick_gelu, args)
