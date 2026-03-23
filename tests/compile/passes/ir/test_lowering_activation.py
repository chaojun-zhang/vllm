# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

import vllm.kernels  # noqa: F401 to register kernels
from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm import ir
from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
from vllm.config import get_current_vllm_config
from vllm.ir import ops
from vllm.platforms import current_platform

from ...backend import TestBackend

class GeluModel(nn.Module):
    """Two gelu calls: one with approximate='none', one with approximate='tanh'."""

    def forward(self, x):
        x1 = x + 4.0
        x2 = ops.gelu(x1)                      # node: gelu
        x3 = x2 * 5.0
        x4 = ops.gelu(x3, approximate="tanh")  # node: gelu_1
        return x4 + 3.0


class GeluAndMulModel(nn.Module):
    """Two gelu_and_mul calls on the same input (different offsets)."""

    def forward(self, x):
        # Use abs() + offset to keep all gate values positive, preventing
        # cancellation between y1 and y2 in the sum.  Cancellation would
        # make the sum near-zero while individual errors stay O(eps_bfloat16 *
        # magnitude), causing atol=1e-3 to be exceeded.
        xp = x.abs() + 1.0
        y1 = ops.gelu_and_mul(xp)          # node: gelu_and_mul
        y2 = ops.gelu_and_mul(xp + 0.5)   # node: gelu_and_mul_1
        return y1 + y2


class GeluNewModel(nn.Module):
    """Two gelu_new calls chained."""

    def forward(self, x):
        x1 = x + 4.0
        x2 = ops.gelu_new(x1)  # node: gelu_new
        x3 = x2 * 5.0
        x4 = ops.gelu_new(x3)  # node: gelu_new_1
        return x4 + 3.0


class GeluFastModel(nn.Module):
    """Two gelu_fast calls chained."""

    def forward(self, x):
        x1 = x + 4.0
        x2 = ops.gelu_fast(x1)  # node: gelu_fast
        x3 = x2 * 5.0
        x4 = ops.gelu_fast(x3)  # node: gelu_fast_1
        return x4 + 3.0


class QuickGeluModel(nn.Module):
    """Two quick_gelu calls chained."""

    def forward(self, x):
        x1 = x + 4.0
        x2 = ops.quick_gelu(x1)  # node: quick_gelu
        x3 = x2 * 5.0
        x4 = ops.quick_gelu(x3)  # node: quick_gelu_1
        return x4 + 3.0


def _assert_outputs(ref, output_lowered, output_unlowered, output2):
    atol = get_default_atol(output_lowered)
    rtol = get_default_rtol(output_lowered)
    torch.testing.assert_close(ref, output_lowered, atol=atol, rtol=rtol)
    torch.testing.assert_close(ref, output_unlowered, atol=atol, rtol=rtol)
    # Same compiled kernel, same input → must be bit-for-bit deterministic
    torch.testing.assert_close(output_lowered, output2, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("provider", ops.gelu.supported_providers())
def test_lowering_gelu(provider, default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = GeluModel()
    x = torch.randn(8, 16, dtype=torch.bfloat16)

    with ops.gelu.set_priority([provider, "native"]), ir.enable_torch_wrap(True):
        ref = model(x)

    with (
        ops.gelu.set_priority([provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output_lowered = compiled_model(x)
        output_unlowered = compiled_unlowered_model(x)

    selected = lowering_pass.selected_impls["gelu"]
    assert len(selected) == 2
    assert selected["gelu"] == provider
    assert selected["gelu_1"] == provider

    # Compiled function guards on global value, avoid recompilation
    with ir.enable_torch_wrap(True):
        output2 = compiled_model(x)

    _assert_outputs(ref, output_lowered, output_unlowered, output2)


@pytest.mark.parametrize("provider", ops.gelu_and_mul.supported_providers())
def test_lowering_gelu_and_mul(provider, default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = GeluAndMulModel()
    # Last dim must be even: gelu_and_mul splits at d = hidden_size // 2
    x = torch.randn(8, 16, dtype=torch.bfloat16)

    with ops.gelu_and_mul.set_priority([provider, "native"]), ir.enable_torch_wrap(True):
        ref = model(x)

    with (
        ops.gelu_and_mul.set_priority([provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output_lowered = compiled_model(x)
        output_unlowered = compiled_unlowered_model(x)

    selected = lowering_pass.selected_impls["gelu_and_mul"]
    assert len(selected) == 2
    assert selected["gelu_and_mul"] == provider
    assert selected["gelu_and_mul_1"] == provider

    with ir.enable_torch_wrap(True):
        output2 = compiled_model(x)

    _assert_outputs(ref, output_lowered, output_unlowered, output2)


@pytest.mark.parametrize("provider", ops.gelu_new.supported_providers())
def test_lowering_gelu_new(provider, default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = GeluNewModel()
    x = torch.randn(8, 16, dtype=torch.bfloat16)

    with ops.gelu_new.set_priority([provider, "native"]), ir.enable_torch_wrap(True):
        ref = model(x)

    with (
        ops.gelu_new.set_priority([provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output_lowered = compiled_model(x)
        output_unlowered = compiled_unlowered_model(x)

    selected = lowering_pass.selected_impls["gelu_new"]
    assert len(selected) == 2
    assert selected["gelu_new"] == provider
    assert selected["gelu_new_1"] == provider

    with ir.enable_torch_wrap(True):
        output2 = compiled_model(x)

    _assert_outputs(ref, output_lowered, output_unlowered, output2)


@pytest.mark.parametrize("provider", ops.gelu_fast.supported_providers())
def test_lowering_gelu_fast(provider, default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = GeluFastModel()
    x = torch.randn(8, 16, dtype=torch.bfloat16)

    with ops.gelu_fast.set_priority([provider, "native"]), ir.enable_torch_wrap(True):
        ref = model(x)

    with (
        ops.gelu_fast.set_priority([provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output_lowered = compiled_model(x)
        output_unlowered = compiled_unlowered_model(x)

    selected = lowering_pass.selected_impls["gelu_fast"]
    assert len(selected) == 2
    assert selected["gelu_fast"] == provider
    assert selected["gelu_fast_1"] == provider

    with ir.enable_torch_wrap(True):
        output2 = compiled_model(x)

    _assert_outputs(ref, output_lowered, output_unlowered, output2)


@pytest.mark.parametrize("provider", ops.quick_gelu.supported_providers())
def test_lowering_quick_gelu(provider, default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = QuickGeluModel()
    x = torch.randn(8, 16, dtype=torch.bfloat16)

    with ops.quick_gelu.set_priority([provider, "native"]), ir.enable_torch_wrap(True):
        ref = model(x)

    with (
        ops.quick_gelu.set_priority([provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output_lowered = compiled_model(x)
        output_unlowered = compiled_unlowered_model(x)

    selected = lowering_pass.selected_impls["quick_gelu"]
    assert len(selected) == 2
    assert selected["quick_gelu"] == provider
    assert selected["quick_gelu_1"] == provider

    with ir.enable_torch_wrap(True):
        output2 = compiled_model(x)

    _assert_outputs(ref, output_lowered, output_unlowered, output2)
