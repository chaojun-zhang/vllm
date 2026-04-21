# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

import vllm.kernels  # noqa: F401 to register kernels
from vllm import ir
from vllm.compilation.passes.ir.lowering_pass import (
    VllmIRLoweringPass,
)
from vllm.config import get_current_vllm_config
from vllm.ir import ops
from vllm.platforms import current_platform

from ...backend import TestBackend


class Model(nn.Module):
    def __init__(self, hidden_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.weight = torch.ones(hidden_size, dtype=torch.bfloat16)

    def forward(self, x):
        x1 = x + 4.0
        x2 = ops.rms_norm(x1, self.weight, 1e-5)
        x3 = x2 * 5.0
        # no weight
        x4 = ops.rms_norm(x3, None, 1e-5)
        x5 = x4 / 2.0
        # dispatch to native due to variance_size parameter
        x6 = ops.rms_norm(x5, self.weight, 1e-5, self.hidden_size // 2)
        return x6 + 3.0


@pytest.mark.parametrize("rms_provider", ops.rms_norm.supported_providers())
def test_lowering_rms_norm(rms_provider, default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = Model()
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    with (
        ops.rms_norm.set_priority([rms_provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output = compiled_model(x)
        output_unlowered = compiled_unlowered_model(x)

    selected = lowering_pass.selected_impls["rms_norm"]
    assert len(selected) == 3
    assert selected["rms_norm"] == rms_provider
    assert selected["rms_norm_1"] == rms_provider
    assert selected["rms_norm_2"] == "native"

    # Compiled function guards on global value, avoid recompilation
    with ir.enable_torch_wrap(True):
        output2 = compiled_model(x)

    torch.testing.assert_close(output_unlowered, output)
    torch.testing.assert_close(output_unlowered, output2)


class RMSNormGatedModel(nn.Module):
    def __init__(self, hidden_size=128, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = torch.ones(hidden_size, dtype=dtype)
        self.bias = torch.zeros(hidden_size, dtype=dtype)

    def forward(self, x, z):
        x1 = x + 4.0
        # op 1: basic call with z, bias=None — dispatches to provider
        x2 = ops.rms_norm_gated(x1, self.weight, None, z, 1e-5, None, False, "swish")
        x3 = x2 * 5.0
        # op 2: with bias — dispatches to provider
        x4 = ops.rms_norm_gated(
            x3, self.weight, self.bias, z, 1e-5, None, False, "swish"
        )
        x5 = x4 / 2.0
        # op 3: z=None — dispatches to provider (triton supports all args)
        x6 = ops.rms_norm_gated(x5, self.weight, None, None, 1e-5, None, False, "swish")
        return x6 + 3.0


@pytest.mark.parametrize("rms_gated_provider", ops.rms_norm_gated.supported_providers())
def test_lowering_rms_norm_gated(rms_gated_provider, default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = RMSNormGatedModel()
    torch.manual_seed(0)
    x = torch.randn(8, model.hidden_size, dtype=torch.bfloat16)
    z = torch.randn(8, model.hidden_size, dtype=torch.bfloat16)
    with (
        ops.rms_norm_gated.set_priority([rms_gated_provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output = compiled_model(x, z)
        output_unlowered = compiled_unlowered_model(x, z)

    selected = lowering_pass.selected_impls["rms_norm_gated"]
    assert len(selected) == 3
    assert selected["rms_norm_gated"] == rms_gated_provider
    assert selected["rms_norm_gated_1"] == rms_gated_provider
    assert selected["rms_norm_gated_2"] == rms_gated_provider

    # Compiled function guards on global value, avoid recompilation
    with ir.enable_torch_wrap(True):
        output2 = compiled_model(x, z)

    torch.testing.assert_close(output_unlowered, output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(output_unlowered, output2, atol=1e-2, rtol=1e-2)
