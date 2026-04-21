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


class Mixer2Model(nn.Module):
    def __init__(self, hidden_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.weight = torch.ones(hidden_size, dtype=torch.bfloat16)

    def forward(self, x, gate):
        x1 = x + 4.0
        x2 = ops.mixer2_rms_norm_gated(x1, gate, self.weight, 1e-5)
        x3 = x2 * 5.0
        # grouped normalization should still use the prioritized provider
        x4 = ops.mixer2_rms_norm_gated(
            x3, gate, self.weight, 1e-5, self.hidden_size // 2
        )
        x5 = x4 / 2.0
        # dispatch to native due to weight=None
        x6 = ops.mixer2_rms_norm_gated(x5, gate, None, 1e-5)
        return x6 + 3.0


@pytest.mark.parametrize(
    "mixer2_provider", ops.mixer2_rms_norm_gated.supported_providers()
)
def test_lowering_mixer2_rms_norm_gated(mixer2_provider, default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = Mixer2Model()
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    gate = torch.randn(8, 16, dtype=torch.bfloat16)
    with (
        ops.mixer2_rms_norm_gated.set_priority([mixer2_provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output = compiled_model(x, gate)
        output_unlowered = compiled_unlowered_model(x, gate)

    selected = lowering_pass.selected_impls["mixer2_rms_norm_gated"]
    assert len(selected) == 3
    assert selected["mixer2_rms_norm_gated"] == mixer2_provider
    assert selected["mixer2_rms_norm_gated_1"] == mixer2_provider
    assert selected["mixer2_rms_norm_gated_2"] == "native"

    # Compiled function guards on global value, avoid recompilation
    with ir.enable_torch_wrap(True):
        output2 = compiled_model(x, gate)

    torch.testing.assert_close(output_unlowered, output)
    torch.testing.assert_close(output_unlowered, output2)
