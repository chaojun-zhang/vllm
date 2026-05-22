# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for bce9bc564615e25c15bc1d85800a03663b380364.

Without .contiguous(), layer.weight.t() is a Fortran-order view that shares
the checkpoint's storage.  oneDNN requires C-contiguous, 64-byte-aligned
tensors; a non-contiguous view sharing potentially-misaligned checkpoint
storage therefore triggers alignment/layout errors at inference time.

These tests run on CPU (no XPU hardware required) and verify:
  1. The FIXED process_weights_after_loading stores a C-contiguous weight.
  2. The BUGGY variant (t() without .contiguous()) would leave the weight
     non-contiguous, sharing storage with the original checkpoint tensor.
  3. Misaligned checkpoint storage is correctly handled by the fix.

The tests are marked xfail on the "buggy" variants so CI documents the
regression without requiring hardware.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.kernels.linear.mxfp8.xpu import XPUMxFp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ONEDNN_ALIGN = 64  # oneDNN / XPU allocator minimum alignment (bytes)


def _make_kernel() -> XPUMxFp8LinearKernel:
    """Instantiate the kernel bypassing the XPU platform guard."""
    cfg = Mxfp8LinearLayerConfig()
    with patch(
        "vllm.model_executor.kernels.linear.mxfp8.xpu.current_platform"
    ) as mock_platform:
        mock_platform.is_xpu.return_value = True
        return XPUMxFp8LinearKernel(cfg)


def _make_layer(weight: torch.Tensor, weight_scale: torch.Tensor) -> nn.Module:
    """Build a minimal nn.Module that mimics a loaded quantized layer."""
    layer = nn.Module()
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = weight_scale
    return layer


def _misaligned_fp8_weight(N: int, K: int) -> torch.Tensor:
    """Return a [N, K] uint8 tensor whose data_ptr is NOT 64-byte aligned.

    Simulates a checkpoint tensor that was stored with only 8-byte alignment
    (e.g. packed inside a SafeTensors file or a sharded parameter blob).
    """
    # Backing store: align to 8 bytes but not to 64 bytes.
    backing = torch.zeros(N * K + _ONEDNN_ALIGN, dtype=torch.uint8)
    # Find an offset that is 8-byte aligned but *not* 64-byte aligned.
    offset = 0
    base_ptr = backing.data_ptr()
    for off in range(0, _ONEDNN_ALIGN, 8):
        if (base_ptr + off) % _ONEDNN_ALIGN != 0:
            offset = off
            break
    assert offset != 0, "Could not construct a misaligned tensor for this test"
    sliced = backing[offset : offset + N * K].view(N, K)
    assert sliced.data_ptr() % _ONEDNN_ALIGN != 0, "precondition: misaligned"
    return sliced


# ---------------------------------------------------------------------------
# Tests: FIXED behaviour (current code with .contiguous())
# ---------------------------------------------------------------------------


def test_process_weights_weight_is_c_contiguous():
    """After process_weights_after_loading the weight must be C-contiguous."""
    N, K = 64, 32
    raw_weight = torch.zeros(N, K, dtype=torch.uint8)  # checkpoint [N, K]
    weight_scale = torch.zeros(N, K // 32, dtype=torch.uint8)

    layer = _make_layer(raw_weight.clone(), weight_scale)
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)

    assert layer.weight.is_contiguous(), (
        "weight must be C-contiguous after process_weights_after_loading "
        "so oneDNN does not encounter a Fortran-order layout"
    )


def test_process_weights_weight_shape_is_K_N():
    """The stored weight must be transposed to [K, N] for GEMM."""
    N, K = 64, 32
    raw_weight = torch.zeros(N, K, dtype=torch.uint8)
    weight_scale = torch.zeros(N, K // 32, dtype=torch.uint8)

    layer = _make_layer(raw_weight.clone(), weight_scale)
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)

    assert layer.weight.shape == (K, N), (
        f"expected [K={K}, N={N}], got {list(layer.weight.shape)}"
    )


def test_process_weights_does_not_share_checkpoint_storage():
    """The stored weight must NOT alias the original checkpoint tensor.

    If the weight shared storage with the checkpoint tensor any misalignment
    in the checkpoint would be inherited, violating oneDNN's alignment
    requirement.
    """
    N, K = 64, 32
    raw_weight = torch.zeros(N, K, dtype=torch.uint8)
    weight_scale = torch.zeros(N, K // 32, dtype=torch.uint8)
    original_data_ptr = raw_weight.data_ptr()

    layer = _make_layer(raw_weight, weight_scale)  # raw_weight IS the parameter
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)

    assert layer.weight.data_ptr() != original_data_ptr, (
        "weight data_ptr must differ from the checkpoint tensor — "
        "sharing storage would inherit potential misalignment"
    )


def test_process_weights_64byte_aligned_after_misaligned_checkpoint():
    """Even when the checkpoint tensor is misaligned, the stored weight
    must satisfy oneDNN's 64-byte alignment requirement."""
    N, K = 64, 32
    raw_weight = _misaligned_fp8_weight(N, K)
    weight_scale = torch.zeros(N, K // 32, dtype=torch.uint8)

    assert raw_weight.data_ptr() % _ONEDNN_ALIGN != 0, "test precondition"

    layer = _make_layer(raw_weight, weight_scale)
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)

    assert layer.weight.data_ptr() % _ONEDNN_ALIGN == 0, (
        f"weight data_ptr {hex(layer.weight.data_ptr())} is not 64-byte "
        "aligned; oneDNN will raise an alignment error at inference time"
    )


def test_process_weights_weight_scale_is_c_contiguous():
    """weight_scale must also be C-contiguous (same fix was already present)."""
    N, K = 64, 32
    raw_weight = torch.zeros(N, K, dtype=torch.uint8)
    weight_scale = torch.zeros(N, K // 32, dtype=torch.uint8)

    layer = _make_layer(raw_weight.clone(), weight_scale)
    kernel = _make_kernel()
    kernel.process_weights_after_loading(layer)

    assert layer.weight_scale.is_contiguous(), (
        "weight_scale must be C-contiguous after process_weights_after_loading"
    )


# ---------------------------------------------------------------------------
# Tests: BUGGY behaviour — document what would break WITHOUT the fix
# These are marked xfail: they assert the BAD property and are expected to
# *pass* in a regressed codebase, but must *fail* (i.e. xfail) with the fix.
# ---------------------------------------------------------------------------


def _buggy_process_weights(layer: nn.Module) -> None:
    """Replicate the pre-fix logic: .t() without .contiguous()."""
    from vllm.model_executor.utils import replace_parameter

    weight_scale = layer.weight_scale.view(torch.uint8)
    weight_scale = weight_scale.t().contiguous()
    # BUG: missing .contiguous() — stores a Fortran-order view
    replace_parameter(layer, "weight", layer.weight.t())
    replace_parameter(layer, "weight_scale", weight_scale.data)


@pytest.mark.xfail(
    strict=True,
    reason="Without .contiguous() the weight is NOT C-contiguous (Fortran-order view)",
)
def test_buggy_weight_is_not_c_contiguous():
    """Demonstrates that the pre-fix code leaves a non-contiguous weight."""
    N, K = 64, 32
    raw_weight = torch.zeros(N, K, dtype=torch.uint8)
    weight_scale = torch.zeros(N, K // 32, dtype=torch.uint8)

    layer = _make_layer(raw_weight, weight_scale)
    _buggy_process_weights(layer)

    # This assertion FAILS with the buggy code (strides are Fortran-order).
    assert layer.weight.is_contiguous()


@pytest.mark.xfail(
    strict=True,
    reason="Without .contiguous() the weight shares the checkpoint storage",
)
def test_buggy_weight_shares_checkpoint_storage():
    """Demonstrates that the pre-fix .t() aliases the original tensor."""
    N, K = 64, 32
    raw_weight = torch.zeros(N, K, dtype=torch.uint8)
    original_data_ptr = raw_weight.data_ptr()
    weight_scale = torch.zeros(N, K // 32, dtype=torch.uint8)

    layer = _make_layer(raw_weight, weight_scale)
    _buggy_process_weights(layer)

    # This assertion FAILS: data_ptr IS the same as the checkpoint.
    assert layer.weight.data_ptr() != original_data_ptr


@pytest.mark.xfail(
    strict=True,
    reason="Without .contiguous() a misaligned checkpoint propagates to inference",
)
def test_buggy_misaligned_checkpoint_propagates():
    """Demonstrates that misaligned checkpoint storage survives into the weight."""
    N, K = 64, 32
    raw_weight = _misaligned_fp8_weight(N, K)
    weight_scale = torch.zeros(N, K // 32, dtype=torch.uint8)

    assert raw_weight.data_ptr() % _ONEDNN_ALIGN != 0

    layer = _make_layer(raw_weight, weight_scale)
    _buggy_process_weights(layer)

    # This assertion FAILS: the misalignment is still present.
    assert layer.weight.data_ptr() % _ONEDNN_ALIGN == 0
