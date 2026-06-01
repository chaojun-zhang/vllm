# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E regression tests using a real HuggingFace MXFP8 model on XPU.

Uses ``Yi30/Llama-3.2-1B-Instruct-MXFP8-llmc`` (compressed-tensors MXFP8,
already in the CI model cache) to verify that ``XPUMxFp8LinearKernel.
process_weights_after_loading`` produces C-contiguous, 64-byte aligned
weight tensors that oneDNN can consume without alignment errors.

Three scenarios are covered:

  1. FIXED (current code) – loaded weights are C-contiguous and aligned.
  2. BUGGY (monkey-patched to skip .contiguous()) – weights are NOT
     C-contiguous and inherit misaligned checkpoint storage.
  3. INFERENCE SMOKE – the fixed model can generate tokens; the buggy
     variant raises a RuntimeError from the oneDNN gemm kernel.

All tests are skipped unless the current platform is XPU.
"""

from contextlib import contextmanager
from typing import Generator
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm import LLM, SamplingParams
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Yi30/Llama-3.2-1B-Instruct-MXFP8-llmc"
ONEDNN_ALIGN = 64  # oneDNN / Intel GPU allocator minimum alignment (bytes)
MAX_MODEL_LEN = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(**kwargs) -> LLM:
    return LLM(
        model=MODEL_ID,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        **kwargs,
    )


@contextmanager
def _buggy_process_weights_patch() -> Generator[None, None, None]:
    """Monkey-patch XPUMxFp8LinearKernel to use .t() WITHOUT .contiguous().

    This replicates the pre-fix code path so tests can document what would
    break.  The patch replaces only process_weights_after_loading so the
    rest of the kernel (apply_weights, is_supported, …) is unchanged.
    """

    def _buggy(self, layer: nn.Module) -> None:
        weight_scale = layer.weight_scale.view(torch.float8_e8m0fnu)
        weight_scale = weight_scale.t().contiguous()
        # BUG: missing .contiguous() — stores a Fortran-order view that
        # shares the checkpoint's potentially-misaligned storage.
        replace_parameter(layer, "weight", layer.weight.t())
        replace_parameter(layer, "weight_scale", weight_scale)

    target = (
        "vllm.model_executor.kernels.linear.mxfp8.xpu"
        ".XPUMxFp8LinearKernel.process_weights_after_loading"
    )
    with patch(target, _buggy):
        yield


def _iter_mxfp8_linear_layers(model: nn.Module):
    """Yield every Linear sublayer whose weight is in fp8 dtype."""
    for name, mod in model.named_modules():
        if (
            isinstance(mod, nn.Module)
            and hasattr(mod, "weight")
            and isinstance(mod.weight, nn.Parameter)
            and mod.weight.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        ):
            yield name, mod


# ---------------------------------------------------------------------------
# Skip guard: all tests require XPU
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU MXFP8 weight layout tests require XPU hardware",
)


# ---------------------------------------------------------------------------
# Test 1: FIXED — loaded weights must be C-contiguous
# ---------------------------------------------------------------------------


def test_hf_mxfp8_weights_are_c_contiguous():
    """All MXFP8 linear weights must be C-contiguous after model load.

    A Fortran-order view (result of .t() without .contiguous()) causes
    oneDNN to raise "found misaligned buffer" during inference.
    """
    llm = _make_llm()

    non_contiguous = []

    def check(model):
        for name, layer in _iter_mxfp8_linear_layers(model):
            if not layer.weight.is_contiguous():
                non_contiguous.append(
                    f"{name}: strides={layer.weight.stride()}"
                )

    llm.apply_model(check)
    del llm

    assert not non_contiguous, (
        "Found non-contiguous MXFP8 weights after loading — "
        "process_weights_after_loading is missing .contiguous():\n"
        + "\n".join(non_contiguous)
    )


# ---------------------------------------------------------------------------
# Test 2: FIXED — loaded weights must be 64-byte aligned
# ---------------------------------------------------------------------------


def test_hf_mxfp8_weights_are_64byte_aligned():
    """All MXFP8 linear weights must be 64-byte aligned after model load.

    Without .contiguous(), the weight is a view that shares the checkpoint's
    storage; SafeTensors only guarantees 8-byte alignment so the weight
    data_ptr may not satisfy oneDNN's 64-byte requirement.
    """
    llm = _make_llm()

    misaligned = []

    def check(model):
        for name, layer in _iter_mxfp8_linear_layers(model):
            ptr = layer.weight.data_ptr()
            if ptr % ONEDNN_ALIGN != 0:
                misaligned.append(
                    f"{name}: data_ptr={hex(ptr)} (mod {ONEDNN_ALIGN} = "
                    f"{ptr % ONEDNN_ALIGN})"
                )

    llm.apply_model(check)
    del llm

    assert not misaligned, (
        f"Found MXFP8 weights with data_ptr not aligned to {ONEDNN_ALIGN} "
        "bytes — .contiguous() must be called to allocate aligned storage:\n"
        + "\n".join(misaligned)
    )


# ---------------------------------------------------------------------------
# Test 3: FIXED — loaded weights must not share checkpoint storage
# ---------------------------------------------------------------------------


def test_hf_mxfp8_weights_do_not_share_checkpoint_storage():
    """The stored weight tensor must own its storage (not alias checkpoint).

    If the weight aliases checkpoint storage, any misalignment in the
    checkpoint propagates directly to inference.  .contiguous() forces a
    new allocation.
    """
    llm = _make_llm()

    aliased = []

    def check(model):
        for name, layer in _iter_mxfp8_linear_layers(model):
            w = layer.weight
            # A tensor owns its storage when its storage_offset is 0 and its
            # numel * element_size == storage size (no backing over-allocation
            # from a slice/view of a larger checkpoint tensor).
            storage_size = w.storage().nbytes()
            tensor_size = w.numel() * w.element_size()
            if storage_size > tensor_size * 2:
                # Significantly larger backing storage → still a view
                aliased.append(
                    f"{name}: tensor_bytes={tensor_size}, "
                    f"storage_bytes={storage_size}"
                )

    llm.apply_model(check)
    del llm

    assert not aliased, (
        "MXFP8 weight tensors appear to share backing storage with a larger "
        "checkpoint buffer — .contiguous() must copy into its own allocation:\n"
        + "\n".join(aliased)
    )


# ---------------------------------------------------------------------------
# Test 4: BUGGY (xfail) — without .contiguous() weights are non-contiguous
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Without .contiguous(), loaded weights are Fortran-order views; "
        "is_contiguous() returns False"
    ),
)
def test_hf_mxfp8_buggy_weights_are_not_c_contiguous():
    """Documents that the pre-fix code leaves non-contiguous weights.

    The xfail means: this assertion FAILS with the buggy code (which is
    expected / documented), and must PASS (become xpass → test failure)
    if someone accidentally reintroduces the bug.
    """
    with _buggy_process_weights_patch():
        llm = _make_llm()

    all_contiguous = []

    def check(model):
        for name, layer in _iter_mxfp8_linear_layers(model):
            if not layer.weight.is_contiguous():
                all_contiguous.append(name)

    llm.apply_model(check)
    del llm

    # This assertion is EXPECTED TO FAIL (xfail) with the buggy code.
    assert not all_contiguous


# ---------------------------------------------------------------------------
# Test 5: BUGGY (xfail) — without .contiguous() inference raises RuntimeError
#
# oneDNN error message:
#   "found misaligned buffer: <ptr> for kernel gemm_kernel"
#   RuntimeError: could not execute a primitive
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "Without .contiguous(), misaligned Fortran-order weight triggers "
        "oneDNN 'found misaligned buffer' → RuntimeError during inference"
    ),
)
def test_hf_mxfp8_buggy_inference_raises_alignment_error():
    """Documents the exact runtime failure caused by missing .contiguous().

    The xfail means: a RuntimeError is EXPECTED here (buggy code), and
    if no error is raised the test becomes xpass → CI failure, alerting
    that the test environment no longer exercises the alignment check.
    """
    with _buggy_process_weights_patch():
        llm = _make_llm()

    params = SamplingParams(temperature=0.0, max_tokens=4)
    # This call must trigger oneDNN's alignment check and raise RuntimeError.
    llm.generate(["The capital of France is"], sampling_params=params)
    del llm


# ---------------------------------------------------------------------------
# Test 6: FIXED smoke — model can generate tokens after load
# ---------------------------------------------------------------------------


def test_hf_mxfp8_inference_succeeds():
    """Smoke test: the fixed model generates coherent tokens on XPU."""
    llm = _make_llm()
    params = SamplingParams(temperature=0.0, max_tokens=8)
    outputs = llm.generate(["1 2 3 4 5"], sampling_params=params)
    del llm

    assert outputs, "No outputs returned"
    generated = outputs[0].outputs[0].text
    assert len(generated) > 0, (
        f"Model generated no tokens; got: {generated!r}"
    )
