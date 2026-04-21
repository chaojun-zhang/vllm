# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests proving Intel XPU triton compiler bugs with
layer_norm_fwd_kernel.

These tests demonstrate two separate bugs in the Intel XPU triton compiler
that prevent layer_norm_fwd_kernel from being compiled through the
torch.compile / Inductor lowering path. The kernel itself works correctly
when called directly — only the compiler path is broken.

Bug 1 (generate_ttir): Intel triton's code_generator fails during AST→TTIR
  conversion for layer_norm_fwd_kernel with:
    triton.compiler.errors.CompilationError: at 1:0:
    def layer_norm_fwd_kernel(
    ^
    IndexError('Function argument index out of range')
  This causes identify_mutated_tensors (used by Inductor to analyze which
  kernel args are mutated) to fall back to "assume every input is mutated".

Bug 2 (triton.compile / LLVM lowering): Intel backend's
  ConvertTritonIntelGPUToLLVM pass fails with:
    Assertion `inElemTy.isF32() && "unsupported conversion"' failed
  in TruncFOpConversion. The kernel's `eps` parameter is Python float (f64),
  which propagates through rsqrt(var + eps) computation, producing f64
  intermediate results. The final f64→bf16 truncation for output storage is
  not supported by the Intel LLVM backend (only f32→bf16 is supported).

Both bugs are upstream Intel triton compiler issues, NOT vLLM bugs.
"""

import pytest
import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_xpu(),
        reason="Intel XPU-specific triton compiler bug tests",
    ),
]

DEVICE = "xpu"


@triton.jit
def simple_mutation_kernel(
    X,
    OUT,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A simple triton kernel for comparison (control test)."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    x = tl.load(X + idx, mask=mask)
    tl.store(OUT + idx, x * 2.0, mask=mask)


def _build_layer_norm_fwd_kernel_kwargs(
    M: int = 8,
    N: int = 128,
    has_bias: bool = False,
    has_z: bool = True,
    activation: str = "swish",
) -> dict:
    """Build kwargs dict for layer_norm_fwd_kernel matching its full signature.

    These are the same kwargs that Inductor's triton_kernel_wrapper_functional
    HOP would construct when lowering the kernel.
    """
    x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
    out = torch.empty_like(x)
    weight = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)
    bias = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)
    z = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
    mean = torch.empty(M, dtype=torch.float32, device=DEVICE)
    rstd = torch.empty(M, dtype=torch.float32, device=DEVICE)

    return {
        "X": x,
        "Y": out,
        "W": weight,
        "B": bias,
        "Z": z,
        "Mean": mean,
        "Rstd": rstd,
        "stride_x_row": x.stride(0),
        "stride_y_row": out.stride(0),
        "stride_z_row": z.stride(0),
        "M": M,
        "N": N,
        "eps": 1e-5,
        "BLOCK_N": triton.next_power_of_2(N),
        "ROWS_PER_BLOCK": 1,
        "HAS_BIAS": has_bias,
        "HAS_Z": has_z,
        "STORE_MEAN": False,
        "NORM_BEFORE_GATE": False,
        "IS_RMS_NORM": True,
        "ACTIVATION": activation,
    }


class TestIntelTritonLayerNormBug:
    """Demonstrates that Intel XPU triton compiler cannot handle
    layer_norm_fwd_kernel through the torch.compile / Inductor path."""

    def test_generate_ttir_simple_kernel_succeeds(self):
        """Control: generate_ttir works for a simple triton kernel on XPU."""
        from torch._higher_order_ops.triton_kernel_wrap import generate_ttir

        N = 128
        x = torch.randn(N, dtype=torch.float32, device=DEVICE)
        out = torch.empty_like(x)

        kwargs = {"X": x, "OUT": out, "N": N, "BLOCK": 128}

        # Should succeed without error
        ttir_module, tensor_names = generate_ttir(
            simple_mutation_kernel, kwargs, {}
        )
        assert ttir_module is not None
        assert "X" in tensor_names
        assert "OUT" in tensor_names

    def test_generate_ttir_layer_norm_fwd_kernel_fails(self):
        """Bug 1: generate_ttir fails for layer_norm_fwd_kernel on Intel XPU.

        Intel triton's code_generator raises IndexError('Function argument
        index out of range') during AST→TTIR conversion. This is the same
        error that causes identify_mutated_tensors to fall back to 'assuming
        every input is mutated' at runtime.

        Stack trace (from production):
          generate_ttir()
          → ASTSource.make_ir()
          → ast_to_ttir()
          → code_generator.visit()
          → CompilationError wrapping IndexError
        """
        from triton.compiler.errors import CompilationError

        from torch._higher_order_ops.triton_kernel_wrap import generate_ttir

        from vllm.model_executor.layers.fla.ops.layernorm_guard import (
            layer_norm_fwd_kernel,
        )

        kwargs = _build_layer_norm_fwd_kernel_kwargs()

        with pytest.raises(CompilationError, match="index out of range"):
            generate_ttir(layer_norm_fwd_kernel, kwargs, {})

    def test_layer_norm_fwd_kernel_works_directly(self):
        """Control: the kernel produces correct results when called directly.

        This proves the kernel logic is correct — only the Intel triton
        compiler's TTIR generation / Inductor lowering path is broken.
        """
        from vllm.model_executor.layers.fla.ops.layernorm_guard import (
            layer_norm_fwd,
        )

        M, N = 8, 128
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        weight = torch.ones(N, dtype=torch.bfloat16, device=DEVICE)

        out, _, _ = layer_norm_fwd(
            x, weight, bias=None, eps=1e-5, z=None, is_rms_norm=True
        )

        assert torch.isfinite(out).all(), "Direct kernel call should not produce NaN/Inf"

        # Compare with PyTorch reference
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        expected = x_f32 * torch.rsqrt(variance + 1e-5)
        torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)

    def test_layer_norm_fwd_kernel_gated_works_directly(self):
        """Control: gated variant (with z) also produces correct results directly."""
        from vllm.model_executor.layers.fla.ops.layernorm_guard import (
            layer_norm_fwd,
        )

        M, N = 8, 128
        torch.manual_seed(42)
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        z = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        weight = torch.ones(N, dtype=torch.bfloat16, device=DEVICE)

        out, _, _ = layer_norm_fwd(
            x,
            weight,
            bias=None,
            eps=1e-5,
            z=z,
            norm_before_gate=False,
            is_rms_norm=True,
            activation="swish",
        )

        assert torch.isfinite(out).all(), (
            "Direct gated kernel call should not produce NaN/Inf"
        )

    def test_triton_compile_layer_norm_fwd_kernel_fails(self):
        """Bug 2: triton.compile fails for layer_norm_fwd_kernel on Intel XPU.

        Intel backend's ConvertTritonIntelGPUToLLVM pass fails with:
          Assertion `inElemTy.isF32() && "unsupported conversion"' failed
        in TruncFOpConversion.

        The kernel's `eps` parameter is Python float (f64), which propagates
        through rsqrt(var + eps). The resulting f64→bf16 truncation for output
        storage is not supported by the Intel LLVM backend.
        """
        from triton.compiler.compiler import ASTSource, make_backend
        from triton.runtime.jit import JITFunction

        from vllm.model_executor.layers.fla.ops.layernorm_guard import (
            layer_norm_fwd_kernel,
        )

        assert isinstance(layer_norm_fwd_kernel, JITFunction)

        # Build the same args that would be used at runtime
        M, N = 8, 128
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        out = torch.empty_like(x)
        weight = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)
        bias = torch.randn(N, dtype=torch.bfloat16, device=DEVICE)
        z = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        mean = torch.empty(M, dtype=torch.float32, device=DEVICE)
        rstd = torch.empty(M, dtype=torch.float32, device=DEVICE)

        ordered_args = {
            "X": x, "Y": out, "W": weight, "B": bias, "Z": z,
            "Mean": mean, "Rstd": rstd,
            "stride_x_row": N, "stride_y_row": N, "stride_z_row": N,
            "M": M, "N": N, "eps": 1e-5,
            "BLOCK_N": 128, "ROWS_PER_BLOCK": 1,
            "HAS_BIAS": False, "HAS_Z": True, "STORE_MEAN": False,
            "NORM_BEFORE_GATE": False, "IS_RMS_NORM": True,
            "ACTIVATION": "swish",
        }

        constants = {
            name: arg for name, arg in ordered_args.items()
            if not isinstance(arg, torch.Tensor)
        }

        target = triton.runtime.driver.active.get_current_target()
        backend = make_backend(target)
        options = backend.parse_options({"num_warps": 1})

        # Build signature the way generate_ttir does
        from torch._inductor.utils import (
            get_triton_attrs_descriptor_version,
            triton_version_uses_attrs_dict,
        )

        if triton_version_uses_attrs_dict():
            from triton.runtime.jit import mangle_type
            signature = {}
            for i, (name, arg) in enumerate(ordered_args.items()):
                if layer_norm_fwd_kernel.params[i].is_constexpr:
                    signature[name] = "constexpr"
                else:
                    signature[name] = mangle_type(arg)
        else:
            constexprs = [
                p.num for p in layer_norm_fwd_kernel.params if p.is_constexpr
            ]
            signature = {
                name: layer_norm_fwd_kernel._type_of(
                    layer_norm_fwd_kernel.key_of(arg)
                )
                for i, (name, arg) in enumerate(ordered_args.items())
                if i not in constexprs
            }

        specialization = backend.get_attrs_descriptor(
            ordered_args.values(), layer_norm_fwd_kernel.params
        )

        src = ASTSource(
            layer_norm_fwd_kernel, signature, constants, specialization
        )

        # Compilation should fail in Intel backend's LLVM lowering
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            triton.compile(src, options=options)
