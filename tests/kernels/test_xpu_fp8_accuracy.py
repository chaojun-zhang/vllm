# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Accuracy tests for XPUFP8ScaledMMLinearKernel.

Uses the same QuantFP8 instance that the kernel itself constructs in __init__,
so the test exercises the exact same quantisation path as production inference.

Also contains a direct equivalence test between _xpu_C.fp8_gemm and
torch._scaled_mm (TestFp8GemmVsScaledMM).  This validates the assumption
behind XPUFp8GEMMReduceScatterPattern / AllGatherXPUFp8GEMMPattern, which
replace fp8_gemm with patched_fused_scaled_matmul_reduce_scatter /
fused_all_gather_scaled_matmul — both of which call torch._scaled_mm
internally on XPU.

Run:
  pytest tests/kernels/test_xpu_fp8_accuracy.py -v -s
"""

import types as _types

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.kernels.linear import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUW8A8FP8LinearKernel,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
)

DEVICE = "xpu"
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


# ─────────────────────────────────────────────────────────────────────────────
# Weight quantisation helpers  (fp8_gemm routes by dtype+numel, not shape)
# ─────────────────────────────────────────────────────────────────────────────


def _quant_weight_per_tensor(w: torch.Tensor):
    """Quantise weight [N,K] with a single scale [1]."""
    scale_val = max(w.float().abs().max().item() / FP8_MAX, 1e-6)
    fp8 = (w.float() / scale_val).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return fp8, torch.tensor([scale_val], dtype=torch.float32, device=w.device)


def _quant_weight_per_channel(w: torch.Tensor):
    """Quantise weight [N,K] with per-output-channel scale [N]."""
    ch_max = w.float().abs().amax(dim=-1).clamp(min=1e-6)  # [N]
    scale = ch_max / FP8_MAX
    fp8 = (w.float() / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return fp8, scale.float()


# ─────────────────────────────────────────────────────────────────────────────
# Layer builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_layer_and_kernel(K, N, per_channel_weight, act_static):
    """
    Build a fake quantised layer + XPUFP8ScaledMMLinearKernel that mirrors
    the real production setup in Fp8LinearMethod.process_weights_after_loading.

    Production sequence (fp8.py):
      1. weight loaded from checkpoint as [N, K] C-contiguous fp8
      2. process_fp8_weight_tensor_strategy  — may requantise to max-scale
      3. weight = weight.t()                 — now [K, N] Fortran-order (non-contiguous)
      4. replace_parameter(layer, "weight", weight.data)
      5. kernel.process_weights_after_loading(layer)
            → _ensure_kn_weight_layout: shape==(K,N) but not contiguous
              → .contiguous()  ← makes it aligned C-contiguous [K,N]

    We replicate step 3 here so the test exercises the same code branch inside
    _ensure_kn_weight_layout as production does.
    """
    torch.manual_seed(42)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE) * 0.1

    if per_channel_weight:
        w_fp8, w_scale = _quant_weight_per_channel(w_bf16)
        w_qkey = kFp8StaticChannelSym
    else:
        w_fp8, w_scale = _quant_weight_per_tensor(w_bf16)
        w_qkey = kFp8StaticTensorSym

    a_qkey = kFp8StaticTensorSym if act_static else kFp8DynamicTokenSym

    layer = type("FakeLayer", (), {})()

    # Simulate what fp8.py does before calling kernel.process_weights_after_loading:
    #   weight is stored as [N,K] in the checkpoint, then transposed to [K,N]
    #   Fortran-order via .t() — NOT contiguous.
    w_fp8_fortran = w_fp8.t()  # [K, N], Fortran-order (non-contiguous)
    assert not w_fp8_fortran.is_contiguous()

    layer.weight = nn.Parameter(w_fp8_fortran, requires_grad=False)
    layer.weight_scale = nn.Parameter(w_scale, requires_grad=False)
    layer.input_scale = None
    layer.input_scale_ub = None
    layer.input_size_per_partition = K
    layer.output_size_per_partition = N

    cfg = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=w_qkey,
        activation_quant_key=a_qkey,
        weight_shape=(N, K),
        input_dtype=FP8_DTYPE,
        out_dtype=torch.bfloat16,
    )
    kernel = XPUW8A8FP8LinearKernel(
        cfg,
        layer_param_names=["weight", "weight_scale", "input_scale", "input_scale_ub"],
    )

    # Simulate load-time preprocessing — same as production call.
    kernel.process_weights_after_loading(layer)

    # Sanity: weight must now be [K, N] C-contiguous.
    assert layer.weight.shape == (K, N), (
        f"weight shape after loading: {layer.weight.shape}, expected ({K},{N})"
    )
    assert layer.weight.is_contiguous(), "weight must be C-contiguous after loading"

    return layer, kernel, w_bf16  # w_bf16: [N, K] bf16 reference


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestXPUFP8KernelAccuracy:
    """
    Verify XPUFP8ScaledMMLinearKernel.apply_weights accuracy vs bf16 matmul.

    The kernel internally uses QuantFP8 (constructed by parent __init__) to
    quantise the activation.  We call apply_weights directly — same code path
    as production inference — and compare against exact bf16 ground truth.

    Tolerance: fp8 quantisation noise is typically < 2 %; we allow 5 %.
    """

    @pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
    @pytest.mark.parametrize(
        "per_channel_weight,act_static",
        [
            (False, True),  # per-tensor weight + static per-tensor act
            (False, False),  # per-tensor weight + dynamic per-token act
            (True, True),  # per-channel weight + static per-tensor act
            (True, False),  # per-channel weight + dynamic per-token act
        ],
        ids=["w_pt-a_static", "w_pt-a_dynamic", "w_pc-a_static", "w_pc-a_dynamic"],
    )
    @pytest.mark.parametrize(
        "M,K,N",
        [
            (32, 2048, 5120),  # non-square
            (128, 4096, 4096),  # square K==N (previously buggy)
        ],
    )
    def test_apply_weights_accuracy(
        self, default_vllm_config, M, K, N, per_channel_weight, act_static
    ):
        layer, kernel, w_bf16 = _build_layer_and_kernel(
            K, N, per_channel_weight, act_static
        )

        torch.manual_seed(7)
        x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.1

        # For static activation, derive a per-tensor scale from x and attach it
        # to the layer so QuantFP8 (static=True) uses it.
        if act_static:
            scale_val = max(x_bf16.float().abs().max().item() / FP8_MAX, 1e-6)
            layer.input_scale = nn.Parameter(
                torch.tensor([scale_val], dtype=torch.float32, device=DEVICE),
                requires_grad=False,
            )

        # ── production inference path ──────────────────────────────────────
        # apply_weights internally calls kernel.quant_fp8 (a QuantFP8 instance)
        # to quantise x, then calls apply_scaled_mm → fp8_gemm.
        out = kernel.apply_weights(layer, x_bf16)

        # ── bf16 ground truth ──────────────────────────────────────────────
        # w_bf16 is [N, K];  x_bf16 @ w_bf16.T = [M, N]
        ref = torch.mm(x_bf16.float(), w_bf16.float().t()).to(torch.bfloat16)

        assert out.shape == ref.shape
        assert out.sum().abs().item() > 0, (
            "output is all-zeros — fp8_gemm returned zeros"
        )

        max_diff = (out - ref).abs().max().item()
        ref_max = ref.abs().max().item()
        rel_err = max_diff / (ref_max + 1e-6)

        assert rel_err < 0.05, (
            f"M={M} K={K} N={N} per_channel={per_channel_weight} "
            f"act_static={act_static}: rel_err={rel_err:.4f} "
            f"(max_diff={max_diff:.4f}, ref_max={ref_max:.4f})"
        )
        print(
            f"  [PASS] M={M} K={K} N={N} "
            f"w={'pc' if per_channel_weight else 'pt'} "
            f"a={'static' if act_static else 'dynamic'}: "
            f"rel_err={rel_err:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-kernel comparison: XPUFP8ScaledMMLinearKernel vs TorchFP8 on CPU
# ─────────────────────────────────────────────────────────────────────────────
# pytorch.py has two static-activation kernel classes:
#   PerTensorTorchFP8ScaledMMLinearKernel  — per-tensor act + per-tensor weight
#   ChannelWiseTorchFP8ScaledMMLinearKernel — everything else (per-channel weight
#                                             or per-token act)
#
# Strategy: run the Torch kernel on CPU (torch._scaled_mm supports CPU fp8),
#           run XPUFP8ScaledMMLinearKernel on XPU, compare results.
#           Both kernels receive the same bf16 input; internal quantisation
#           paths (QuantFP8) are exercised end-to-end via apply_weights().
# ─────────────────────────────────────────────────────────────────────────────


def _make_layer(device, w_fp8, w_scale, input_scale):
    """Build a minimal fake layer that apply_weights can read from."""
    layer = _types.SimpleNamespace(
        weight=nn.Parameter(w_fp8, requires_grad=False),
        weight_scale=nn.Parameter(w_scale, requires_grad=False),
        input_scale=None
        if input_scale is None
        else nn.Parameter(input_scale, requires_grad=False),
        input_scale_ub=None,
    )
    return layer


class TestXPUFP8KernelVsTorchCPU:
    """
    Compare XPUFP8ScaledMMLinearKernel (XPU) against
    PerTensorTorchFP8ScaledMMLinearKernel / ChannelWiseTorchFP8ScaledMMLinearKernel
    running on CPU.

    Both kernels receive the same bf16 input; apply_weights() is called on each
    so that the internal QuantFP8 quantisation path is exercised end-to-end.

    Covered cases (all use static activation scale → matches checkpoint models):
      pt_pt    : per-tensor act  × per-tensor weight  → PerTensorTorchFP8
      tok_pt   : per-token act   × per-tensor weight  → ChannelWiseTorchFP8
      tok_pc   : per-token act   × per-channel weight → ChannelWiseTorchFP8
    Note: ChannelWiseTorchFP8 requires per-token (not per-tensor) activation
    because its unfused-DQ path calls torch.narrow(As, 0, 0, num_tokens).
    """

    PARAM_NAMES = ["weight", "weight_scale", "input_scale", "input_scale_ub"]

    @pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
    @pytest.mark.parametrize(
        "act_per_token,weight_per_channel",
        [
            (False, False),  # per-tensor × per-tensor  → PerTensorTorchFP8
            (True, False),  # per-token  × per-tensor  → ChannelWiseTorchFP8
            (True, True),  # per-token  × per-channel → ChannelWiseTorchFP8
        ],
        ids=["pt_pt", "tok_pt", "tok_pc"],
    )
    @pytest.mark.parametrize(
        "M,K,N",
        [
            (32, 2048, 5120),
            (128, 4096, 4096),  # square K==N
        ],
    )
    def test_vs_torch_cpu(
        self,
        default_vllm_config,
        M: int,
        K: int,
        N: int,
        act_per_token: bool,
        weight_per_channel: bool,
    ):
        torch.manual_seed(42)
        out_dtype = torch.bfloat16

        # ── choose quant keys ──────────────────────────────────────────────
        w_qkey = kFp8StaticChannelSym if weight_per_channel else kFp8StaticTensorSym
        # Static per-tensor or static per-token for activation
        a_qkey = kFp8StaticTokenSym if act_per_token else kFp8StaticTensorSym

        # ── build weight on CPU (fp8.py delivers [K,N] Fortran-order) ─────
        w_bf16_cpu = torch.randn(N, K, dtype=torch.bfloat16) * 0.1

        if weight_per_channel:
            w_scale_cpu = w_bf16_cpu.float().abs().amax(dim=1).clamp(min=1e-6) / FP8_MAX
            w_fp8_cpu = (
                (w_bf16_cpu.float() / w_scale_cpu.unsqueeze(1))
                .clamp(-FP8_MAX, FP8_MAX)
                .to(FP8_DTYPE)
            )  # [N,K]
        else:
            w_scale_cpu = (
                w_bf16_cpu.float().abs().max().clamp(min=1e-6).reshape(1) / FP8_MAX
            )
            w_fp8_cpu = (
                (w_bf16_cpu.float() / w_scale_cpu)
                .clamp(-FP8_MAX, FP8_MAX)
                .to(FP8_DTYPE)
            )  # [N,K]

        # Weight layouts used directly below: Fortran-order [K,N] for Torch CPU
        # kernel, C-contiguous [K,N] for XPU kernel.

        # ── build activation scale (static) ───────────────────────────────
        # For static activation, derive scale from a representative bf16 input
        # and attach it to the layer.
        x_ref = torch.randn(M, K, dtype=torch.bfloat16) * 0.1
        if act_per_token:
            act_scale_cpu = (
                x_ref.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / FP8_MAX
            )  # [M,1]
        else:
            act_scale_cpu = (
                x_ref.float().abs().max().clamp(min=1e-6).reshape(1) / FP8_MAX
            )  # [1]

        # ── Torch CPU reference kernel ─────────────────────────────────────
        # ScaledMMLinearKernel.__init__ has `assert self.is_supported()[0]` which
        # fails on XPU. Use object.__new__ to bypass it and set fields manually.
        if not act_per_token and not weight_per_channel:
            torch_cls = PerTensorTorchFP8ScaledMMLinearKernel
        else:
            # ChannelWiseTorchFP8 requires per-token activation scale ([M,1])
            # when weight is per-channel, because its unfused-DQ path does
            # torch.narrow(As, 0, 0, num_tokens).  Ensure act is per-token.
            torch_cls = ChannelWiseTorchFP8ScaledMMLinearKernel

        cfg_cpu = FP8ScaledMMLinearLayerConfig(
            weight_quant_key=w_qkey,
            activation_quant_key=a_qkey,
            weight_shape=(N, K),
            input_dtype=out_dtype,
            out_dtype=out_dtype,
        )
        # Manually initialise: replicate FP8ScaledMMLinearKernel.__init__ fields
        # without the is_supported() assertion.
        torch_kernel = object.__new__(torch_cls)
        act_desc = a_qkey.scale
        torch_kernel.quant_fp8 = QuantFP8(
            static=act_desc.static,
            group_shape=act_desc.group_shape,
            num_token_padding=None,  # no padding needed for reference
        )
        torch_kernel.fp8_dtype = FP8_DTYPE
        torch_kernel.config = cfg_cpu
        torch_kernel.layer_param_names = self.PARAM_NAMES

        # Build CPU layer
        cpu_layer = _make_layer(
            "cpu",
            w_fp8_cpu.t(),  # [K,N] Fortran-order — process_weights is no-op
            w_scale_cpu.cpu(),
            act_scale_cpu.cpu(),
        )
        torch_kernel.process_weights_after_loading(cpu_layer)

        x_cpu = x_ref.clone()
        ref = torch_kernel.apply_weights(cpu_layer, x_cpu)  # [M, N] bfloat16

        # ── XPU kernel ─────────────────────────────────────────────────────
        cfg_xpu = FP8ScaledMMLinearLayerConfig(
            weight_quant_key=w_qkey,
            activation_quant_key=a_qkey,
            weight_shape=(N, K),
            input_dtype=out_dtype,
            out_dtype=out_dtype,
        )
        xpu_kernel = XPUW8A8FP8LinearKernel(cfg_xpu, self.PARAM_NAMES)

        # XPU layer: weight in [K,N] Fortran-order (same as fp8.py delivers)
        xpu_layer = _make_layer(
            DEVICE,
            w_fp8_cpu.t().to(DEVICE),  # [K,N] Fortran-order
            w_scale_cpu.to(DEVICE),
            act_scale_cpu.to(DEVICE),
        )
        xpu_kernel.process_weights_after_loading(xpu_layer)  # → [K,N] C-contiguous

        x_xpu = x_ref.to(DEVICE)
        out = xpu_kernel.apply_weights(xpu_layer, x_xpu)  # [M, N] bfloat16

        # ── compare ───────────────────────────────────────────────────────
        out_cpu = out.cpu().float()
        ref_f = ref.float()

        assert out_cpu.shape == ref_f.shape
        assert out_cpu.abs().sum().item() > 0, "XPU fp8_gemm returned all-zeros"

        max_diff = (out_cpu - ref_f).abs().max().item()
        ref_norm = ref_f.abs().max().item()
        rel_err = max_diff / (ref_norm + 1e-8)

        assert rel_err < 0.05, (
            f"M={M} K={K} N={N} act_per_token={act_per_token} "
            f"weight_per_channel={weight_per_channel} "
            f"torch_cls={torch_cls.__name__}: "
            f"rel_err={rel_err:.4f} "
            f"(max_diff={max_diff:.4f}, ref_norm={ref_norm:.4f})\n"
            f"  XPU[:3,:3]={out_cpu[:3, :3].tolist()}\n"
            f"  CPU[:3,:3]={ref_f[:3, :3].tolist()}"
        )
        print(
            f"  [PASS] M={M} K={K} N={N} "
            f"act={'tok' if act_per_token else 'pt'} "
            f"w={'pc' if weight_per_channel else 'pt'} "
            f"ref={torch_cls.__name__}: rel_err={rel_err:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Direct equivalence: _xpu_C.fp8_gemm  vs  torch._scaled_mm
# ─────────────────────────────────────────────────────────────────────────────
# XPUFp8GEMMReduceScatterPattern and AllGatherXPUFp8GEMMPattern replace
# fp8_gemm with patched_fused_scaled_matmul_reduce_scatter /
# fused_all_gather_scaled_matmul, which both delegate to torch._scaled_mm on
# XPU.  This class verifies the two operations are numerically equivalent for
# every scale granularity that XPUFP8ScaledMMLinearKernel supports.
#
# Layout / scale conversion required when calling _scaled_mm in place of
# fp8_gemm:
#
#   fp8_gemm(A, B, out_dtype, As, Bs, bias)
#     A  : [M, K]  FP8  row-major
#     B  : [K, N]  FP8  C-contiguous (row-major)   ← fp8_gemm requirement
#     As : [1]  or [M, 1]   float32 activation scale
#     Bs : [1]  or [N]      float32 weight scale
#
#   torch._scaled_mm(A, mat2, scale_a, scale_b, ..., out_dtype=...)
#     A      : [M, K]  FP8  row-major
#     mat2   : [K, N]  FP8  col-major (Fortran order, strides (1, K))
#                            ← same data, different stride vs fp8_gemm
#     scale_a: [1, 1]  or [M, 1]  float32
#     scale_b: [1, 1]  or [1, N]  float32  ← [N] must be reshaped to [1, N]
#
# ─────────────────────────────────────────────────────────────────────────────

_FP8_GEMM_AVAILABLE = hasattr(torch.ops._xpu_C, "fp8_gemm")


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU not available")
@pytest.mark.skipif(
    not _FP8_GEMM_AVAILABLE,
    reason="_xpu_C.fp8_gemm not compiled in this build",
)
class TestFp8GemmVsScaledMM:
    """
    Verify that torch._scaled_mm (with col-major B) produces results that are
    bit-for-bit identical to _xpu_C.fp8_gemm (with row-major B) for all scale
    granularities supported by XPUFP8ScaledMMLinearKernel.

    Covered combinations
    --------------------
    As shape  Bs shape   Description
    --------  ---------  -----------
    [1]       [1]        per-tensor × per-tensor
    [M, 1]    [1]        per-token  × per-tensor
    [1]       [N]        per-tensor × per-channel
    [M, 1]    [N]        per-token  × per-channel
    """

    # XPU _scaled_mm requires M, K, N divisible by 16.
    @pytest.mark.parametrize(
        "M,K,N",
        [
            (32, 64, 128),  # non-square
            (64, 64, 64),  # square
            (16, 32, 16),  # minimal
        ],
    )
    @pytest.mark.parametrize(
        "scale_mode",
        [
            "pt_pt",  # per-tensor act,   per-tensor weight
            "tok_pt",  # per-token  act,   per-tensor weight
            "pt_pc",  # per-tensor act,   per-channel weight
            "tok_pc",  # per-token  act,   per-channel weight
        ],
    )
    def test_fp8_gemm_equals_scaled_mm(
        self, M: int, K: int, N: int, scale_mode: str
    ) -> None:
        torch.manual_seed(0)
        dev = "xpu"
        out_dtype = torch.bfloat16

        # ── build FP8 inputs ──────────────────────────────────────────────
        A_fp32 = torch.randn(M, K, dtype=torch.float32, device=dev) * 0.1
        B_fp32 = torch.randn(K, N, dtype=torch.float32, device=dev) * 0.1

        A_fp8 = A_fp32.to(FP8_DTYPE)
        # fp8_gemm requires B as [K, N] C-contiguous (row-major).
        B_row = B_fp32.to(FP8_DTYPE).contiguous()
        # _scaled_mm requires mat2 as [K, N] col-major (Fortran order).
        B_col = B_row.t().contiguous().t()  # shape [K,N], strides (1, K)
        assert B_row.shape == B_col.shape == (K, N)
        assert B_row.is_contiguous()
        assert not B_col.is_contiguous()

        # ── build scales ──────────────────────────────────────────────────
        # fp8_gemm accepts As: [1] or [M,1],  Bs: [1] or [N]
        # _scaled_mm on XPU only supports two configurations:
        #   TensorWise: scale_a=[1,1]  + scale_b=[1,1]  (both per-tensor)
        #   RowWise:    scale_a=[M,1]  + scale_b=[1,N]  (both row-wise)
        # Mixed granularity (per-token + per-tensor, per-tensor + per-channel)
        # is NOT supported by _scaled_mm on XPU.
        per_token = scale_mode.startswith("tok")
        per_channel = scale_mode.endswith("pc")

        if per_token:
            # Per-token: As is [M, 1]; compute row-wise max
            row_max = A_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
            As = (row_max / FP8_MAX).float()  # [M, 1] — both kernels accept this
        else:
            As = torch.full(
                (1,),
                A_fp32.abs().max().item() / FP8_MAX,
                dtype=torch.float32,
                device=dev,
            )

        if per_channel:
            # Per-channel: Bs is [N] for fp8_gemm; reshaped to [1, N] for _scaled_mm
            # Note: B is [K, N] so per-output-channel scale is over dim 1 (N axis).
            col_max = B_fp32.abs().amax(dim=0).clamp(min=1e-6)  # [N]
            Bs_1d = (col_max / FP8_MAX).float()  # [N]  — for fp8_gemm
            Bs_2d = Bs_1d.unsqueeze(0)  # [1, N] — for _scaled_mm
        else:
            scalar = B_fp32.abs().max().item() / FP8_MAX
            Bs_1d = torch.full((1,), scalar, dtype=torch.float32, device=dev)
            Bs_2d = torch.full((1, 1), scalar, dtype=torch.float32, device=dev)

        # scale_a for _scaled_mm must always be 2-D
        As_2d = As if per_token else As.view(1, 1)

        # _scaled_mm rejects mixed granularity (RowWise requires both scale_a=[M,1]
        # AND scale_b=[1,N]; TensorWise requires both to be singletons).
        mixed_granularity = per_token != per_channel
        if mixed_granularity:
            # Verify that fp8_gemm handles these cases, but _scaled_mm rejects them.
            out_fp8 = torch.ops._xpu_C.fp8_gemm(
                A_fp8, B_row, out_dtype, As, Bs_1d, None
            )
            assert out_fp8.shape == (M, N), "fp8_gemm should handle mixed granularity"
            with pytest.raises(RuntimeError, match="Invalid scaling configuration"):
                torch._scaled_mm(
                    A_fp8,
                    B_col,
                    As_2d,
                    Bs_2d,
                    out_dtype=out_dtype,
                    use_fast_accum=False,
                )
            print(
                f"  [OK] M={M} K={K} N={N} scale={scale_mode}: "
                f"fp8_gemm supports mixed granularity; "
                f"_scaled_mm correctly rejects it"
            )
            return

        # ── call fp8_gemm ─────────────────────────────────────────────────
        out_fp8 = torch.ops._xpu_C.fp8_gemm(
            A_fp8, B_row, out_dtype, As, Bs_1d, None
        )  # [M, N]

        # ── call _scaled_mm ───────────────────────────────────────────────
        out_smm = torch._scaled_mm(
            A_fp8,
            B_col,
            As_2d,
            Bs_2d,
            bias=None,
            scale_result=None,
            out_dtype=out_dtype,
            use_fast_accum=False,
        )  # [M, N]

        assert out_fp8.shape == out_smm.shape == (M, N)
        assert out_fp8.dtype == out_smm.dtype == out_dtype

        # ── compare ───────────────────────────────────────────────────────
        # Both operations perform the same scaled FP8 GEMM; results should
        # be bit-for-bit identical (same hardware path on XPU).
        max_diff = (out_fp8.float() - out_smm.float()).abs().max().item()
        ref_norm = out_fp8.float().abs().max().item()

        assert max_diff == 0.0, (
            f"fp8_gemm and _scaled_mm are NOT bit-identical for "
            f"M={M} K={K} N={N} scale_mode={scale_mode}: "
            f"max_diff={max_diff:.6f} (ref_norm={ref_norm:.6f})\n"
            f"  fp8_gemm[:2,:4]={out_fp8.cpu().float()[:2, :4].tolist()}\n"
            f"  _scaled_mm[:2,:4]={out_smm.cpu().float()[:2, :4].tolist()}"
        )
        print(
            f"  [PASS] M={M} K={K} N={N} scale={scale_mode}: "
            f"fp8_gemm == _scaled_mm (max_diff=0.0)"
        )
