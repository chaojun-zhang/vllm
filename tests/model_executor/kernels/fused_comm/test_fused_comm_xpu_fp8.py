# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SP correctness of the XPU FP8 comm+GEMM kernels (``fused_comm/fp8/xpu.py``
and ``fused_comm/fp8/unfused.py``) against the plain TP linear."""

import functools

import pytest
import torch

import vllm._xpu_ops  # noqa: F401  registers torch.ops.vllm.xpu_mxfp8_quantize
from vllm.model_executor.kernels.fused_comm.fp8.unfused import (
    UnfusedBlockScaledMMAllGather,
)
from vllm.model_executor.kernels.fused_comm.fp8.xpu import (
    XPUAllGatherBlockScaledMM,
    XPUAllGatherMXFP8,
    XPUAllGatherScaledMM,
    XPUBlockScaledMMReduceScatter,
    XPUMXFP8ReduceScatter,
    XPUScaledMMReduceScatter,
)
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mxfp8.xpu import XPUMxFp8LinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUFp8BlockScaledMMKernel,
    XPUW8A8FP8LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    xpu_mxfp8_quantize as quant_mxfp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

from .fused_comm_test_utils import (
    STATIC_INPUT_SCALE,
    FakeQuantMethod,
    quantize_block_fp8,
    quantize_per_channel_fp8,
    quantize_per_tensor_fp8,
    run_sp_matrix,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="these kernels require XPU"
)

_ACT_KEYS = {
    "static_tensor": kFp8StaticTensorSym,
    "dynamic_tensor": kFp8DynamicTensorSym,
    "dynamic_token": kFp8DynamicTokenSym,
}
_WEIGHT_KEYS = {"tensor": kFp8StaticTensorSym, "channel": kFp8StaticChannelSym}


def _build_w8a8(w: torch.Tensor, act: str, weight: str):
    quantize = (
        quantize_per_tensor_fp8 if weight == "tensor" else quantize_per_channel_fp8
    )
    w_q, w_scale = quantize(w)
    tensors = {"weight": w_q, "weight_scale": w_scale}
    if act == "static_tensor":
        tensors["input_scale"] = torch.tensor([STATIC_INPUT_SCALE], device=w.device)
    fp8_linear = init_fp8_linear_kernel(
        activation_quant_key=_ACT_KEYS[act],
        weight_quant_key=_WEIGHT_KEYS[weight],
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=tuple(w.shape),
        force_kernel=XPUW8A8FP8LinearKernel,
    )
    return FakeQuantMethod(fp8_linear=fp8_linear), tensors


def _build_block(w: torch.Tensor):
    w_q, w_scale = quantize_block_fp8(w)
    fp8_linear = init_fp8_linear_kernel(
        activation_quant_key=kFp8Dynamic128Sym,
        weight_quant_key=kFp8Static128BlockSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=tuple(w.shape),
        force_kernel=XPUFp8BlockScaledMMKernel,
    )
    return FakeQuantMethod(fp8_linear=fp8_linear), {
        "weight": w_q,
        "weight_scale_inv": w_scale,
    }


def _build_mxfp8(w: torch.Tensor):
    w_q, w_scale = quant_mxfp8(w)
    kernel = XPUMxFp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=tuple(w.shape)))
    return FakeQuantMethod(kernel=kernel), {"weight": w_q, "weight_scale": w_scale}


_W8A8_CONFIGS = [
    ("static_tensor", "tensor"),
    ("dynamic_tensor", "tensor"),
    ("dynamic_token", "channel"),
    ("dynamic_tensor", "channel"),
    ("dynamic_token", "tensor"),
    ("static_tensor", "channel"),
]


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("act,weight", _W8A8_CONFIGS)
@pytest.mark.parametrize(
    "kernel_cls,row_parallel",
    [(XPUAllGatherScaledMM, False), (XPUScaledMMReduceScatter, True)],
    ids=["ag", "rs"],
)
def test_xpu_w8a8(kernel_cls, row_parallel, act, weight, world_size):
    build = functools.partial(_build_w8a8, act=act, weight=weight)
    run_sp_matrix(build, kernel_cls, row_parallel, world_size)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize(
    "build,kernel_cls,row_parallel",
    [
        (_build_block, XPUAllGatherBlockScaledMM, False),
        (_build_block, XPUBlockScaledMMReduceScatter, True),
        (_build_block, UnfusedBlockScaledMMAllGather, False),
        (_build_mxfp8, XPUAllGatherMXFP8, False),
        (_build_mxfp8, XPUMXFP8ReduceScatter, True),
    ],
    ids=["block-ag", "block-rs", "block-unfused-ag", "mxfp8-ag", "mxfp8-rs"],
)
def test_xpu_block_and_mxfp8(build, kernel_cls, row_parallel, world_size):
    run_sp_matrix(build, kernel_cls, row_parallel, world_size)
