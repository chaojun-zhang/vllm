# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SP correctness of the generic-Torch FP8 comm+GEMM kernels
(``fused_comm/fp8/pytorch.py`` and ``fused_comm/fp8/unfused.py``) against the
plain TP linear."""

import functools

import pytest
import torch

from vllm.model_executor.kernels.fused_comm.fp8.pytorch import (
    TorchAllGatherBlockScaledMM,
    TorchAllGatherScaledMM,
    TorchBlockScaledMMReduceScatter,
    TorchScaledMMReduceScatter,
)
from vllm.model_executor.kernels.fused_comm.fp8.unfused import (
    UnfusedBlockScaledMMAllGather,
)
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    BlockWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

from .fused_comm_test_utils import (
    STATIC_INPUT_SCALE,
    FakeQuantMethod,
    quantize_block_fp8,
    quantize_per_tensor_fp8,
    run_sp_matrix,
)

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="symmetric memory requires CUDA/ROCm/XPU",
)


def _build_per_tensor(w: torch.Tensor, act: str):
    w_q, w_scale = quantize_per_tensor_fp8(w)
    # Fp8LinearMethod stores the weight as a [K, N] transposed view.
    tensors = {"weight": w_q.t(), "weight_scale": w_scale}
    if act == "static_tensor":
        tensors["input_scale"] = torch.tensor([STATIC_INPUT_SCALE], device=w.device)
    fp8_linear = init_fp8_linear_kernel(
        activation_quant_key=(
            kFp8StaticTensorSym if act == "static_tensor" else kFp8DynamicTensorSym
        ),
        weight_quant_key=kFp8StaticTensorSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=tuple(w.shape),
        force_kernel=PerTensorTorchFP8ScaledMMLinearKernel,
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
        force_kernel=BlockWiseTorchFP8ScaledMMLinearKernel,
    )
    return FakeQuantMethod(fp8_linear=fp8_linear), {
        "weight": w_q,
        "weight_scale_inv": w_scale,
    }


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("act", ["static_tensor", "dynamic_tensor"])
@pytest.mark.parametrize(
    "kernel_cls,row_parallel",
    [(TorchAllGatherScaledMM, False), (TorchScaledMMReduceScatter, True)],
    ids=["ag", "rs"],
)
def test_torch_per_tensor(kernel_cls, row_parallel, act, world_size):
    build = functools.partial(_build_per_tensor, act=act)
    run_sp_matrix(build, kernel_cls, row_parallel, world_size)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize(
    "kernel_cls,row_parallel",
    [
        (TorchAllGatherBlockScaledMM, False),
        (TorchBlockScaledMMReduceScatter, True),
        (UnfusedBlockScaledMMAllGather, False),
    ],
    ids=["ag", "rs", "unfused-ag"],
)
def test_torch_block(kernel_cls, row_parallel, world_size):
    run_sp_matrix(_build_block, kernel_cls, row_parallel, world_size)
