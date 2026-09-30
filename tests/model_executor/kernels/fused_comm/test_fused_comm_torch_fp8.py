# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the generic-Torch FP8 fused-comm kernels
(``fused_comm/fp8/torch.py``): per-tensor
(``TorchAllGatherScaledMM``/``TorchScaledMMReduceScatter``)
and block-scaled
(``TorchAllGatherBlockScaledMM``/``TorchBlockScaledMMReduceScatter``).

Each fused kernel's output is compared against the same (non-fused)
``*TorchFP8ScaledMMLinearKernel.apply_weights`` GEMM, driven by plain
``torch.distributed`` all-gather/reduce-scatter, so a mismatch means the
fused kernel's gather/scatter+scale plumbing -- not the GEMM/quantization
math -- is wrong.
"""

import pytest
import torch
import torch.multiprocessing as mp

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.fused_comm.fp8.torch import (
    TorchAllGatherBlockScaledMM,
    TorchAllGatherScaledMM,
    TorchBlockScaledMMReduceScatter,
    TorchScaledMMReduceScatter,
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
    FP8_MAX,
    FakeLinear,
    all_gather_ref,
    init_worker_distributed,
    quantize_block_fp8,
    quantize_per_tensor_fp8,
    reduce_scatter_ref,
    run_workers,
)

WORLD_SIZE = 2
M_PER_RANK = 16
K = 256
N = 256
PORT = 29701


class _FakeFp8QuantMethod:
    """Stand-in for ``Fp8LinearMethod``: only needs the ``fp8_linear``
    attribute the fused kernels read `apply_scaled_mm`/`apply_weights`
    off of."""

    def __init__(self, fp8_linear) -> None:
        self.fp8_linear = fp8_linear


def _ag_worker_per_tensor(local_rank: int, world_size: int, q: mp.Queue) -> None:
    try:
        device = init_worker_distributed(local_rank, world_size, PORT)

        torch.manual_seed(0)
        x_full = torch.randn(
            M_PER_RANK * world_size, K, device=device, dtype=torch.bfloat16
        )
        w_full = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.05
        bias_full = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.1

        n_per_rank = N // world_size
        w_shard = w_full[local_rank * n_per_rank : (local_rank + 1) * n_per_rank]
        bias_shard = bias_full[local_rank * n_per_rank : (local_rank + 1) * n_per_rank]
        x_shard = x_full[local_rank * M_PER_RANK : (local_rank + 1) * M_PER_RANK]

        w_q, w_scale = quantize_per_tensor_fp8(w_shard)
        # Use a *static* activation scale: the fused kernel quantizes each
        # rank's own token shard (not the post-gather tensor), so a dynamic
        # (recomputed-per-call) scale would legitimately differ between the
        # fused and reference paths (each rank sees different token-shard
        # statistics) even with correct gather/scatter plumbing. A static
        # scale, shared by construction, isolates the comparison to the
        # fused kernel's gather/scale-application logic.
        x_scale = x_full.abs().amax().float() / FP8_MAX
        # PerTensorTorchFP8ScaledMMLinearKernel.process_weights_after_loading is
        # a no-op, so `layer.weight` must already be in the `[K, N]` layout
        # `torch._scaled_mm` consumes directly.
        layer = FakeLinear(
            weight=w_q.t().contiguous(), weight_scale=w_scale, input_scale=x_scale
        )
        fp8_kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8StaticTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(n_per_rank, K),
            force_kernel=PerTensorTorchFP8ScaledMMLinearKernel,
        )
        fp8_kernel.process_weights_after_loading(layer)

        kernel = TorchAllGatherScaledMM(_FakeFp8QuantMethod(fp8_kernel))
        kernel.set_fuse_gemm_comms(True)
        y_fused = fp8_kernel.apply_weights(layer, x_shard, bias_shard)

        group = get_tp_group().device_group
        x_gathered = all_gather_ref(x_shard, group)
        kernel.set_fuse_gemm_comms(False)
        y_ref = fp8_kernel.apply_weights(layer, x_gathered, bias_shard)

        torch.testing.assert_close(y_fused, y_ref, rtol=2e-2, atol=2e-2)
        q.put("OK")
    except Exception as e:  # noqa: BLE001
        q.put(f"rank{local_rank}: {type(e).__name__}: {e}")


def _rs_worker_per_tensor(local_rank: int, world_size: int, q: mp.Queue) -> None:
    try:
        device = init_worker_distributed(local_rank, world_size, PORT + 1)

        torch.manual_seed(1)
        M = M_PER_RANK * world_size
        x_full = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w_full = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.05
        bias = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.1

        k_per_rank = K // world_size
        x_shard = x_full[:, local_rank * k_per_rank : (local_rank + 1) * k_per_rank]
        w_shard = w_full[:, local_rank * k_per_rank : (local_rank + 1) * k_per_rank]

        w_q, w_scale = quantize_per_tensor_fp8(w_shard)
        layer = FakeLinear(
            weight=w_q.t().contiguous(),
            weight_scale=w_scale,
            bias=bias,
            skip_bias_add=False,
        )
        fp8_kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(N, k_per_rank),
            force_kernel=PerTensorTorchFP8ScaledMMLinearKernel,
        )
        fp8_kernel.process_weights_after_loading(layer)

        kernel = TorchScaledMMReduceScatter(_FakeFp8QuantMethod(fp8_kernel), layer)
        kernel.set_fuse_gemm_comms(True)
        # Only rank 0 gets a real bias here, mirroring
        # RowParallelLinear.forward's own all-reduce convention -- the
        # fused kernel must ignore this and recompute the real bias from
        # `layer.bias` instead (every rank needs it, see apply_fused).
        bias_in = bias if local_rank == 0 else None
        y_fused = fp8_kernel.apply_weights(layer, x_shard, bias_in)

        group = get_tp_group().device_group
        kernel.set_fuse_gemm_comms(False)
        y_local = fp8_kernel.apply_weights(layer, x_shard, bias=None)
        y_ref = reduce_scatter_ref(y_local, group) + bias

        torch.testing.assert_close(y_fused, y_ref, rtol=2e-2, atol=2e-2)
        q.put("OK")
    except Exception as e:  # noqa: BLE001
        q.put(f"rank{local_rank}: {type(e).__name__}: {e}")


def _ag_worker_block(local_rank: int, world_size: int, q: mp.Queue) -> None:
    try:
        device = init_worker_distributed(local_rank, world_size, PORT + 2)

        torch.manual_seed(2)
        x_full = torch.randn(
            M_PER_RANK * world_size, K, device=device, dtype=torch.bfloat16
        )
        w_full = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.05
        bias_full = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.1

        n_per_rank = N // world_size
        w_shard = w_full[local_rank * n_per_rank : (local_rank + 1) * n_per_rank]
        bias_shard = bias_full[local_rank * n_per_rank : (local_rank + 1) * n_per_rank]
        x_shard = x_full[local_rank * M_PER_RANK : (local_rank + 1) * M_PER_RANK]

        w_q, w_scale = quantize_block_fp8(w_shard)
        layer = FakeLinear(weight=w_q, weight_scale_inv=w_scale)
        fp8_kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8Dynamic128Sym,
            weight_quant_key=kFp8Static128BlockSym,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(n_per_rank, K),
            force_kernel=BlockWiseTorchFP8ScaledMMLinearKernel,
        )
        fp8_kernel.process_weights_after_loading(layer)

        kernel = TorchAllGatherBlockScaledMM(_FakeFp8QuantMethod(fp8_kernel))
        kernel.set_fuse_gemm_comms(True)
        y_fused = fp8_kernel.apply_weights(layer, x_shard, bias_shard)

        group = get_tp_group().device_group
        x_gathered = all_gather_ref(x_shard, group)
        kernel.set_fuse_gemm_comms(False)
        y_ref = fp8_kernel.apply_weights(layer, x_gathered, bias_shard)

        torch.testing.assert_close(y_fused, y_ref, rtol=2e-2, atol=2e-2)
        q.put("OK")
    except Exception as e:  # noqa: BLE001
        q.put(f"rank{local_rank}: {type(e).__name__}: {e}")


def _rs_worker_block(local_rank: int, world_size: int, q: mp.Queue) -> None:
    try:
        device = init_worker_distributed(local_rank, world_size, PORT + 3)

        torch.manual_seed(3)
        M = M_PER_RANK * world_size
        x_full = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w_full = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.05
        bias = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.1

        k_per_rank = K // world_size
        x_shard = x_full[:, local_rank * k_per_rank : (local_rank + 1) * k_per_rank]
        w_shard = w_full[:, local_rank * k_per_rank : (local_rank + 1) * k_per_rank]

        w_q, w_scale = quantize_block_fp8(w_shard)
        layer = FakeLinear(
            weight=w_q, weight_scale_inv=w_scale, bias=bias, skip_bias_add=False
        )
        fp8_kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8Dynamic128Sym,
            weight_quant_key=kFp8Static128BlockSym,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(N, k_per_rank),
            force_kernel=BlockWiseTorchFP8ScaledMMLinearKernel,
        )
        fp8_kernel.process_weights_after_loading(layer)

        kernel = TorchBlockScaledMMReduceScatter(_FakeFp8QuantMethod(fp8_kernel), layer)
        kernel.set_fuse_gemm_comms(True)
        y_fused = fp8_kernel.apply_weights(layer, x_shard, bias)

        group = get_tp_group().device_group
        kernel.set_fuse_gemm_comms(False)
        y_local = fp8_kernel.apply_weights(layer, x_shard, bias=None)
        y_ref = reduce_scatter_ref(y_local, group) + bias

        torch.testing.assert_close(y_fused, y_ref, rtol=2e-2, atol=2e-2)
        q.put("OK")
    except Exception as e:  # noqa: BLE001
        q.put(f"rank{local_rank}: {type(e).__name__}: {e}")


pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="symmetric memory requires CUDA/ROCm/XPU",
)


def _skip_if_not_enough_accelerators():
    if torch.accelerator.device_count() < WORLD_SIZE:
        pytest.skip("Not enough accelerators to run the test.")


def test_torch_symm_mem_all_gather_scaled_mm_per_tensor():
    _skip_if_not_enough_accelerators()
    run_workers(_ag_worker_per_tensor, WORLD_SIZE)


def test_torch_symm_mem_scaled_mm_reduce_scatter_per_tensor():
    _skip_if_not_enough_accelerators()
    run_workers(_rs_worker_per_tensor, WORLD_SIZE)


def test_torch_symm_mem_all_gather_block_scaled_mm():
    _skip_if_not_enough_accelerators()
    run_workers(_ag_worker_block, WORLD_SIZE)


def test_torch_symm_mem_block_scaled_mm_reduce_scatter():
    _skip_if_not_enough_accelerators()
    run_workers(_rs_worker_block, WORLD_SIZE)
