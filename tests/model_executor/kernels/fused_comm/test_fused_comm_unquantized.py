# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the unquantized fused-comm kernels
(``fused_comm/unquantized.py``): ``SymmMemAllGatherGemm`` and
``SymmMemGemmReduceScatter``.

Each fused kernel's output is compared against plain
``torch.distributed`` all-gather/reduce-scatter driving the same (bf16)
GEMM, to isolate correctness of the fused gather/scatter+GEMM plumbing.
"""

import pytest
import torch
import torch.multiprocessing as mp

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.fused_comm.unquantized import (
    SymmMemAllGatherGemm,
    SymmMemGemmReduceScatter,
)
from vllm.platforms import current_platform

from .fused_comm_test_utils import (
    FakeLinear,
    all_gather_ref,
    init_worker_distributed,
    reduce_scatter_ref,
    run_workers,
)

WORLD_SIZE = 2
M_PER_RANK = 16
K = 256
N = 256
PORT = 29601


def _ag_worker(local_rank: int, world_size: int, q: mp.Queue) -> None:
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

        layer = FakeLinear(weight=w_shard)
        kernel = SymmMemAllGatherGemm(layer)

        y_fused = kernel.apply(x_shard, bias_shard)

        group = get_tp_group().device_group
        x_gathered = all_gather_ref(x_shard, group)
        y_ref = x_gathered @ w_shard.t() + bias_shard

        torch.testing.assert_close(y_fused, y_ref, rtol=2e-2, atol=2e-2)
        q.put("OK")
    except Exception as e:  # noqa: BLE001
        q.put(f"rank{local_rank}: {type(e).__name__}: {e}")


def _rs_worker(local_rank: int, world_size: int, q: mp.Queue) -> None:
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

        layer = FakeLinear(weight=w_shard)
        kernel = SymmMemGemmReduceScatter(layer)

        y_fused = kernel.apply(x_shard, bias)

        group = get_tp_group().device_group
        y_local = x_shard @ w_shard.t()  # no bias: added once, after reduction
        y_ref = reduce_scatter_ref(y_local, group) + bias

        torch.testing.assert_close(y_fused, y_ref, rtol=2e-2, atol=2e-2)
        q.put("OK")
    except Exception as e:  # noqa: BLE001
        q.put(f"rank{local_rank}: {type(e).__name__}: {e}")


@pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="symmetric memory requires XPU",
)
def test_symm_mem_all_gather_gemm_matches_plain_all_gather_gemm():
    if torch.accelerator.device_count() < WORLD_SIZE:
        pytest.skip("Not enough accelerators to run the test.")
    run_workers(_ag_worker, WORLD_SIZE)


@pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="symmetric memory requires XPU",
)
def test_symm_mem_gemm_reduce_scatter_matches_plain_gemm_reduce_scatter():
    if torch.accelerator.device_count() < WORLD_SIZE:
        pytest.skip("Not enough accelerators to run the test.")
    run_workers(_rs_worker, WORLD_SIZE)
