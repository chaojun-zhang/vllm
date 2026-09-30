# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the fused-comm (eager sequence-parallel AllGather/GEMM
and GEMM/ReduceScatter) kernel correctness tests.

These tests spin up a real (2-rank) distributed process group and compare
each fused kernel's output against the *same* underlying (non-fused) GEMM
kernel driven by plain ``torch.distributed`` collectives -- i.e. "ordinary"
all-gather+gemm / gemm+reduce-scatter -- so a mismatch means the fused
kernel itself (its gather/scatter plumbing or scale handling), not the
GEMM math, is wrong.
"""

from __future__ import annotations

import queue
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max

# Keeps the generator-based `set_current_vllm_config` context manager(s)
# created by `init_worker_distributed` alive for the life of the worker
# process -- otherwise they'd be garbage-collected immediately, which would
# run their `finally` clause and revert the "current" vLLM config.
_held_vllm_config_ctxs: list = []


class FakeLinear(torch.nn.Module):
    """Minimal stand-in for ``ColumnParallelLinear``/``RowParallelLinear``.

    The fused-comm kernels and the (non-fused) FP8 kernels underneath them
    only ever access named attributes on ``layer`` (``layer.weight``,
    ``layer.weight_scale``, ...) -- they don't rely on any
    ``ColumnParallelLinear``/``RowParallelLinear``-specific behavior -- so a
    plain module with the right named ``Parameter``s is sufficient.
    """

    def __init__(self, **tensors: torch.Tensor | None) -> None:
        super().__init__()
        for name, value in tensors.items():
            if isinstance(value, torch.Tensor):
                self.register_parameter(
                    name, torch.nn.Parameter(value, requires_grad=False)
                )
            else:
                setattr(self, name, value)


def quantize_per_tensor_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor FP8 quantization: ``w ~= q * scale``."""
    scale = w.abs().max().float() / FP8_MAX
    scale = scale.clamp(min=1e-12)
    q = (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scale.reshape(1)


def quantize_block_fp8(
    w: torch.Tensor, block_n: int = 128, block_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D block FP8 quantization of a ``[N, K]`` weight.

    Returns the quantized weight (``[N, K]``) and a per-block scale
    (``[N // block_n, K // block_k]``), matching the checkpoint layout
    ``Fp8BlockScaledMMLinearKernel``/``XPUFp8BlockScaledMMKernel`` expect
    (``N`` and ``K`` must already be block-aligned).
    """
    N, K = w.shape
    assert N % block_n == 0 and K % block_k == 0
    n_tiles, k_tiles = N // block_n, K // block_k
    blocks = w.float().view(n_tiles, block_n, k_tiles, block_k).permute(0, 2, 1, 3)
    scale = blocks.abs().amax(dim=(-2, -1)).clamp(min=1e-12) / FP8_MAX
    q_blocks = (blocks / scale[..., None, None]).clamp(-FP8_MAX, FP8_MAX)
    q = q_blocks.permute(0, 2, 1, 3).reshape(N, K).to(FP8_DTYPE)
    return q, scale.contiguous()


def run_workers(worker: Callable[[int, int, mp.Queue], None], world_size: int) -> None:
    """Spawn ``world_size`` copies of ``worker`` and fail the test if any
    reported an error (or request a skip if none are usable)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(worker, args=(world_size, q), nprocs=world_size, join=True)
    results = []
    try:
        while True:
            results.append(q.get(timeout=1))
    except queue.Empty:
        pass
    finally:
        cleanup_dist_env_and_memory()

    assert len(results) == world_size, (
        f"expected {world_size} worker results, got {len(results)}: {results}"
    )
    skips = [r for r in results if isinstance(r, str) and r.startswith("SKIP:")]
    if skips:
        import pytest

        pytest.skip(skips[0][len("SKIP:") :])
    failures = [r for r in results if r != "OK"]
    assert not failures, f"worker failure(s): {failures}"


def init_worker_distributed(
    local_rank: int, world_size: int, port: int
) -> torch.device:
    """Set this process's accelerator device and bring up vLLM's
    distributed/tensor-parallel environment. Returns the device to run on.

    Also activates a (dummy) current ``VllmConfig`` for the remaining
    lifetime of this worker process, since the fused-comm kernels'
    constructors (``get_tp_group()`` et al.) rely on one being set.
    """
    device = torch.device(f"{current_platform.device_type}:{local_rank}")
    torch.accelerator.set_device_index(local_rank)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    ctx = set_current_vllm_config(config)
    ctx.__enter__()
    _held_vllm_config_ctxs.append(ctx)

    init_distributed_environment(backend=current_platform.dist_backend)
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return device


def all_gather_ref(x_shard: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Reference AllGather using plain ``torch.distributed`` collectives."""
    shards = [torch.empty_like(x_shard) for _ in range(group.size())]
    dist.all_gather(shards, x_shard, group=group)
    return torch.cat(shards, dim=0)


def reduce_scatter_ref(y_local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Reference GEMM+ReduceScatter tail: all-reduce (sum) the per-rank
    partial GEMM output, then slice out this rank's token shard -- the
    same result ``dist.reduce_scatter`` would produce, computed with a
    primitive that's simpler to reason about here."""
    y_summed = y_local.clone()
    dist.all_reduce(y_summed, op=dist.ReduceOp.SUM, group=group)
    shard_size = y_summed.shape[0] // group.size()
    rank = group.rank()
    return y_summed[rank * shard_size : (rank + 1) * shard_size]
