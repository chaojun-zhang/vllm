# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn.parameter import Parameter

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context, is_forward_context_available

if TYPE_CHECKING:
    from vllm.config import ParallelConfig
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, sp_pad))
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
        x = torch.nn.functional.pad(x, pad)
    chunk = x.shape[0] // tp_size
    return x[tp_rank * chunk : (tp_rank + 1) * chunk]


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens

    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-num_tokens) % tp_size
    if sp_pad > 0:
        is_padding = torch.nn.functional.pad(is_padding, (0, sp_pad), value=True)
    chunk = is_padding.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]


def is_sp_active(parallel_config: "ParallelConfig") -> bool:
    """Whether SP should run for the in-flight forward call.

    True only if SP is enabled, there's a live batch to check (a forward
    context with a batch descriptor), and its token count is at least
    ``sequence_parallel_min_tokens``.

    Args:
        parallel_config: Pass the ``vllm_config`` a model's ``__init__``
            receives, cached on ``self`` -- not ``get_current_vllm_config()``,
            which isn't set during a real forward call in the worker process.

    """
    if not parallel_config.use_sequence_parallel:
        return False

    if not is_forward_context_available():
        return False
    ctx = get_forward_context()
    if ctx.batch_descriptor is None:
        return False

    num_tokens = ctx.batch_descriptor.num_tokens
    return num_tokens >= parallel_config.sequence_parallel_min_tokens


def mark_sp_region(
    entry_after: "RowParallelLinear",
    exit_before: "ColumnParallelLinear",
    parallel_config: "ParallelConfig",
) -> None:
    """Mark the region between two linear layers as sequence-parallel.

    Args:
        entry_after: Row-parallel layer whose output enters the region --
            it's reduce-scattered down to a local shard.
        exit_before: Column-parallel layer whose input exits the region --
            it's all-gathered back to full tokens before its own GEMM.
        parallel_config: The parallelism configuration.

    Everything in between (typically a residual add + norm) runs on the
    local shard. Only each layer's ``forward`` is wrapped; no other
    behavior (bias, ``gather_output``, ``return_bias``, ...) changes.

    The wrapper picks one of two things per call, based on whether SP is
    active right now:
    - Fused kernel available: flips its ``set_fuse_gemm_comms`` switch,
      which redirects the quant method's GEMM call to the fused
      comm+GEMM op for that one call.
    - Otherwise: all-gathers/reduce-scatters around the plain
      ``linear_forward``, which always still runs.

    No-op when ``enable_sequence_parallel`` is off.

    """
    if not parallel_config.enable_sequence_parallel:
        return

    def _patch_column_parallel_linear(linear: "ColumnParallelLinear") -> None:
        linear_forward = linear.forward
        fused_kernel = None
        if parallel_config.enable_sequence_parallel_fuse_gemm_comms:
            from vllm.model_executor.kernels.fused_comm import (
                init_all_gather_gemm_kernel,
            )

            fused_kernel = init_all_gather_gemm_kernel(linear)

        def forward(
            x, *args, **kwargs
        ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
            sp_active = is_sp_active(parallel_config)
            if fused_kernel is not None:
                fused_kernel.set_fuse_gemm_comms(sp_active)
            elif sp_active:
                x = sp_all_gather(x)
            return linear_forward(x, *args, **kwargs)

        linear.forward = forward

    def _patch_row_parallel_linear(linear: "RowParallelLinear") -> None:
        linear_forward = linear.forward
        fused_kernel = None
        if parallel_config.enable_sequence_parallel_fuse_gemm_comms:
            from vllm.model_executor.kernels.fused_comm import (
                init_gemm_reduce_scatter_kernel,
            )

            fused_kernel = init_gemm_reduce_scatter_kernel(linear)
        use_fused_kernel = fused_kernel is not None

        def forward(
            x: torch.Tensor,
        ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
            sp_active = is_sp_active(parallel_config)
            if use_fused_kernel:
                fused_kernel.set_fuse_gemm_comms(sp_active)
            if not sp_active:
                return linear_forward(x)

            reduce_results = linear.reduce_results
            linear.reduce_results = False
            try:
                output = linear_forward(x)
            finally:
                linear.reduce_results = reduce_results

            if use_fused_kernel:
                return output
            if linear.return_bias:
                output, bias = output
                return sp_reduce_scatter(output), bias
            return sp_reduce_scatter(output)

        linear.forward = forward

    _patch_column_parallel_linear(exit_before)
    _patch_row_parallel_linear(entry_after)
