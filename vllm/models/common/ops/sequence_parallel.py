# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn.parameter import Parameter

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    split_tensor_along_last_dim,
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
    """Whether sequence parallelism is active for the in-flight forward call.

    ``parallel_config`` must come from a reliable, non-global source (e.g.
    the ``vllm_config`` a model's ``__init__`` receives directly, cached on
    ``self``) -- ``get_current_vllm_config()`` is only guaranteed to be set
    during model construction, not during a real per-request forward call
    in the serving worker process (see ``gpu_worker.py::execute_model``,
    which doesn't wrap itself in ``set_current_vllm_config()``).
    """
    if not parallel_config.use_sequence_parallel:
        return False

    if not is_forward_context_available():
        # No live batch to check against; conservatively do not engage SP.
        return False
    ctx = get_forward_context()
    if ctx.batch_descriptor is None:
        return False

    # Padded cudagraph bucket size, fixed per bucket.
    num_tokens = ctx.batch_descriptor.num_tokens
    return num_tokens >= parallel_config.sequence_parallel_min_tokens


def mark_sp_region(
    entry_after: "RowParallelLinear",
    exit_before: "ColumnParallelLinear",
    parallel_config: "ParallelConfig",
) -> None:
    """Register an SP region on a row/column-linear boundary pair:
    `entry_after`'s output boundary scatters the row-parallel GEMM's
    output down to a local shard (entering SP); `exit_before`'s input
    boundary gathers the incoming local shard back to full tokens before
    its own column-parallel GEMM (exiting SP again). Everything between
    the two -- typically a residual add + norm -- runs sharded.
    """
    if not parallel_config.enable_sequence_parallel:
        return

    def _wrap_all_gather(linear: "ColumnParallelLinear") -> None:
        linear_forward = linear.forward

        def forward(
            x, *args, **kwargs
        ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
            if not is_sp_active(parallel_config):
                # SP off for this call: restore the layer's own gather behavior.
                return linear_forward(x, *args, **kwargs)

            bias = linear.bias if not linear.skip_bias_add else None
            output = linear.quant_method.apply(linear, sp_all_gather(x), bias)

            # Match the un-wrapped ColumnParallelLinear.forward, which
            # all-gathers its output across the output-feature dim when
            # `gather_output` is set.
            if linear.gather_output and linear.tp_size > 1:
                output = tensor_model_parallel_all_gather(output)

            if not linear.return_bias:
                return output
            output_bias = linear.bias if linear.skip_bias_add else None
            return output, output_bias

        linear.forward = forward

    def _wrap_reduce_scatter(linear: "RowParallelLinear") -> None:
        # `linear.reduce_results` (normally True) is intentionally left
        # untouched here, rather than forced to False at construction:
        # whether SP is active is a per-call, runtime decision, so which
        # collective runs (all-reduce vs. reduce-scatter) has to be a
        # per-call decision too. When SP is off below, we fall through to
        # the original `linear_forward`, which still honors
        # `reduce_results` and all-reduces normally. When SP is on, we
        # bypass `linear.forward` entirely (calling `quant_method.apply`
        # directly on the un-reduced partial output), so `reduce_results`
        # plays no role either way -- forcing it to False would silently
        # break the SP-off branch instead.
        linear_forward = linear.forward

        def forward(
            x: torch.Tensor,
        ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
            if not is_sp_active(parallel_config):
                return linear_forward(x)

            if linear.input_is_parallel:
                input_parallel = x
            else:
                split_input = split_tensor_along_last_dim(
                    x, num_partitions=linear.tp_size
                )
                input_parallel = split_input[linear.tp_rank].contiguous()
            bias_ = (
                None if (linear.tp_rank > 0 or linear.skip_bias_add) else linear.bias
            )

            output = linear.quant_method.apply(linear, input_parallel, bias_)
            output = sp_reduce_scatter(output)

            if not linear.return_bias:
                return output
            output_bias = linear.bias if linear.skip_bias_add else None
            return output, output_bias

        linear.forward = forward

    _wrap_all_gather(exit_before)
    _wrap_reduce_scatter(entry_after)
