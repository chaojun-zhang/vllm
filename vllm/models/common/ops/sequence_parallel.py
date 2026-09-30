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
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ParallelConfig
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

logger = init_logger(__name__)


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



def is_sequence_parallel_active(parallel_config: "ParallelConfig") -> bool:
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


def suspend_sequence_parallel_boundary(
    suspend_before: "ColumnParallelLinear",
    resume_after: "RowParallelLinear",
    parallel_config: "ParallelConfig",
) -> None:
    """Register one SP-suspended region on a column/row-linear boundary
    pair: `suspend_before`'s input boundary gathers the incoming local
    shard back to full tokens before its own column-parallel GEMM;
    `resume_after`'s output boundary scatters the row-parallel GEMM's
    output back down to a local shard.
    """
    if not parallel_config.enable_sequence_parallel:
        return

    def _current_num_tokens() -> int | None:
        if not is_forward_context_available():
            return None
        ctx = get_forward_context()
        if ctx.batch_descriptor is None:
            return None

        return ctx.batch_descriptor.num_tokens

    def _wrap_all_gather(linear: "ColumnParallelLinear") -> None:
        linear_forward = linear.forward
        fused_kernel = None
        if parallel_config.enable_sequence_parallel_fuse_gemm_comms:
            from vllm.model_executor.kernels.fused_comm import (
                init_fused_comm_kernel,
            )
            fused_kernel = init_fused_comm_kernel(linear)

        def forward(x, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
            if not is_sequence_parallel_active(parallel_config):
                # SP off for this call: restore the layer's own gather behavior.
                return linear_forward(x, *args, **kwargs)

            bias = linear.bias if not linear.skip_bias_add else None
            if fused_kernel is not None:
                logger.info_once("SP input boundary: using fused all_gather + gemm")
                output = fused_kernel.apply(x, bias)
            else:
                logger.info_once("SP input boundary: using all_gather + gemm")
                output = linear.quant_method.apply(linear, sp_all_gather(x), bias)

            # Both branches gather (and then GEMM) on the padded, TP-aligned
            # chunk size, which can be a few rows longer than the true token
            # count; trim once here so neither branch has to be trusted to
            # do it on its own.
            full_num_tokens = _current_num_tokens()
            if full_num_tokens is not None:
                output = output[:full_num_tokens]

            if not linear.return_bias:
                return output
            output_bias = linear.bias if linear.skip_bias_add else None
            return output, output_bias

        linear.forward = forward


    def _wrap_reduce_scatter(linear: "RowParallelLinear") -> None:
        linear_forward = linear.forward
        fused_kernel = None
        if parallel_config.enable_sequence_parallel_fuse_gemm_comms:
            from vllm.model_executor.kernels.fused_comm import (
                init_fused_comm_kernel,
            )
            fused_kernel = init_fused_comm_kernel(linear)

        def forward(
                x: torch.Tensor,
        ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
            if not is_sequence_parallel_active(parallel_config):
                return linear_forward(x)

            if linear.input_is_parallel:
                input_parallel = x
            else:
                split_input = split_tensor_along_last_dim(x, num_partitions=linear.tp_size)
                input_parallel = split_input[linear.tp_rank].contiguous()
            bias_ = None if (linear.tp_rank > 0 or linear.skip_bias_add) else linear.bias

            if fused_kernel is not None:
                logger.info_once(
                    "SP output boundary: using fused gemm + reduce_scatter")
                output = fused_kernel.apply(input_parallel, bias_)
            else:
                logger.info_once("SP output boundary: using gemm + reduce_scatter")
                output = linear.quant_method.apply(linear, input_parallel, bias_)
                output = sp_reduce_scatter(output)

            if not linear.return_bias:
                return output
            output_bias = linear.bias if linear.skip_bias_add else None
            return output, output_bias

        linear.forward = forward


    _wrap_all_gather(suspend_before)
    _wrap_reduce_scatter(resume_after)


def suspend_sequence_parallel_module(
    module: torch.nn.Module,
    parallel_config: "ParallelConfig",
) -> None:
    """Register one SP-suspended region on an arbitrary module with no
    linear boundary to hook into: wraps `module.forward` with a plain
    `sp_all_gather` on entry and `sp_reduce_scatter` on exit.
    """
    if not parallel_config.enable_sequence_parallel:
        return

    module_forward = module.forward

    def forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not is_sequence_parallel_active(parallel_config):
            return module_forward(x, *args, **kwargs)

        output = module_forward(sp_all_gather(x), *args, **kwargs)
        return sp_reduce_scatter(output)

    module.forward = forward
