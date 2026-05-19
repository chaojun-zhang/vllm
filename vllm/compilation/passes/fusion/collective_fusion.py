# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import suppress

import torch
import torch._inductor.pattern_matcher as pm
import torch.distributed.distributed_c10d as c10d
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmInductorPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
)

FP8_DTYPE = current_platform.fp8_dtype()
_XPU_BYTE_ALL_GATHER_MAX_NUMEL = 512 * 1024

logger = init_logger(__name__)


def _flashinfer_scaled_mm_out(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale_result: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> None:
    # Import lazily to avoid a circular import during module initialization
    # when docs or other tooling import the pass without FlashInfer.
    from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm_out

    assert bias is None, "FlashInfer symm_mem adapter does not support bias"
    assert scale_result is None, (
        "FlashInfer symm_mem adapter does not support result scaling"
    )
    assert not use_fast_accum, (
        "FlashInfer symm_mem adapter does not support use_fast_accum"
    )
    assert A.ndim == 2 and B.ndim == 2 and out.ndim == 2, (
        "FlashInfer symm_mem adapter expects 2D inputs and output"
    )
    assert scale_a.numel() == 1 and scale_b.numel() == 1, (
        "FlashInfer symm_mem adapter only supports tensor-wise FP8 scales"
    )

    flashinfer_scaled_fp8_mm_out(
        A,
        B,
        scale_a,
        scale_b,
        out=out,
        out_dtype=out_dtype or out.dtype,
    )


def _flashinfer_fp4_mm_out(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    use_8x4_sf_layout: bool = False,
    backend: str = "cutlass",
) -> None:
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm_out

    assert A.ndim == 2 and B.ndim == 2 and out.ndim == 2, (
        "FlashInfer FP4 symm_mem adapter expects 2D inputs and output"
    )
    flashinfer_scaled_fp4_mm_out(
        A,
        B,
        scale_a,
        scale_b,
        alpha,
        out=out,
        out_dtype=out_dtype or out.dtype,
        use_8x4_sf_layout=use_8x4_sf_layout,
        backend=backend,
    )


def fused_flashinfer_scaled_matmul_reduce_scatter_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    result_shape = list(output_shape)
    result_shape[orig_scatter_dim] //= world_size
    return torch.empty(
        result_shape,
        dtype=out_dtype or torch.bfloat16,
        device=A.device,
    )


def fused_flashinfer_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert orig_scatter_dim == 0 and scatter_dim_after_maybe_reshape == 0, (
        "FlashInfer symm_mem adapter currently only supports scatter_dim=0"
    )
    world_size = c10d._resolve_process_group(group_name).size()
    assert A.ndim == 2 and B.ndim == 2, "FlashInfer symm_mem adapter expects 2D inputs"
    assert A.is_contiguous(), "FlashInfer symm_mem adapter expects contiguous A"
    assert A_scale.numel() == 1 and B_scale.numel() == 1, (
        "FlashInfer symm_mem adapter only supports tensor-wise FP8 scales"
    )
    assert A.shape[0] % world_size == 0, (
        "FlashInfer symm_mem adapter expects M divisible by world size"
    )

    kwargs = {
        "scale_b": B_scale,
        "bias": None,
        "scale_result": None,
        "out_dtype": out_dtype,
        "use_fast_accum": False,
    }
    return torch.distributed._symmetric_memory._fused_scaled_matmul_reduce_scatter_impl(
        mm_out_op=_flashinfer_scaled_mm_out,
        A=A,
        B=B,
        A_scale=A_scale,
        kwargs=kwargs,
        out_dtype=out_dtype,
        reduce_op=reduce_op,
        orig_scatter_dim=orig_scatter_dim,
        scatter_dim_after_maybe_reshape=scatter_dim_after_maybe_reshape,
        group_name=group_name,
        output_shape=output_shape,
    )


def fused_all_gather_flashinfer_scaled_matmul_fake(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    output_shape = list(A_shard.shape)
    output_shape[gather_dim] *= world_size
    output_shape[-1] = B.shape[1]
    return torch.empty(
        output_shape,
        dtype=out_dtype or torch.bfloat16,
        device=A_shard.device,
    )


def fused_all_gather_flashinfer_scaled_matmul(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert gather_dim == 0, (
        "FlashInfer symm_mem adapter currently only supports gather_dim=0"
    )
    _, outputs = torch.distributed._symmetric_memory._fused_all_gather_matmul_impl(
        mm_out_op=_flashinfer_scaled_mm_out,
        A_shard=A_shard,
        Bs=[B],
        A_scale=A_scale,
        kwargs_list=[
            {
                "scale_b": B_scale,
                "bias": None,
                "scale_result": None,
                "out_dtype": out_dtype,
                "use_fast_accum": False,
            }
        ],
        out_dtypes=[out_dtype],
        gather_dim=gather_dim,
        group_name=group_name,
        return_A=False,
    )
    return outputs[0]


def fused_all_gather_flashinfer_fp4_matmul_fake(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale_shard: torch.Tensor,
    B_scale: torch.Tensor,
    alpha: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
    view_a_scale_as_fp8: bool = False,
    use_8x4_sf_layout: bool = False,
    backend: str = "cutlass",
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    output_shape = list(A_shard.shape)
    output_shape[gather_dim] *= world_size
    output_shape[-1] = B.shape[1]
    return torch.empty(
        output_shape,
        dtype=out_dtype or torch.bfloat16,
        device=A_shard.device,
    )


def fused_all_gather_flashinfer_fp4_matmul(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    A_scale_shard: torch.Tensor,
    B_scale: torch.Tensor,
    alpha: torch.Tensor,
    gather_dim: int,
    group_name: str,
    out_dtype: torch.dtype | None = None,
    view_a_scale_as_fp8: bool = False,
    use_8x4_sf_layout: bool = False,
    backend: str = "cutlass",
) -> torch.Tensor:
    assert gather_dim == 0, (
        "FlashInfer FP4 symm_mem adapter currently only supports gather_dim=0"
    )
    assert A_shard.ndim == 2 and A_scale_shard.ndim == 2 and B.ndim == 2, (
        "FlashInfer FP4 symm_mem adapter expects 2D inputs"
    )
    if view_a_scale_as_fp8:
        A_scale_shard = A_scale_shard.view(torch.float8_e4m3fn)

    group = c10d._resolve_process_group(group_name)
    world_size = group.size()
    output = A_shard.new_empty(
        A_shard.shape[0] * world_size,
        B.shape[1],
        dtype=out_dtype or torch.bfloat16,
    )
    output_shards = output.chunk(world_size)

    A = A_shard.new_empty(A_shard.shape[0] * world_size, A_shard.shape[1])
    A_scale = A_scale_shard.new_empty(
        A_scale_shard.shape[0] * world_size,
        A_scale_shard.shape[1],
    )

    def fp4_shard_consumer(shards: list[torch.Tensor], rank: int) -> None:
        _flashinfer_fp4_mm_out(
            shards[0],
            B,
            scale_a=shards[1],
            scale_b=B_scale,
            alpha=alpha,
            out=output_shards[rank],
            out_dtype=out_dtype,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
        )

    torch.distributed._symmetric_memory._pipelined_multi_all_gather_and_consume(
        [A_shard, A_scale_shard],
        fp4_shard_consumer,
        [A, A_scale],
        group_name,
        False,
    )
    return output


direct_register_custom_op(
    op_name="fused_flashinfer_scaled_matmul_reduce_scatter",
    op_func=fused_flashinfer_scaled_matmul_reduce_scatter,
    fake_impl=fused_flashinfer_scaled_matmul_reduce_scatter_fake,
)

direct_register_custom_op(
    op_name="fused_all_gather_flashinfer_scaled_matmul",
    op_func=fused_all_gather_flashinfer_scaled_matmul,
    fake_impl=fused_all_gather_flashinfer_scaled_matmul_fake,
)

direct_register_custom_op(
    op_name="fused_all_gather_flashinfer_fp4_matmul",
    op_func=fused_all_gather_flashinfer_fp4_matmul,
    fake_impl=fused_all_gather_flashinfer_fp4_matmul_fake,
)


def _xpu_fp8_mm_out(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
    scale_result: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> None:
    assert scale_result is None, "XPU fp8_gemm does not support result scaling"
    assert not use_fast_accum, "XPU fp8_gemm does not support use_fast_accum"
    if hasattr(torch.ops._xpu_C, "fp8_gemm_out"):
        torch.ops._xpu_C.fp8_gemm_out(out, A, B, scale_a, scale_b, bias)
        return
    result = torch.ops._xpu_C.fp8_gemm(
        A, B, out_dtype or out.dtype, scale_a, scale_b, bias
    )
    out.copy_(result)


def fused_xpu_fp8_matmul_reduce_scatter_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    group_name: str,
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    return torch.empty(
        [x.shape[0] // world_size, weight.shape[1]],
        dtype=out_dtype,
        device=x.device,
    )


def fused_xpu_fp8_matmul_reduce_scatter(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    group_name: str,
) -> torch.Tensor:
    group = c10d._resolve_process_group(group_name)
    world_size = group.size()
    mm = torch.ops._xpu_C.fp8_gemm(
        x, weight, out_dtype, input_scale, weight_scale, None
    )
    output = torch.empty(
        [mm.shape[0] // world_size, mm.shape[1]], dtype=mm.dtype, device=mm.device
    )
    torch.distributed.reduce_scatter_tensor(
        output, mm.contiguous(), op=torch.distributed.ReduceOp.SUM, group=group
    )
    return output


def _xpu_all_gather_dim0_chunked(
    output: torch.Tensor,
    input_: torch.Tensor,
    group: torch.distributed.ProcessGroup,
) -> None:
    if (
        input_.numel() <= _XPU_BYTE_ALL_GATHER_MAX_NUMEL
        or input_.element_size() != 1
    ):
        torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)
        return

    local_rows = input_.shape[0]
    if local_rows == 0:
        return

    world_size = group.size()
    row_numel = input_.numel() // local_rows
    rows_per_chunk = max(1, _XPU_BYTE_ALL_GATHER_MAX_NUMEL // row_numel)
    for start in range(0, local_rows, rows_per_chunk):
        chunk = input_[start : start + rows_per_chunk].contiguous()
        chunk_rows = chunk.shape[0]
        gathered = torch.empty(
            [chunk_rows * world_size] + list(input_.shape[1:]),
            dtype=input_.dtype,
            device=input_.device,
        )
        torch.distributed.all_gather_into_tensor(gathered, chunk, group=group)
        for rank in range(world_size):
            output.narrow(0, rank * local_rows + start, chunk_rows).copy_(
                gathered.narrow(0, rank * chunk_rows, chunk_rows)
            )


def fused_all_gather_xpu_fp8_matmul_fake(
    fp8_shard: torch.Tensor,
    scale_shard: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    group_name: str,
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    return torch.empty(
        [fp8_shard.shape[0] * world_size, weight.shape[1]],
        dtype=out_dtype,
        device=fp8_shard.device,
    )


def fused_all_gather_xpu_fp8_matmul(
    fp8_shard: torch.Tensor,
    scale_shard: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    group_name: str,
) -> torch.Tensor:
    group = c10d._resolve_process_group(group_name)
    world_size = group.size()

    scale_full = torch.empty(
        [scale_shard.shape[0] * world_size] + list(scale_shard.shape[1:]),
        dtype=scale_shard.dtype,
        device=scale_shard.device,
    )
    torch.distributed.all_gather_into_tensor(
        scale_full, scale_shard.contiguous(), group=group
    )

    fp8_full = torch.empty(
        [fp8_shard.shape[0] * world_size] + list(fp8_shard.shape[1:]),
        dtype=fp8_shard.dtype,
        device=fp8_shard.device,
    )
    _xpu_all_gather_dim0_chunked(fp8_full, fp8_shard, group)

    # scale_full from all_gather of per-token scales has shape [M, 1].
    # fp8_gemm handles [M, 1] natively (per-token branch, group={1, K}).
    # No expansion needed.
    return torch.ops._xpu_C.fp8_gemm(
        fp8_full, weight, out_dtype, scale_full, weight_scale, None
    )


direct_register_custom_op(
    op_name="fused_xpu_fp8_matmul_reduce_scatter",
    op_func=fused_xpu_fp8_matmul_reduce_scatter,
    fake_impl=fused_xpu_fp8_matmul_reduce_scatter_fake,
)

direct_register_custom_op(
    op_name="fused_all_gather_xpu_fp8_matmul",
    op_func=fused_all_gather_xpu_fp8_matmul,
    fake_impl=fused_all_gather_xpu_fp8_matmul_fake,
)



def _xpu_mxfp8_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    A = A.contiguous()
    scale_a = scale_a.contiguous()
    scale_b = scale_b.contiguous()
    # Do NOT call B.contiguous() — weight is stored as layer.weight.t() with
    # strides (1, K) (column-major).  oneDNN detects this via
    #   is_nt = B.strides()[B.dim()-2] == 1
    # and uses trans_type_t::nt for the correct MXFP8 scale application.
    # Calling B.contiguous() would change strides to (N, 1), flipping is_nt to
    # False and causing completely wrong scale mapping (0% accuracy).
    # fp8_gemm_out cannot handle column-major B (crashes with "could not
    # construct a memory descriptor using strides"), so fall back to fp8_gemm
    # which handles non-contiguous B correctly.
    if hasattr(torch.ops._xpu_C, "fp8_gemm_out") and B.is_contiguous():
        out = torch.empty(
            [*A.shape[:-1], B.shape[1]], dtype=out_dtype, device=A.device
        )
        torch.ops._xpu_C.fp8_gemm_out(out, A, B, scale_a, scale_b, bias)
        return out
    return torch.ops._xpu_C.fp8_gemm(A, B, out_dtype, scale_a, scale_b, bias)


def fused_xpu_mxfp8_matmul_reduce_scatter_fake(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    group_name: str,
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    return torch.empty(
        [x_fp8.shape[0] // world_size, weight.shape[1]],
        dtype=out_dtype,
        device=x_fp8.device,
    )


_fused_rs_debug_count = 0


def fused_xpu_mxfp8_matmul_reduce_scatter(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    group_name: str,
) -> torch.Tensor:
    """XPU MXFP8: explicit fp8_gemm + reduce_scatter.

    The oneDNN XPU MXFP8 kernel silently returns zeros when M is too large
    (e.g. M=8192, K=2048, N=5120).  The sp-mxfp8 non-fused baseline works
    because each rank only sees M=2048 (already TP-sharded).

    To match that behaviour we split x_fp8 and x_scale along the M dimension
    into world_size chunks of [M/world_size, K], run fp8_gemm on each chunk
    independently, sum the partial results, then reduce_scatter.  This is
    mathematically equivalent to the full-M GEMM followed by reduce_scatter
    because GEMM is linear: sum_i(A_i @ W) == (vstack(A_i)) @ W.
    """
    global _fused_rs_debug_count
    group = c10d._resolve_process_group(group_name)
    world_size = group.size()

    M = x_fp8.shape[0]
    local_m = M // world_size

    # weight is C-contiguous and aligned (guaranteed by
    # XPUMxFp8LinearKernel.process_weights_after_loading).
    # No contiguous() copy needed here.

    if _fused_rs_debug_count < 2:
        _fused_rs_debug_count += 1
        logger.warning(
            "[DBG RS#%d PRE] M=%d local_m=%d K=%d N=%d "
            "x_fp8.ptr=0x%x x_scale.ptr=0x%x "
            "weight.ptr=0x%x weight_scale.ptr=0x%x weight.is_contig=%s",
            _fused_rs_debug_count, M, local_m, x_fp8.shape[1], weight.shape[1],
            x_fp8.data_ptr(), x_scale.data_ptr(),
            weight.data_ptr(), weight_scale.data_ptr(), weight.is_contiguous(),
        )

    # Split M into world_size chunks, run fp8_gemm on each [local_m, K] chunk.
    # This works around an oneDNN XPU MXFP8 bug that silently returns zeros
    # for M=8192 (K=2048) while M=2048 works correctly.
    # Row slices of a C-contiguous aligned tensor are views — no copy needed.
    mm_chunks = []
    for i in range(world_size):
        x_chunk = x_fp8[i * local_m : (i + 1) * local_m]
        s_chunk = x_scale[i * local_m : (i + 1) * local_m]
        mm_chunks.append(
            torch.ops._xpu_C.fp8_gemm(
                x_chunk, weight, out_dtype, s_chunk, weight_scale, None
            )
        )
    mm = torch.cat(mm_chunks, dim=0)  # [M, N]

    output = torch.empty(
        [local_m, mm.shape[1]], dtype=mm.dtype, device=mm.device
    )
    torch.distributed.reduce_scatter_tensor(
        output, mm.contiguous(), op=torch.distributed.ReduceOp.SUM, group=group
    )
    return output


def fused_all_gather_xpu_mxfp8_matmul_fake(
    fp8_shard: torch.Tensor,
    scale_shard: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    group_name: str,
) -> torch.Tensor:
    world_size = c10d._resolve_process_group(group_name).size()
    return torch.empty(
        [fp8_shard.shape[0] * world_size, weight.shape[1]],
        dtype=out_dtype,
        device=fp8_shard.device,
    )


_fused_ag_debug_count = 0


def fused_all_gather_xpu_mxfp8_matmul(
    fp8_shard: torch.Tensor,
    scale_shard: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    group_name: str,
) -> torch.Tensor:
    """XPU MXFP8: all_gather(fp8, scale) + fp8_gemm.

    oneCCL SYCL GPU only supports float16/bfloat16/float32 for allgather;
    uint8 and int8 are not supported.  Pack 8-bit tensors into bfloat16
    (2 bytes = 2 fp8 values) before gathering and unpack afterwards.
    """
    global _fused_ag_debug_count
    group = c10d._resolve_process_group(group_name)
    world_size = group.size()

    # Allocate output tensors with the correct fp8 dtypes from the start.
    # Gather via bfloat16 packing (oneCCL SYCL GPU does not support fp8 allgather):
    #   fp8(1B) → view as uint8(1B) → view as bfloat16(2B, halves last dim)
    # Writing into the bfloat16 view of a pre-allocated fp8 output avoids any
    # view-back dtype conversion that may not preserve the ATen scalar type on XPU.

    # --- scale gather ---
    scale_full = torch.empty(
        [scale_shard.shape[0] * world_size, scale_shard.shape[1]],
        dtype=scale_shard.dtype,
        device=scale_shard.device,
    )
    scale_shard_bf16 = scale_shard.contiguous().view(torch.uint8).view(torch.bfloat16)
    scale_full_bf16 = scale_full.view(torch.uint8).view(torch.bfloat16)
    torch.distributed.all_gather_into_tensor(scale_full_bf16, scale_shard_bf16, group=group)

    # --- fp8 activation gather ---
    fp8_full = torch.empty(
        [fp8_shard.shape[0] * world_size, fp8_shard.shape[1]],
        dtype=fp8_shard.dtype,
        device=fp8_shard.device,
    )
    fp8_shard_bf16 = fp8_shard.contiguous().view(torch.uint8).view(torch.bfloat16)
    fp8_full_bf16 = fp8_full.view(torch.uint8).view(torch.bfloat16)
    torch.distributed.all_gather_into_tensor(fp8_full_bf16, fp8_shard_bf16, group=group)

    out = _xpu_mxfp8_gemm(fp8_full, weight, out_dtype, scale_full, weight_scale, None)
    if _fused_ag_debug_count < 2:
        _fused_ag_debug_count += 1
        logger.warning(
            "[DBG AG#%d] fp8_shard=%s/%s scale_shard=%s/%s fp8_full=%s/%s scale_full=%s/%s out_sum=%s",
            _fused_ag_debug_count,
            fp8_shard.shape, fp8_shard.dtype,
            scale_shard.shape, scale_shard.dtype,
            fp8_full.shape, fp8_full.dtype,
            scale_full.shape, scale_full.dtype,
            out.float().sum().item(),
        )
    return out


direct_register_custom_op(
    op_name="fused_xpu_mxfp8_matmul_reduce_scatter",
    op_func=fused_xpu_mxfp8_matmul_reduce_scatter,
    fake_impl=fused_xpu_mxfp8_matmul_reduce_scatter_fake,
)

direct_register_custom_op(
    op_name="fused_all_gather_xpu_mxfp8_matmul",
    op_func=fused_all_gather_xpu_mxfp8_matmul,
    fake_impl=fused_all_gather_xpu_mxfp8_matmul_fake,
)



class BasePattern:
    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()


class GEMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        mul = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mm_weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [mul, mm_weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul,
                mm_weight,
                "sum",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

class AllGatherGEMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

            return torch.ops.aten.mm.default(all_gather, weight)

        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                x,
                [weight],
                gather_dim=0,
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)
        return [input, mm_weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            scaled_mm = torch.ops.aten._scaled_mm.default(
                input,
                mat2=mat2,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                scaled_mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        return [x, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            return torch.ops.aten._scaled_mm.default(
                all_gather,
                mat2=weight,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class CutlassScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        cutlass_mm_output = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        return [input, mm_weight, scale_a, scale_b, cutlass_mm_output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=cutlass_mm_output,
                a=input,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )

            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                cutlass_scaled_mm[1],
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherCutlassScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        s2 = weight.shape[1]
        output = torch.empty([s1, s2], device=self.device, dtype=self.dtype)

        return [x, weight, scale_a, scale_b, output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=output,
                a=all_gather,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )
            return cutlass_scaled_mm[1]

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class XPUFp8GEMMReduceScatterPattern(ScaledMMReduceScatterPattern):

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            input_scale: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            mm = torch.ops._xpu_C.fp8_gemm.default(
                x, weight, self.dtype, input_scale, weight_scale, None
            )
            return torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        def replacement(
                input: torch.Tensor,
                mat2: torch.Tensor,
                scale_a: torch.Tensor,
                scale_b: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherXPUFp8GEMMPattern(AllGatherScaledMMPattern):

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
                x: torch.Tensor,
                weight: torch.Tensor,
                scale_a: torch.Tensor,
                scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )
            return torch.ops._xpu_C.fp8_gemm.default(
                all_gather, weight,self.dtype, scale_a, scale_b, None
            )

        def replacement(
                x: torch.Tensor,
                weight: torch.Tensor,
                scale_a: torch.Tensor,
                scale_b: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class XPUMxFp8GEMMReduceScatterPattern(BasePattern):
    """XPU: fuse fp8_gemm + reduce_scatter for MXFP8 block-32.

    Matches:
        fp8_gemm(x_fp8, weight, dtype, x_scale, weight_scale, None)
            -> reduce_scatter
    """

    def get_inputs(self) -> list[torch.Tensor]:
        # x_fp8: local MXFP8 activation, last dim divisible by MXFP8_BLOCK_SIZE=32
        x_fp8 = torch.empty([16, 32], device=self.device, dtype=FP8_DTYPE)
        # x_scale: e8m0fnu scale, shape [M, K//32]
        x_scale = torch.empty([16, 1], device=self.device, dtype=torch.float8_e8m0fnu)
        weight = (
            torch.empty([32, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
        )
        weight_scale = torch.empty(
            [1, 16], device=self.device, dtype=torch.float8_e8m0fnu
        )
        return [x_fp8, x_scale, weight, weight_scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x_fp8: torch.Tensor,
            x_scale: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            mm = torch.ops._xpu_C.fp8_gemm.default(
                x_fp8, weight, self.dtype, x_scale, weight_scale, None
            )
            return torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        def replacement(
            x_fp8: torch.Tensor,
            x_scale: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.fused_xpu_mxfp8_matmul_reduce_scatter.default(
                x_fp8,
                x_scale,
                weight,
                weight_scale,
                self.dtype,
                self.tp.device_group.group_name,
            )

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherXPUMxFp8GEMMPattern(BasePattern):
    """XPU: fuse 2×all_gather + fp8_gemm for MXFP8 block-32.

    After the SP pass, the graph for a column-parallel linear looks like:
        reduce_scatter -> rms_norm -> xpu_mxfp8_quantize
                                         |-> all_gather(fp8_local)   -> fp8_full
                                         |-> all_gather(scale_local) -> scale_full
                                         fp8_gemm(fp8_full, weight, dtype, scale_full, weight_scale)

    This pattern fuses the two all_gathers with the subsequent fp8_gemm.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        # K must be divisible by 64 so that K//32 is even, allowing
        # view(float8_e8m0fnu -> bfloat16) on the scale tensor during tracing.
        fp8_local = torch.empty([8, 64], device=self.device, dtype=FP8_DTYPE)
        # scale_local: e8m0fnu scale, shape [M_local, K//32] = [8, 2]
        scale_local = torch.empty([8, 2], device=self.device, dtype=torch.float8_e8m0fnu)
        weight = torch.empty([64, 16], device=self.device, dtype=FP8_DTYPE).contiguous()
        weight_scale = torch.empty(
            [1, 16], device=self.device, dtype=torch.float8_e8m0fnu
        )
        return [fp8_local, scale_local, weight, weight_scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # def pattern(
        #     fp8_local: torch.Tensor,
        #     scale_local: torch.Tensor,
        #     weight: torch.Tensor,
        #     weight_scale: torch.Tensor,
        # ) -> torch.Tensor:
        #     fp8_full = torch.ops.vllm.all_gather.default(
        #         fp8_local, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
        #     )
        #     scale_full = torch.ops.vllm.all_gather.default(
        #         scale_local, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
        #     )
        #     return torch.ops._xpu_C.fp8_gemm.default(
        #         fp8_full, weight, self.dtype, scale_full, weight_scale, None
        #     )

        def replacement(
            fp8_local: torch.Tensor,
            scale_local: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.fused_all_gather_xpu_mxfp8_matmul.default(
                fp8_local, scale_local, weight, weight_scale,
                self.dtype, self.tp.device_group.group_name
            )

        # pm.register_replacement(
        #     pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        # )

        def pattern(
            fp8_local: torch.Tensor,
            scale_local: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ) -> torch.Tensor:
            # fp8: pack as bfloat16 for allgather (XCCL SYCL GPU: no uint8 support)
            fp8_bf16 = torch.ops.aten.view.dtype(fp8_local, torch.bfloat16)
            fp8_bf16_full = torch.ops.vllm.all_gather.default(
                fp8_bf16, 0, self.tp_size, self.tp.unique_name
            )
            fp8_full = torch.ops.aten.view.dtype(fp8_bf16_full, torch.float8_e4m3fn)
            # scale: same bfloat16 packing trick for float8_e8m0fnu
            scale_bf16 = torch.ops.aten.view.dtype(scale_local, torch.bfloat16)
            scale_bf16_full = torch.ops.vllm.all_gather.default(
                scale_bf16,
                0,
                self.tp_size,
                self.tp.unique_name,
            )
            scale_full = torch.ops.aten.view.dtype(
                scale_bf16_full, torch.float8_e8m0fnu
            )
            return torch.ops._xpu_C.fp8_gemm.default(
                fp8_full, weight, self.dtype, scale_full, weight_scale, None
            )

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
        )


class FlashInferBMMFP8ReduceScatterPattern(
    BasePattern, VllmPatternReplacement[..., torch.Tensor]
):
    def get_inputs(self) -> list[torch.Tensor]:
        a_2d = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        b_2d = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        a_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        b_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        return [a_2d, b_2d, a_scale, b_scale]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
            a_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            bmm = torch.ops.vllm.bmm_fp8.default(
                torch.ops.aten.unsqueeze.default(a_2d, 0),
                torch.ops.aten.unsqueeze.default(b_2d, 0),
                a_scale,
                b_scale,
                self.dtype,
                "auto",
            )
            output = torch.ops.aten.reshape.default(bmm, list(bmm.shape[1:]))
            return torch.ops.vllm.reduce_scatter.default(
                output,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            a_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default(
                a_2d,
                b_2d,
                a_scale,
                b_scale,
                "sum",
                0,
                0,
                self.tp.device_group.group_name,
                [a_2d.shape[0], b_2d.shape[1]],
                self.dtype,
            )

        return _replacement


class FlashInferAllGatherBMMFP8Pattern(
    BasePattern, VllmPatternReplacement[..., torch.Tensor]
):
    def get_inputs(self) -> list[torch.Tensor]:
        a_shard_2d = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        b_2d = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        a_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        b_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        return [a_shard_2d, b_2d, a_scale, b_scale]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                a_shard_2d,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return torch.ops.vllm.bmm_fp8.default(
                torch.ops.aten.unsqueeze.default(all_gather, 0),
                torch.ops.aten.unsqueeze.default(b_2d, 0),
                a_scale,
                b_scale,
                self.dtype,
                "auto",
            )

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            fused = torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default(
                a_shard_2d,
                b_2d,
                a_scale,
                b_scale,
                0,
                self.tp.device_group.group_name,
                self.dtype,
            )
            return torch.ops.aten.unsqueeze.default(fused, 0)

        return _replacement


class FlashInferAllGatherFP4Pattern(
    BasePattern, VllmPatternReplacement[..., torch.Tensor]
):
    def __init__(
        self,
        dtype: torch.dtype,
        device: str | None,
        backend: str,
        use_8x4_sf_layout: bool,
        a_scale_view: str,
    ) -> None:
        super().__init__(dtype, device)
        self.backend = backend
        self.use_8x4_sf_layout = use_8x4_sf_layout
        self.a_scale_view = a_scale_view

    def get_inputs(self) -> list[torch.Tensor]:
        a_shard_2d = torch.empty([8, 8], device=self.device, dtype=torch.uint8)
        b_2d = torch.empty([8, 16], device=self.device, dtype=torch.uint8)
        a_scale_shard = torch.empty([128, 4], device=self.device, dtype=torch.int32)
        b_scale = torch.empty([4, 128], device=self.device, dtype=torch.uint8)
        alpha = torch.empty([], device=self.device, dtype=torch.float32)
        return [
            a_shard_2d,
            b_2d,
            a_scale_shard,
            b_scale,
            alpha,
        ]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale_shard: torch.Tensor,
            b_scale: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            all_gather_a = torch.ops.vllm.all_gather.default(
                a_shard_2d,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            all_gather_a_scale = torch.ops.vllm.all_gather.default(
                a_scale_shard,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            a_scale = all_gather_a_scale
            if self.a_scale_view in ("float8", "float8_uint8"):
                a_scale = torch.ops.aten.view.dtype(a_scale, torch.float8_e4m3fn)
            if self.a_scale_view in ("uint8", "float8_uint8"):
                a_scale = torch.ops.aten.view.dtype(a_scale, torch.uint8)
            return torch.ops.vllm.flashinfer_mm_fp4.default(
                all_gather_a,
                b_2d,
                a_scale,
                b_scale,
                alpha,
                self.dtype,
                self.use_8x4_sf_layout,
                self.backend,
            )

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale_shard: torch.Tensor,
            b_scale: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.fused_all_gather_flashinfer_fp4_matmul.default(
                a_shard_2d,
                b_2d,
                a_scale_shard,
                b_scale,
                alpha,
                0,
                self.tp.device_group.group_name,
                self.dtype,
                self.a_scale_view in ("float8", "float8_uint8"),
                self.use_8x4_sf_layout,
                self.backend,
            )

        return _replacement


class AsyncTPPass(VllmFusionPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, pass_name="async_tp_pass")

        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        GEMMReduceScatterPattern(self.model_dtype, self.device).register(self.pm_pass)

        AllGatherGEMMPattern(self.model_dtype, self.device).register(self.pm_pass)

        # These fusions are enabled only for bfloat16 models because
        # `scaled_mm` or `cutlass_scaled_mm` with per-token (row-wise) scaling
        # only supports bfloat16 as the output dtype.
        if self.model_dtype == torch.bfloat16:
            ScaledMMReduceScatterPattern(self.model_dtype, self.device).register(
                self.pm_pass
            )
            AllGatherScaledMMPattern(self.model_dtype, self.device).register(
                self.pm_pass
            )

            if current_platform.is_cuda():
                CutlassScaledMMReduceScatterPattern(self.model_dtype, self.device).register(
                    self.pm_pass
                )
                AllGatherCutlassScaledMMPattern(self.model_dtype, self.device).register(
                    self.pm_pass
                )
            with suppress(ImportError):
                import vllm.utils.flashinfer  # noqa: F401
            if hasattr(torch.ops.vllm, "bmm_fp8"):
                self.register(
                    FlashInferAllGatherBMMFP8Pattern(self.model_dtype, self.device)
                )
                self.register(
                    FlashInferBMMFP8ReduceScatterPattern(self.model_dtype, self.device)
                )
            if hasattr(torch.ops.vllm, "flashinfer_mm_fp4"):
                for backend in ("cutlass", "cudnn"):
                    for a_scale_view in ("float8_uint8", "uint8"):
                        self.register(
                            FlashInferAllGatherFP4Pattern(
                                self.model_dtype,
                                self.device,
                                backend,
                                use_8x4_sf_layout=False,
                                a_scale_view=a_scale_view,
                            )
                        )
                for use_8x4_sf_layout in (False, True):
                    for a_scale_view in ("float8",):
                        self.register(
                            FlashInferAllGatherFP4Pattern(
                                self.model_dtype,
                                self.device,
                                "trtllm",
                                use_8x4_sf_layout=use_8x4_sf_layout,
                                a_scale_view=a_scale_view,
                            )
                        )
                # NVFP4 reduce-scatter does not need scale communication: FP4
                # scales are consumed by the local GEMM and only BF16 partial
                # outputs are reduced. Keep this PR scoped to the all-gather
                # path; reduce-scatter needs a dedicated FP4 producer rather
                # than the existing FP8-style helper.

            if current_platform.is_xpu():
                # XPU W8A8 FP8 linear now emits _xpu_C.fp8_gemm (not
                # aten._scaled_mm), so register the dedicated XPU fp8_gemm
                # patterns for both RS and AG paths.
                XPUFp8GEMMReduceScatterPattern(
                    self.model_dtype, self.device
                ).register(self.pm_pass)
                AllGatherXPUFp8GEMMPattern(
                    self.model_dtype, self.device
                ).register(self.pm_pass)
                # MXFP8 checkpoints emit xpu_mxfp8_quantize + fp8_gemm.
                # XPUMxFp8GEMMReduceScatterPattern(
                #     self.model_dtype, self.device
                # ).register(self.pm_pass)
                # AllGatherXPUMxFp8GEMMPattern(
                #     self.model_dtype, self.device
                # ).register(self.pm_pass)

        self.dump_patterns(config, self.pm_pass)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # This pass is applied on top of the sequence parallelism pass,
        # which is only supported in fullgraph compilation mode.
        assert (
            self.compilation_config.use_inductor_graph_partition
            or not self.compilation_config.splitting_ops
        ), "AsyncTPPass requires full-graph compilation"
        return True

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.pm_pass.apply(graph)
        # if False:
        #     count = self._rewrite_all_gather_xpu_mxfp8_matmul(graph)
        #     logger.info("Rewrote %s XPU MXFP8 quantize patterns", count)
        #     if self.matched_count == 0 and count == 0:
        #         # Debug: show what fp8_gemm and reduce_scatter nodes look like
        #         fp8_gemm_op = torch.ops._xpu_C.fp8_gemm.default
        #         rs_op = torch.ops.vllm.reduce_scatter.default
        #         fp8_nodes = list(graph.find_nodes(op="call_function", target=fp8_gemm_op))
        #         rs_nodes = list(graph.find_nodes(op="call_function", target=rs_op))
        #         if fp8_nodes or rs_nodes:
        #             logger.warning("[DBG AsyncTP small bucket] fp8_gemm nodes=%d, reduce_scatter nodes=%d",
        #                            len(fp8_nodes), len(rs_nodes))
        #             for n in fp8_nodes[:2]:
        #                 logger.warning("[DBG AsyncTP] fp8_gemm args: %s",
        #                                [(a, type(a).__name__) for a in n.args])
        #                 for u in n.users:
        #                     logger.warning("[DBG AsyncTP] fp8_gemm user: op=%s target=%s args=%s",
        #                                    u.op, u.target, [(a, type(a).__name__) for a in u.args])
        #             for n in rs_nodes[:2]:
        #                 logger.warning("[DBG AsyncTP] reduce_scatter args: %s",
        #                                [(a, type(a).__name__) for a in n.args])
        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        logger.info("Replaced %s patterns", self.matched_count)

    def _rewrite_all_gather_xpu_mxfp8_matmul(self, graph: fx.Graph) -> int:
        """Fuse the XPU MXFP8 all-gather + GEMM graph shape emitted by SP.

        XCCL SYCL GPU only supports float16/bfloat16/float32 for allgather;
        uint8 and int8 are not supported.  SP therefore packs 8-bit tensors
        into bfloat16 before gathering:
            fp8_full  = view(all_gather(view(fp8_local,   bfloat16)), float8_e4m3fn)
            scale_full = view(all_gather(view(scale_local, bfloat16)), float8_e8m0fnu)
            fp8_gemm(fp8_full, weight, dtype, scale_full, weight_scale, None)
        """
        all_gather_op = torch.ops.vllm.all_gather.default
        view_dtype_op = torch.ops.aten.view.dtype
        fp8_gemm_op = torch.ops._xpu_C.fp8_gemm.default
        fused_op = torch.ops.vllm.fused_all_gather_xpu_mxfp8_matmul.default
        count = 0

        for gemm in list(graph.find_nodes(op="call_function", target=fp8_gemm_op)):
            if len(gemm.args) < 5:
                continue

            fp8_full = gemm.args[0]
            scale_full = gemm.args[3]

            # fp8_full must be: view(all_gather(view(fp8_local, bfloat16)), float8_e4m3fn)
            if not (
                isinstance(fp8_full, fx.Node)
                and fp8_full.op == "call_function"
                and fp8_full.target == view_dtype_op
                and len(fp8_full.args) == 2
                and fp8_full.args[1] == torch.float8_e4m3fn
            ):
                continue

            fp8_bf16_full = fp8_full.args[0]
            if not (
                isinstance(fp8_bf16_full, fx.Node)
                and fp8_bf16_full.op == "call_function"
                and fp8_bf16_full.target == all_gather_op
            ):
                continue

            fp8_bf16_local = fp8_bf16_full.args[0]
            if not (
                isinstance(fp8_bf16_local, fx.Node)
                and fp8_bf16_local.op == "call_function"
                and fp8_bf16_local.target == view_dtype_op
                and len(fp8_bf16_local.args) == 2
                and fp8_bf16_local.args[1] == torch.bfloat16
            ):
                continue

            fp8_local = fp8_bf16_local.args[0]
            if not isinstance(fp8_local, fx.Node):
                continue

            scale_local: fx.Node | None = None
            scale_nodes_to_erase: list[fx.Node] = []

            # scale_full must be: view(all_gather(view(scale_local, bfloat16)), float8_e8m0fnu)
            if (
                isinstance(scale_full, fx.Node)
                and scale_full.op == "call_function"
                and scale_full.target == view_dtype_op
                and len(scale_full.args) == 2
                and scale_full.args[1] == torch.float8_e8m0fnu
            ):
                scale_bf16_full = scale_full.args[0]
                if not (
                    isinstance(scale_bf16_full, fx.Node)
                    and scale_bf16_full.op == "call_function"
                    and scale_bf16_full.target == all_gather_op
                ):
                    continue
                # both all_gather nodes must use the same group
                if (
                    fp8_bf16_full.args[1:] != scale_bf16_full.args[1:]
                    or fp8_bf16_full.kwargs != scale_bf16_full.kwargs
                ):
                    continue
                scale_bf16_local = scale_bf16_full.args[0]
                if not (
                    isinstance(scale_bf16_local, fx.Node)
                    and scale_bf16_local.op == "call_function"
                    and scale_bf16_local.target == view_dtype_op
                    and len(scale_bf16_local.args) == 2
                    and scale_bf16_local.args[1] == torch.bfloat16
                ):
                    continue
                scale_local = scale_bf16_local.args[0]
                if not isinstance(scale_local, fx.Node):
                    continue
                scale_nodes_to_erase = [scale_full, scale_bf16_full, scale_bf16_local]
            elif (
                # Fallback: scale_full is a bare all_gather (no bfloat16 wrapping)
                isinstance(scale_full, fx.Node)
                and scale_full.op == "call_function"
                and scale_full.target == all_gather_op
            ):
                if (
                    fp8_bf16_full.args[1:] != scale_full.args[1:]
                    or fp8_bf16_full.kwargs != scale_full.kwargs
                ):
                    continue
                scale_local = scale_full.args[0]
                if not isinstance(scale_local, fx.Node):
                    continue
                scale_nodes_to_erase = [scale_full]
            else:
                continue

            with graph.inserting_before(gemm):
                fused = graph.call_function(
                    fused_op,
                    args=(
                        fp8_local,
                        scale_local,
                        gemm.args[1],
                        gemm.args[4],
                        gemm.args[2],
                        get_tp_group().device_group.group_name,
                    ),
                )
            fused.meta.update(gemm.meta)
            gemm.replace_all_uses_with(fused)

            for node in [gemm, *scale_nodes_to_erase, fp8_full, fp8_bf16_full, fp8_bf16_local]:
                if not node.users:
                    graph.erase_node(node)
            count += 1

        return count
