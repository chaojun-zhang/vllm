# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from torch.distributed._symmetric_memory import (
    _pipelined_multi_all_gather_and_consume,
    enable_symm_mem_for_group,
)

from vllm.distributed import get_tp_group
from vllm.model_executor.kernels.fused_comm.base import (
    FusedAllGatherGemmKernel,
    FusedGemmReduceScatterKernel,
)
from vllm.model_executor.kernels.fused_comm.fp8.utils import (
    as_scaled_mm_scales,
    resolve_fp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.mxfp8.xpu import XPUMxFp8LinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUFp8BlockScaledMMKernel,
    XPUW8A8FP8LinearKernel,
)
from vllm.model_executor.layers.linear import LinearBase, RowParallelLinear
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    xpu_mxfp8_quantize as quant_mxfp8,
)


class XPUAllGatherScaledMM(FusedAllGatherGemmKernel):
    """Fused AllGather+GEMM for a per-tensor/per-channel FP8 ``quant_method``
    -- either vLLM's native ``Fp8LinearMethod`` or a ``compressed-tensors``
    FP8 scheme (``CompressedTensorsLinearMethod`` + ``layer.scheme``)."""

    def __init__(self, layer: LinearBase) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert fp8_linear is not None
        self._fp8_kernel = fp8_linear
        # Only a static activation scale is identical on every rank, so only
        # then can the gathered shards share one tensor-wise scale.
        self._static_act_scale = fp8_linear.config.activation_quant_key.scale.static
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_scaled_mm

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if fp8_linear is None:
            return False, (
                "requires Fp8LinearMethod or a compressed-tensors FP8 scheme"
            )
        if not isinstance(fp8_linear, XPUW8A8FP8LinearKernel):
            return False, "requires XPUW8A8FP8LinearKernel"
        return True, None

    def target_gemm_method(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        As, Bs = as_scaled_mm_scales(A, B, As, Bs, tensorwise=self._static_act_scale)
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            A,
            [B],
            As,
            [Bs],
            gather_dim=0,
            biases=[bias],
            result_scales=[None],
            out_dtypes=[out_dtype],
            use_fast_accum=[False],
            group_name=self._group_name,
        )
        return mm_outputs[0]


class XPUScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """Fused GEMM+ReduceScatter counterpart of ``XPUAllGatherScaledMM``."""

    def __init__(self, layer: RowParallelLinear) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert fp8_linear is not None
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_scaled_mm

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if fp8_linear is None:
            return False, (
                "requires Fp8LinearMethod or a compressed-tensors FP8 scheme"
            )
        if not isinstance(fp8_linear, XPUW8A8FP8LinearKernel):
            return False, "requires XPUW8A8FP8LinearKernel"
        return True, None

    def target_gemm_method(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        As, Bs = as_scaled_mm_scales(A, B, As, Bs, tensorwise=True)
        output = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
            A,
            B,
            As,
            Bs,
            "sum",
            0,
            0,
            self._group_name,
            output_shape,
            None,
            None,
            out_dtype,
            False,
        )
        return output


class XPUAllGatherBlockScaledMM(FusedAllGatherGemmKernel):
    """Fused AllGather+GEMM for a block-quantized FP8 ``quant_method``
    (vLLM's native ``Fp8LinearMethod`` or a compressed-tensors block FP8
    scheme)."""

    def __init__(self, layer: LinearBase) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert fp8_linear is not None
        self._fp8_kernel = fp8_linear
        self._group = get_tp_group()
        enable_symm_mem_for_group(self._group.device_group.group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if fp8_linear is None:
            return False, (
                "requires Fp8LinearMethod or a compressed-tensors FP8 scheme"
            )
        if not isinstance(fp8_linear, XPUFp8BlockScaledMMKernel):
            return False, "requires XPUFp8BlockScaledMMKernel"
        return True, None

    def target_gemm_method(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[-1])
        params = self._fp8_kernel._get_layer_params(layer)
        x_q, a_scale = self._fp8_kernel.quant_fp8(
            x_2d, params.input_scale, params.input_scale_ub
        )
        weight_t = params.weight.t()  # [N, K] -> [K, N] view
        weight_scale_t = params.block_scale.t()  # [n_blocks, k_blocks] view
        out_dtype = self._fp8_kernel.config.out_dtype

        world_size = self._group.world_size
        output = x_q.new_empty(
            x_q.shape[0] * world_size, weight_t.shape[1], dtype=out_dtype
        )
        output_shards = output.chunk(world_size)
        A = x_q.new_empty(x_q.shape[0] * world_size, x_q.shape[1])
        A_scale = a_scale.new_empty(a_scale.shape[0] * world_size, a_scale.shape[1])

        def shard_consumer(shards: list[torch.Tensor], rank: int) -> None:
            torch.ops.aten._scaled_mm.out(
                shards[0],
                weight_t,
                scale_a=shards[1],
                scale_b=weight_scale_t,
                out_dtype=out_dtype,
                out=output_shards[rank],
            )

        _pipelined_multi_all_gather_and_consume(
            [x_q, a_scale],
            shard_consumer,
            [A, A_scale],
            self._group.device_group.group_name,
            False,
        )
        if bias is not None:
            output = output + bias
        return output


class XPUBlockScaledMMReduceScatter(FusedGemmReduceScatterKernel):
    """Fused GEMM+ReduceScatter counterpart of ``XPUAllGatherBlockScaledMM``."""

    def __init__(self, layer: RowParallelLinear) -> None:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        assert fp8_linear is not None
        self._fp8_kernel = fp8_linear
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        fp8_linear = resolve_fp8_linear_kernel(layer)
        if fp8_linear is None:
            return False, (
                "requires Fp8LinearMethod or a compressed-tensors FP8 scheme"
            )
        if not isinstance(fp8_linear, XPUFp8BlockScaledMMKernel):
            return False, "requires XPUFp8BlockScaledMMKernel"
        return True, None

    def target_gemm_method(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x_2d = x.view(-1, x.shape[-1])
        params = self._fp8_kernel._get_layer_params(layer)
        weight_t = params.weight.t()
        weight_scale_t = params.block_scale.t()
        x_q, a_scale = self._fp8_kernel.quant_fp8(
            x_2d, params.input_scale, params.input_scale_ub
        )
        out_dtype = self._fp8_kernel.config.out_dtype

        output_shape = [*x.shape[:-1], params.weight.shape[0]]
        output = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
            x_q,
            weight_t,
            a_scale,
            weight_scale_t,
            "sum",
            0,
            0,
            self._group_name,
            output_shape,
            None,
            None,
            out_dtype,
            False,
        )
        return output


class XPUAllGatherMXFP8(FusedAllGatherGemmKernel):
    """MXFP8 fused AllGather+GEMM for ``XPUMxFp8LinearKernel``.

    Structurally identical to ``XPUAllGatherBlockScaledMM`` (MXFP8
    activation quantization is also dynamic/per-rank, so this installs onto
    ``apply_weights`` too), with two differences driven by
    ``XPUMxFp8LinearKernel.apply_weights``: the bias is passed straight into
    the GEMM (not added afterwards), and the ``e8m0`` scale dtype must be
    bitcast to ``uint8`` around the collective -- the collective backend
    (e.g. XCCL) does not support ``float8_e8m0fnu`` directly, so the scale
    is gathered as its bitwise-identical unsigned view and viewed back
    before the GEMM.
    """

    def __init__(self, layer: LinearBase) -> None:
        quant_method = layer.quant_method
        self._fp8_kernel = quant_method.kernel
        self._group = get_tp_group()
        enable_symm_mem_for_group(self._group.device_group.group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        quant_method = layer.quant_method
        if not isinstance(quant_method, Mxfp8OnlineLinearMethod):
            return False, "requires Mxfp8OnlineLinearMethod"
        if not isinstance(quant_method.kernel, XPUMxFp8LinearKernel):
            return False, "requires XPUMxFp8LinearKernel"
        return True, None

    def target_gemm_method(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        x_2d = x.view(-1, x.shape[-1])
        x_q, a_scale = quant_mxfp8(x_2d)
        weight_t = layer.weight.t()  # [N, K] -> [K, N] view
        weight_scale_t = layer.weight_scale.t()  # [k_blocks, n_blocks] view

        scale_dtype = a_scale.dtype
        a_scale_u8 = a_scale.view(torch.uint8)

        world_size = self._group.world_size
        output = x_q.new_empty(
            x_q.shape[0] * world_size, weight_t.shape[1], dtype=out_dtype
        )
        output_shards = output.chunk(world_size)
        A = x_q.new_empty(x_q.shape[0] * world_size, x_q.shape[1])
        A_scale = a_scale_u8.new_empty(
            a_scale_u8.shape[0] * world_size, a_scale_u8.shape[1]
        )

        def shard_consumer(shards: list[torch.Tensor], rank: int) -> None:
            torch.ops.aten._scaled_mm.out(
                shards[0],
                weight_t,
                scale_a=shards[1].view(scale_dtype),
                scale_b=weight_scale_t,
                bias=bias,
                out_dtype=out_dtype,
                out=output_shards[rank],
            )

        _pipelined_multi_all_gather_and_consume(
            [x_q, a_scale_u8],
            shard_consumer,
            [A, A_scale],
            self._group.device_group.group_name,
            False,
        )
        return output


class XPUMXFP8ReduceScatter(FusedGemmReduceScatterKernel):
    """MXFP8 fused GEMM+ReduceScatter counterpart of
    ``XPUAllGatherMXFP8``.

    As with the block-FP8 ReduceScatter direction, the public
    ``fused_scaled_matmul_reduce_scatter`` op's scale check has no
    restriction on the scale's trailing dim, and its scale handling is
    dtype-agnostic (shape-based only), so the ``float8_e8m0fnu`` scale can
    be passed straight through without the AllGather direction's bitcast --
    no collective ever touches the scale here, it stays local to this rank.
    """

    def __init__(self, layer: RowParallelLinear) -> None:
        quant_method = layer.quant_method
        self._fp8_kernel = quant_method.kernel
        self._group_name = get_tp_group().device_group.group_name
        enable_symm_mem_for_group(self._group_name)
        super().__init__(layer)

    def source_gemm_method(self):
        return self._fp8_kernel.apply_weights

    @classmethod
    def is_supported(cls, layer: LinearBase) -> tuple[bool, str | None]:
        quant_method = layer.quant_method
        if not isinstance(quant_method, Mxfp8OnlineLinearMethod):
            return False, "requires Mxfp8OnlineLinearMethod"
        if not isinstance(quant_method.kernel, XPUMxFp8LinearKernel):
            return False, "quant_method.kernel must be XPUMxFp8LinearKernel"
        return True, None

    def target_gemm_method(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        x_2d = x.view(-1, x.shape[-1])
        x_q, a_scale = quant_mxfp8(x_2d)
        weight_t = layer.weight.t()
        weight_scale_t = layer.weight_scale.t()

        output_shape = [*x.shape[:-1], layer.weight.shape[0]]
        output = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
            x_q,
            weight_t,
            a_scale,
            weight_scale_t,
            "sum",
            0,
            0,
            self._group_name,
            output_shape,
            None,
            None,
            out_dtype,
            False,
        )
        return output
