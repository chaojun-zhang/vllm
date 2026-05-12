# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from tests.compile.backend import TestBackend
from tests.utils import TestFP8Layer, multi_gpu_test
from vllm.compilation.passes.fusion.rms_quant_fusion import RMSNormQuantFusionPass
from vllm.compilation.passes.fusion.sequence_parallelism import SequenceParallelismPass
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.config.utils import Range
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Only test CUDA or XPU",
)

FP8_DTYPE = current_platform.fp8_dtype()
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

DEVICE_TYPE = current_platform.device_type


class TestAllReduceRMSNormModel(torch.nn.Module):
    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = [RMSNorm(hidden_size, eps) for i in range(4)]
        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(3)]

    def forward(self, x):
        z = torch.relu(x)
        x = resid = tensor_model_parallel_all_reduce(z)
        y = self.norm[0](x)

        z2 = torch.mm(y, self.w[0])
        x2 = tensor_model_parallel_all_reduce(z2)

        y2, resid = self.norm[1](x2, resid)

        z3 = torch.mm(y2, self.w[1])
        x3 = tensor_model_parallel_all_reduce(z3)

        y3, resid = self.norm[2](x3, resid)

        z4 = torch.mm(y3, self.w[2])
        x4 = tensor_model_parallel_all_reduce(z4)

        y4, resid = self.norm[3](x4, resid)
        return y4

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        return [
            torch.ops.vllm.all_gather.default,
            torch.ops.vllm.reduce_scatter.default,
        ]

    def ops_in_model(self):
        return [
            torch.ops.vllm_ir.rms_norm,
            torch.ops.vllm_ir.fused_add_rms_norm,
        ]


class TestAllReduceRMSNormStaticQuantFP8Model(torch.nn.Module):
    quant_key = kFp8StaticTensorSym

    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.vllm_config = get_current_vllm_config()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = [RMSNorm(hidden_size, eps) for i in range(4)]
        self.fp8_linear_layers = [
            TestFP8Layer(
                weight_shape=(hidden_size, hidden_size),
                activation_quant_key=self.quant_key,
                weight_quant_key=self.quant_key,
                input_dtype=self.vllm_config.model_config.dtype,
            )
            for i in range(3)
        ]

    def forward(self, hidden_states):
        # avoid having graph input be an arg to a pattern directly
        z = torch.relu(hidden_states)
        x = resid = tensor_model_parallel_all_reduce(z)
        y = self.norm[0](x)

        z2 = self.fp8_linear_layers[0](y)

        x2 = tensor_model_parallel_all_reduce(z2)
        y2, resid = self.norm[1](x2, resid)

        z3 = self.fp8_linear_layers[1](y2)

        x3 = tensor_model_parallel_all_reduce(z3)
        y3, resid = self.norm[2](x3, resid)  # use resid here

        z4 = self.fp8_linear_layers[2](y3)
        x4 = tensor_model_parallel_all_reduce(z4)
        y4, resid = self.norm[3](x4, resid)  # use resid here
        return y4

    def ops_in_model_after(self):
        return [
            torch.ops.vllm.all_gather.default,
            torch.ops.vllm.reduce_scatter.default,
        ]

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.all_reduce.default,
        ]

    def ops_in_model(self):
        if self.vllm_config.compilation_config.pass_config.fuse_norm_quant:
            return [torch.ops._C.fused_add_rms_norm_static_fp8_quant.default]
        else:
            quant_ops = (
                [torch.ops._C.static_scaled_fp8_quant.default]
                if any(layer.is_quant_fp8_enabled() for layer in self.fp8_linear_layers)
                else [torch.ops.aten.reciprocal]
            )
            return [
                torch.ops.vllm_ir.rms_norm,
                torch.ops.vllm_ir.fused_add_rms_norm,
                *quant_ops,
            ]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model_cls, custom_ops",
    [
        (TestAllReduceRMSNormModel, "+rms_norm"),
        (TestAllReduceRMSNormModel, "-rms_norm"),
        (TestAllReduceRMSNormStaticQuantFP8Model, "+rms_norm,+quant_fp8"),
        (TestAllReduceRMSNormStaticQuantFP8Model, "+rms_norm,-quant_fp8"),
        (TestAllReduceRMSNormStaticQuantFP8Model, "-rms_norm,+quant_fp8"),
        (TestAllReduceRMSNormStaticQuantFP8Model, "-rms_norm,-quant_fp8"),
    ],
)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fuse_norm_quant", [True, False])
@pytest.mark.parametrize("dynamic", [False, True])
def test_sequence_parallelism_pass(
    test_model_cls: type[torch.nn.Module],
    custom_ops: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    fuse_norm_quant: bool,
    dynamic: bool,
):
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                test_model_cls,
                custom_ops,
                batch_size,
                seq_len,
                hidden_size,
                dtype,
                fuse_norm_quant,
                dynamic,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(sequence_parallelism_pass_on_test_model, num_processes)


def test_sequence_parallelism_pass_requires_full_graph_compilation():
    vllm_config = VllmConfig()
    vllm_config.compilation_config.use_inductor_graph_partition = False
    vllm_config.compilation_config.splitting_ops = [
        "vllm::unified_attention_with_output"
    ]

    sequence_parallelism_pass = object.__new__(SequenceParallelismPass)
    sequence_parallelism_pass.compilation_config = vllm_config.compilation_config
    sequence_parallelism_pass.min_token_num = 1

    with pytest.raises(
        AssertionError,
        match="SequenceParallelismPass requires full-graph compilation",
    ):
        sequence_parallelism_pass.is_applicable_for_range(Range(start=8, end=8))


def sequence_parallelism_pass_on_test_model(
    local_rank: int,
    world_size: int,
    test_model_cls: type[torch.nn.Module],
    custom_ops: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    fuse_norm_quant: bool,
    dynamic: bool,
):
    set_random_seed(0)

    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
    )

    # initialize distributed
    init_distributed_environment(backend=current_platform.dist_backend)

    # configure vllm config for SequenceParallelismPass
    custom_ops_list = custom_ops.split(",") if custom_ops else []
    compilation_config = CompilationConfig(
        splitting_ops=[],  # avoid automatic rms_norm enablement
        cudagraph_mode=CUDAGraphMode.NONE,  # avoid piecewise warnings
        custom_ops=custom_ops_list,
        pass_config=PassConfig(
            enable_sp=True,
            fuse_norm_quant=fuse_norm_quant,
            eliminate_noops=True,
        ),
    )  # NoOp needed for fusion
    device_config = DeviceConfig(device=torch.device(DEVICE_TYPE))

    # this is a fake model name to construct the model config
    # in the vllm_config, it's not really used.
    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=device_config,
        compilation_config=compilation_config,
    )

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        noop_pass = NoOpEliminationPass(vllm_config)
        sequence_parallelism_pass = SequenceParallelismPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        assert (
            sequence_parallelism_pass.compilation_config.splitting_ops
            == vllm_config.compilation_config.splitting_ops
        )
        assert (
            sequence_parallelism_pass.compilation_config.use_inductor_graph_partition
            == vllm_config.compilation_config.use_inductor_graph_partition
        )
        passes_for_backend: list[VllmInductorPass] = [
            noop_pass,
            sequence_parallelism_pass,
        ]

        if fuse_norm_quant:
            fusion_pass = RMSNormQuantFusionPass(vllm_config)
            passes_for_backend.append(fusion_pass)

        passes_for_backend.append(cleanup_pass)

        backend = TestBackend(*passes_for_backend)

        model = test_model_cls(hidden_size)

        hidden_states = torch.randn((batch_size * seq_len, hidden_size), dtype=dtype)

        if dynamic:
            torch._dynamo.mark_dynamic(hidden_states, 0)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        assert sequence_parallelism_pass.matched_count == 4

        # In pre-nodes, all reduce should be there,
        # reduce scatter and all gather should not
        for op in model.ops_in_model_before():
            assert backend.op_count(op, before=True) == 4

        # In post-nodes, reduce scatter and all gather should be there,
        # all reduce should not
        for op in model.ops_in_model_after():
            assert backend.op_count(op, before=False) == 4

        for op in model.ops_in_model():
            assert backend.op_count(op, before=False) > 0


# ---------------------------------------------------------------------------
# XPU MXFP8 SP patterns
# ---------------------------------------------------------------------------

MXFP8_E8M0_DTYPE = torch.float8_e8m0fnu


class TestAllReduceRMSNormXPUMxFP8Model(torch.nn.Module):
    """Model with all_reduce → rms_norm/fused_add_rms_norm → xpu_mxfp8_quantize.

    Tests FirstAllReduceRMSNormXPUMxFP8Pattern (layer 0) and
    MiddleAllReduceRMSNormXPUMxFP8Pattern (layers 1-3).
    hidden_size must be divisible by 32 (MXFP8 block size).
    """

    def __init__(self, hidden_size=32, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = torch.bfloat16
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(4)]
        # 4 weights: one per layer (weight [K, N], weight_scale [K//32, N])
        self.w = [
            torch.empty([hidden_size, hidden_size], dtype=FP8_DTYPE) for _ in range(4)
        ]
        self.ws = [
            torch.empty([hidden_size // 32, hidden_size], dtype=MXFP8_E8M0_DTYPE)
            for _ in range(4)
        ]

    def forward(self, x):
        z = torch.relu(x)
        x = resid = tensor_model_parallel_all_reduce(z)
        y = self.norm[0](x)
        # First pattern: rms_norm → xpu_mxfp8_quantize
        fp8, scale = torch.ops.vllm.xpu_mxfp8_quantize(y, None)
        z2 = torch.ops._xpu_C.fp8_gemm(fp8, self.w[0], self.dtype, scale, self.ws[0], None)

        x2 = tensor_model_parallel_all_reduce(z2)
        y2, resid = self.norm[1](x2, resid)
        # Middle pattern 1
        fp8_2, scale_2 = torch.ops.vllm.xpu_mxfp8_quantize(y2, None)
        z3 = torch.ops._xpu_C.fp8_gemm(
            fp8_2, self.w[1], self.dtype, scale_2, self.ws[1], None
        )

        x3 = tensor_model_parallel_all_reduce(z3)
        y3, resid = self.norm[2](x3, resid)
        # Middle pattern 2
        fp8_3, scale_3 = torch.ops.vllm.xpu_mxfp8_quantize(y3, None)
        z4 = torch.ops._xpu_C.fp8_gemm(
            fp8_3, self.w[2], self.dtype, scale_3, self.ws[2], None
        )

        x4 = tensor_model_parallel_all_reduce(z4)
        y4, resid = self.norm[3](x4, resid)
        # Middle pattern 3: fp8_gemm uses both fp8 and scale; return resid so
        # residual_out has a consumer → full 3-tuple pattern matches (2 all_gathers)
        fp8_4, scale_4 = torch.ops.vllm.xpu_mxfp8_quantize(y4, None)
        z5 = torch.ops._xpu_C.fp8_gemm(
            fp8_4, self.w[3], self.dtype, scale_4, self.ws[3], None
        )
        return z5, resid

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        # After SP pass: reduce_scatter × 4, all_gather × 8 (fp8 + scale per layer,
        # all 4 layers match the full 3-tuple pattern variant)
        return [
            torch.ops.vllm.all_gather.default,
            torch.ops.vllm.reduce_scatter.default,
        ]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [32])  # must be multiple of 32 for MXFP8
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU MXFP8 SP patterns require xpu_mxfp8_quantize, XPU only",
)
def test_sequence_parallelism_pass_xpu_mxfp8(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dynamic: bool,
):
    num_processes = 2
    torch.multiprocessing.spawn(
        sequence_parallelism_pass_on_xpu_mxfp8_model,
        args=(num_processes, batch_size, seq_len, hidden_size, dynamic),
        nprocs=num_processes,
    )


def sequence_parallelism_pass_on_xpu_mxfp8_model(
    local_rank: int,
    world_size: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dynamic: bool,
):
    dtype = torch.bfloat16
    set_random_seed(0)
    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12346",
        }
    )

    init_distributed_environment(backend=current_platform.dist_backend)

    compilation_config = CompilationConfig(
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
        pass_config=PassConfig(enable_sp=True, eliminate_noops=True),
    )
    device_config = DeviceConfig(device=torch.device(DEVICE_TYPE))
    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=device_config,
        compilation_config=compilation_config,
    )

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        noop_pass = NoOpEliminationPass(vllm_config)
        sp_pass = SequenceParallelismPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        backend = TestBackend(noop_pass, sp_pass, cleanup_pass)

        model = TestAllReduceRMSNormXPUMxFP8Model(hidden_size)
        hidden_states = torch.randn((batch_size * seq_len, hidden_size), dtype=dtype)
        if dynamic:
            torch._dynamo.mark_dynamic(hidden_states, 0)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        assert sp_pass.matched_count == 4

        # Before: all_reduce should appear 4 times
        assert backend.op_count(torch.ops.vllm.all_reduce.default, before=True) == 4

        # After: reduce_scatter 4 times, all_gather 8 times (fp8 + scale per layer)
        assert backend.op_count(torch.ops.vllm.reduce_scatter.default, before=False) == 4
        assert backend.op_count(torch.ops.vllm.all_gather.default, before=False) == 8

        # all_reduce should be gone
        assert backend.op_count(torch.ops.vllm.all_reduce.default, before=False) == 0


class TestAllReduceRMSNormXPUDynamicTokenFP8Model(torch.nn.Module):
    """XPU scaled_mm/xpu.py W8A8 path with dynamic per-token activation scales."""

    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = torch.bfloat16
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(4)]
        self.quant = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
        self.w = [
            torch.empty([hidden_size, hidden_size], dtype=FP8_DTYPE) for _ in range(4)
        ]
        self.ws = [torch.empty([1], dtype=torch.float32) for _ in range(4)]

    def _fp8_gemm(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        fp8, scale = self.quant(x)
        return torch.ops._xpu_C.fp8_gemm(
            fp8, self.w[layer_idx], self.dtype, scale, self.ws[layer_idx], None
        )

    def forward(self, x):
        z = torch.relu(x)
        x = resid = tensor_model_parallel_all_reduce(z)
        y = self.norm[0](x)
        z2 = self._fp8_gemm(y, 0)

        x2 = tensor_model_parallel_all_reduce(z2)
        y2, resid = self.norm[1](x2, resid)
        z3 = self._fp8_gemm(y2, 1)

        x3 = tensor_model_parallel_all_reduce(z3)
        y3, resid = self.norm[2](x3, resid)
        z4 = self._fp8_gemm(y3, 2)

        x4 = tensor_model_parallel_all_reduce(z4)
        y4, resid = self.norm[3](x4, resid)
        z5 = self._fp8_gemm(y4, 3)
        return z5, resid

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        return [
            torch.ops.vllm.all_gather.default,
            torch.ops.vllm.reduce_scatter.default,
        ]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU dynamic-token FP8 SP patterns require _xpu_C.fp8_gemm",
)
def test_sequence_parallelism_pass_xpu_dynamic_token_fp8(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dynamic: bool,
):
    num_processes = 2
    torch.multiprocessing.spawn(
        sequence_parallelism_pass_on_xpu_dynamic_token_fp8_model,
        args=(num_processes, batch_size, seq_len, hidden_size, dynamic),
        nprocs=num_processes,
    )


def sequence_parallelism_pass_on_xpu_dynamic_token_fp8_model(
    local_rank: int,
    world_size: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dynamic: bool,
):
    dtype = torch.bfloat16
    set_random_seed(0)
    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12347",
        }
    )

    init_distributed_environment(backend=current_platform.dist_backend)

    compilation_config = CompilationConfig(
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
        pass_config=PassConfig(enable_sp=True, eliminate_noops=True),
    )
    device_config = DeviceConfig(device=torch.device(DEVICE_TYPE))
    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=device_config,
        compilation_config=compilation_config,
    )

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        noop_pass = NoOpEliminationPass(vllm_config)
        sp_pass = SequenceParallelismPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        backend = TestBackend(noop_pass, sp_pass, cleanup_pass)

        model = TestAllReduceRMSNormXPUDynamicTokenFP8Model(hidden_size)
        hidden_states = torch.randn((batch_size * seq_len, hidden_size), dtype=dtype)
        if dynamic:
            torch._dynamo.mark_dynamic(hidden_states, 0)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        assert sp_pass.matched_count == 4
        assert backend.op_count(torch.ops.vllm.all_reduce.default, before=True) == 4
        assert backend.op_count(torch.ops.vllm.reduce_scatter.default, before=False) == 4
        assert backend.op_count(torch.ops.vllm.all_gather.default, before=False) == 8
        assert backend.op_count(torch.ops.vllm.all_reduce.default, before=False) == 0
