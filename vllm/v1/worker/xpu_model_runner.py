# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import supports_xpu_graph
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as GPUModelRunnerV2,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # oneDNN fp8_gemm requires 64-byte aligned buffer pointers.
        # Per-token scale shards (float32, 4 B) have offset rank*(M//WS)*4 B;
        # alignment needs M//WS % 16 == 0, i.e. M % (16*WS) == 0.
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if self.compilation_config.pass_config.enable_sp and tp_size > 1:
            return round_up(num_scheduled_tokens, 16 * tp_size)
        return num_scheduled_tokens


class XPUModelRunnerV2(GPUModelRunnerV2):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)


@contextmanager
def _torch_cuda_wrapper():
    # replace cuda APIs with xpu APIs, this should work by default
    torch.cuda.Stream = torch.xpu.Stream
    torch.cuda.default_stream = torch.xpu.current_stream
    torch.cuda.current_stream = torch.xpu.current_stream
    torch.cuda.stream = torch.xpu.stream
    torch.cuda.mem_get_info = torch.xpu.mem_get_info
    torch.cuda.Event = torch.Event
    torch.cuda.set_stream = torch.xpu.set_stream
    if supports_xpu_graph():
        torch.cuda.graph = torch.xpu.graph
        torch.cuda.CUDAGraph = torch.xpu.XPUGraph
        torch.cuda.graph_pool_handle = torch.xpu.graph_pool_handle
    yield
