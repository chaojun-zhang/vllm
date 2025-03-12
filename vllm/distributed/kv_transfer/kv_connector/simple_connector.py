"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache 
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import time

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import (
    SimpleBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.distributed import tensor_model_parallel_all_gather

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    from vllm.worker.hpu_model_runner import ModelInputForHPUWithSamplingMetadata
from vllm_hpu_extension.utils import VLLMKVCache

logger = init_logger(__name__)


class SimpleConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):

        self.config = config.kv_transfer_config
        self.tp_size = config.parallel_config.tensor_parallel_size
        self.local_rank = local_rank
        self.local_offset_start = (64+512) // self.tp_size * local_rank
        self.local_offset_end = (64+512) // self.tp_size * (local_rank + 1)

        if self.config.kv_connector == "PyNcclConnector":
            from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import (
                PyNcclPipe)
            logger.info(
                "Initializing PyNcclConfig under kv_transfer_config %s",
                self.config)
        elif self.config.kv_connector == "MooncakeConnector":
            # Check if MOONCAKE_CONFIG_PATH is set
            import os
            use_mooncake_distributed_pipe = os.getenv(
                'MOONCAKE_CONFIG_PATH') is not None

            if not use_mooncake_distributed_pipe:
                raise ValueError(
                    "To use MooncakeConnector, you need to pass the ENV: "
                    "'MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json'.")
            else:
                from vllm.distributed.kv_transfer.kv_pipe.mooncake_pipe import (  # noqa: E501
                    MooncakePipe)
                logger.info(
                    "Initializing MooncakeConfig under kv_transfer_config %s",
                    self.config)

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.producer_buffer: Optional[SimpleBuffer] = None
        self.consumer_buffer: Optional[SimpleBuffer] = None

        self.producer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.producer_signal_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_signal_pipe: Union[PyNcclPipe, MooncakePipe]
        self.k_head_size = 64
        self.v_head_size = 512
        self.block_size = 128
        self.padding_k_tensor = torch.zeros((self.block_size, self.k_head_size), dtype=torch.bfloat16, device="hpu")
        self.padding_v_tensor = torch.zeros((self.block_size, self.v_head_size), dtype=torch.bfloat16, device="hpu")
        self.cache_k = VLLMKVCache()
        self.cache_v = VLLMKVCache()
        # 2 pipes for every rank in the world
        port_offset_base = 2 * rank

        # In disaggregated prefill, the prefill vLLM only uses send pipe
        # and the decode vLLM only uses recv pipe
        if self.config.is_kv_producer:

            if self.config.kv_connector == "PyNcclConnector":
                self.producer_data_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base,
                )
                self.producer_signal_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base + 1,
                    device="cpu",
                )
            elif self.config.kv_connector == "MooncakeConnector":
                self.producer_data_pipe = MooncakePipe(
                    local_rank=local_rank,
                    config=self.config,
                )
                # We only need to initialize MooncakePipe once
                self.producer_signal_pipe = self.producer_data_pipe

            self.producer_buffer = SimpleBuffer(self.producer_signal_pipe,
                                                self.producer_data_pipe,
                                                self.config.kv_buffer_size)

        else:

            # the current vLLM instance is KV consumer, so it needs to connect
            # its recv pipe to the send pipe of KV producder
            if self.config.kv_connector == "PyNcclConnector":
                self.consumer_data_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base,
                )
                self.consumer_signal_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base + 1,
                    device="cpu",
                )
            elif self.config.kv_connector == "MooncakeConnector":
                self.consumer_data_pipe = MooncakePipe(
                    local_rank=local_rank,
                    config=self.config,
                )
                self.consumer_signal_pipe = self.consumer_data_pipe

            self.consumer_buffer = SimpleBuffer(
                self.consumer_signal_pipe,
                self.consumer_data_pipe,
                self.config.kv_buffer_size,
            )

    def select(self, input_tokens: Optional[torch.Tensor],
               roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

        assert self.consumer_buffer is not None, "Please initialize the "\
            "consumer buffer before calling select."
        return self.consumer_buffer.drop_select(input_tokens, roi)

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        assert self.producer_buffer is not None, "Please initialize the "\
            "producer buffer before calling insert."

        self.producer_buffer.insert(input_tokens, roi, key, value, hidden)

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]

            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)

            self.insert(current_tokens,
                        torch.ones_like(current_tokens,
                                        dtype=bool), keys, values,
                        hidden_or_intermediate_states[start_pos:end_pos])

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def send_kv_caches_and_hidden_states_hpu(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForHPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        input_tokens_tensor = model_input.input_tokens # shape: [batch_size, seq_len_padding_to_128]
        seq_lens = model_input.attn_metadata.seq_lens # 2D list
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        num_kv_heads = 1
        k_head_size = 64
        v_head_size = 512
        
        model_config = model_executable.model.config
        # not used for MLA since key and value size are not equal 
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)

        # For each sequence in the batch, we send
        # 0. current_tokens [seq_len]
        # 1. bool mask [seq_len]
        # 2. key [num_layers, seq_len, num_kv_heads, (k_head_size + v_head_size) // tp_size ], [61, seq_len, 1, 72]
        # 3. empty tensor
        # 4. hidden_or_intermediate_states [???seq_len, hidden_size]
        for idx, slen in enumerate(seq_lens):
            if slen == 1: # we think this is a padding sequence, so we skip it
                continue
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[idx][:slen]
            logger.debug(f"send token len: {slen}, token: {current_tokens}")
            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]
                # only get current rank shard 
                key_cache = kv_cache[0].reshape(-1, num_kv_heads, k_head_size)
                value_cache = kv_cache[1].reshape(-1, num_kv_heads, v_head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)
            # we pack kv together, only need send one tensor
            key_values = torch.cat([keys, values], dim=-1)
            logger.debug(f"idx: {idx}, slen: {slen}, start_pos: {start_pos}, end_pos: {end_pos}")
            logger.debug(f"keys shape: {keys.shape}, values shape: {values.shape}, hidden_or_intermediate_states: {hidden_or_intermediate_states.shape}")
            key_values = key_values[..., self.local_offset_start:self.local_offset_end]
            self.insert(current_tokens.cpu(),
                        torch.ones_like(current_tokens,
                                        dtype=bool).cpu(),
                        key_values.cpu(),
                        torch.zeros(1, device="cpu"),
                        hidden_or_intermediate_states[idx].unsqueeze(0).cpu())

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())


    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        for idx, slen in enumerate(seq_lens):

            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            ret = self.select(current_tokens,
                              torch.ones_like(current_tokens, dtype=bool))
            if ret[0] is None:
                # didn't find any match.
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                continue

            roi: torch.Tensor = ret[1]
            keys: torch.Tensor = ret[2]
            values: torch.Tensor = ret[3]
            hidden: torch.Tensor = ret[4]

            num_computed_tokens = roi.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # put received KV caches into paged memory
            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):

                kv_cache = kv_caches[i - model_executable.model.start_layer]
                layer = model_executable.model.layers[i]

                key_cache, value_cache = kv_cache[0], kv_cache[1]
                ops.reshape_and_cache_flash(
                    keys[i - model_executable.model.start_layer].to(
                        key_cache.device),
                    values[i - model_executable.model.start_layer].to(
                        value_cache.device),
                    key_cache,
                    value_cache,
                    slot_mapping[start_pos:end_pos],
                    layer.self_attn.attn.kv_cache_dtype,
                    layer.self_attn.attn._k_scale,
                    layer.self_attn.attn._v_scale,
                )

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.debug(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def recv_kv_caches_and_hidden_states_hpu(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForHPUWithSamplingMetadata",
        attn_metadata: object,
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForHPUWithSamplingMetadata"]:
        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        input_tokens_list_=input_tokens_tensor.tolist()
        seq_lens_tensor = model_input.attn_metadata.seq_lens_tensor
        seq_lens = seq_lens_tensor.tolist() #2D list
        slot_mapping = attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []
        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []
        num_kv_heads = 1
        start_block_idx = 0
        # write to cache

        # For each sequence in the batch, we recv
        # 0. current_tokens [seq_len]
        # 1. bool mask [seq_len]
        # 2. key_values [num_layers, seq_len, num_kv_heads, (k_head_size + v_head_size) // 8], [61, seq_len, 1, 72]
        # 3. empty tensor
        # 4. hidden_or_intermediate_states [???seq_len, hidden_size]
        for idx, slen in enumerate(seq_lens):
            start = time.time()

            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[idx][:slen]
            num_tokens = slen
            num_blocks = (slen + 127) // 128
            end_block_idx = start_block_idx + num_blocks
                        
            # we think this is a padding sequence, so we skip it. but we still need write kv cache
            if slen == 1:
                self.cache_k(self.padding_k_tensor.unsqueeze(0),
                        key_cache,
                        attn_metadata.block_indices[start_block_idx:end_block_idx],
                        attn_metadata.block_offsets,
                        )
                self.cache_v(self.padding_v_tensor.unsqueeze(0),
                        value_cache,
                        attn_metadata.block_indices[start_block_idx:end_block_idx],
                        attn_metadata.block_offsets,
                        )
                hidden_or_intermediate_states_for_one_req.append(hidden_or_intermediate_states_for_one_req[0])
                start_block_idx = end_block_idx
                continue

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)
            logger.debug(f"call select API from decode server, select tokens: {current_tokens.shape}")
            ret = self.select(current_tokens.cpu(),
                              torch.ones_like(current_tokens, dtype=bool).cpu())
            logger.info(f"select time takes: {time.time() - start}")
            if ret[0] is None:
                # didn't find any match.
                logger.warning(f"cannot find match, token: {current_tokens}")
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                continue

            roi: torch.Tensor = ret[1]
            key_values: torch.Tensor = ret[2]
            placeholder: torch.Tensor = ret[3]
            hidden: torch.Tensor = ret[4]

            # all gather here
            key_values = key_values.to("hpu")
            torch.hpu.synchronize()
            all_gather_start = time.time()
            key_values = tensor_model_parallel_all_gather(key_values, -1)
            torch.hpu.synchronize()
            end = time.time()
            logger.info(f"all gather time takes: {end - all_gather_start}")
            
            keys = key_values[..., :self.k_head_size]
            values = key_values[..., self.k_head_size:]
            num_computed_tokens = roi.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)
            cur = time.time()
            logger.info(f"select + allgather time for this request: {cur - start}")

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                logger.info(f"Cannot bypass, num_computed_tokens: {num_computed_tokens}, num_tokens: {num_tokens}, hidden: {hidden}")
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # put received KV caches into paged memory layer by layer
            # for each layer, we need to pad the key and value to 128, so 
            # key shape should be [num_blocks, block_size, num_kv_heads(1,ommited), k_head_size]
            # value shape should be [num_blocks, block_size, num_kv_heads(1,ommited), v_head_size]
            
            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):
                current_layer_idx = i - model_executable.model.start_layer
                kv_cache = kv_caches[current_layer_idx]
                layer = model_executable.model.layers[i] # for kv scale

                key_cache, value_cache = kv_cache[0], kv_cache[1]

                # [num_layers, seq_len, num_kv_heads, k/v_head_size] -> [seq_len, k/v_head_size]
                key = keys[current_layer_idx].squeeze(-2)
                value = values[current_layer_idx].squeeze(-2) 
                
                # [seq_len, k/v_head_size] ->(padding [seq_len % block_size, k/v_head_size]) ->
                # [num_blocks * block_size, k/v_head_size]
                key = torch.cat([key, self.padding_k_tensor[slen % self.block_size:]], dim=0)
                value = torch.cat([value, self.padding_v_tensor[slen % self.block_size:]], dim=0)
                
                # [num_blocks, block_size, k/v_head_size]
                key = key.view(num_blocks, self.block_size, self.k_head_size)
                value = value.view(num_blocks, self.block_size, self.v_head_size)

                # ====== D2D =======
                self.cache_k(key,
                        key_cache,
                        attn_metadata.block_indices[start_block_idx:end_block_idx],
                        attn_metadata.block_offsets,
                        )
                self.cache_v(value,
                        value_cache,
                        attn_metadata.block_indices[start_block_idx:end_block_idx],
                        attn_metadata.block_offsets,
                        )
            start_block_idx = end_block_idx
            hidden_or_intermediate_states_for_one_req.append(hidden.to("hpu"))
            end = time.time()
            logger.info(f"cache time: {end - cur}")
        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.warning(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0).to("hpu")

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self):
        self.producer_data_pipe.close()
        self.consumer_data_pipe.close()
        if self.config.kv_connector == "PyNcclConnector":
            self.producer_signal_pipe.close()
            self.consumer_signal_pipe.close()
        elif self.config.kv_connector == "MooncakeConnector":
            # MooncakePipe reuses data_pipe for signal_pipe, so we only have to
            # close the data_pipe.
            pass
