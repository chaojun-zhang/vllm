# export HABANA_LOGS=~/.habana_logs
# export LOG_LEVEL_GC=0
export VLLM_MLA_DISABLE_REQUANTIZATION=1 
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"

export VLLM_EP_SIZE=8
export VLLM_SKIP_WARMUP=True
# export VLLM_LOGGING_LEVEL=DEBUG
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
# export PT_HPU_LAZY_MODE=0
export MAX_MODEL_LEN=8192
# export MODEL_PATH=/software/data/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/
export MODEL_PATH=/software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/

MOONCAKE_CONFIG_PATH=./mooncake.json python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --port 8100 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 -tp 8 --disable-async-output-proc --max-num-seqs 32  --use-padding-aware-scheduling False --trust-remote-code  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":1e10}'

