set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

export VLLM_MLA_DISABLE_REQUANTIZATION=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"

export VLLM_EP_SIZE=8
export VLLM_SKIP_WARMUP=True
#unset VLLM_SKIP_WARMUP
#export VLLM_LOGGING_LEVEL=DEBUG
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export MAX_MODEL_LEN=2048
export MODEL_PATH=/software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/

export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_GRAPH_PROMPT_RATIO=0
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1

max_num_batched_tokens=2048
max_num_seqs=32
input_min=128
input_max=1024
output_max=1024

unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

set_bucketing

#export VLLM_PROMPT_BS_BUCKET_MIN=1
#export VLLM_PROMPT_BS_BUCKET_STEP=1
#export VLLM_PROMPT_BS_BUCKET_MAX=1

#export VLLM_PROMPT_SEQ_BUCKET_MIN=128
#export VLLM_PROMPT_SEQ_BUCKET_STEP=128
#export VLLM_PROMPT_SEQ_BUCKET_MAX=128

export PT_HPU_RECIPE_CACHE_CONFIG=cache,false,32768
unset PT_HPU_RECIPE_CACHE_CONFIG
#export VLLM_DECODE_BS_BUCKET_MIN=1
#export VLLM_DECODE_BS_BUCKET_STEP=1
#export VLLM_DECODE_BS_BUCKET_MAX=1

#export VLLM_DECODE_BLOCK_BUCKET_MIN=1
#export VLLM_DECODE_BLOCK_BUCKET_STEP=1
#export VLLM_DECODE_BLOCK_BUCKET_MAX=128

#MOONCAKE_CONFIG_PATH=../mooncake.json python3 -m vllm.entrypoints.openai.api_server --model /mnt/disk2/hf_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/ --port 8200 --max-model-len 16384 --gpu-memory-utilization 0.9 -tp 8 --max-num-seqs 32 --trust-remote-code --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":1e10}'

MOONCAKE_CONFIG_PATH=../mooncake.json python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --port 8200 --max-model-len 2048 --gpu-memory-utilization 0.9 -tp 8 --max-num-seqs 32 --trust-remote-code --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":1e10}'

