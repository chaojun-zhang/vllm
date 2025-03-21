# vLLM Disaggregated Prefill with MooncakeStore

This guide details how to implement vLLM's disaggregated prefill functionality using MooncakeStore.

## Installation

### Install Mooncake

1.  Start the Mooncake Docker container:

    ```bash
    docker run -it -d --net=host --name mooncake --ulimit memlock=-1 -t -i alogfans/mooncake:latest /bin/bash
    ```

2.  Enter the Mooncake container:

    ```bash
    docker exec -it mooncake bash
    ```

3.  Launch the Mooncake Master service:

    ```bash
    mooncake_master --port 50001
    ```

### Install vLLM

1.  Launch the Habana Docker container:

    ```bash
    docker run -it -d --runtime=habana --name deepseek-xpyd -v `pwd`:/workspace/vllm/ -v /software/data/disk10:/software/data -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/software/data/ artifactory-kfs.habana-labs.com/docker-local/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:1.20.0-521 /bin/bash
    ```

2.  Enter the Habana Docker container:

    ```bash
    docker exec -it deepseek-xpyd bash
    ```

3.  Configure HTTP Proxy (if required):

    ```bash
    export http_proxy=http://child-prc.intel.com:913
    export https_proxy=http://child-prc.intel.com:913
    export no_proxy=10.112.*,localhost,127.0.0.1
    ```

4.  Install etcd:

    ```bash
    apt update
    apt install sudo etcd -y
    ```

5.  Install vLLM:

    ```bash
    cd /workspace/
    git clone git@github.com:habanaai/vllm-fork.git vllm
    cd vllm
    git checkout dev/pd_dp
    pip install -r requirements-hpu.txt
    pip install modelscope quart
    VLLM_TARGET_DEVICE=hpu python3 setup.py develop
    ```

## Configuration

1.  Prepare the Mooncake configuration file `mooncake.json` (for both Prefill and Decode instances):

    ```json
    {
        "local_hostname": "192.168.0.137",
        "metadata_server": "etcd://192.168.0.137:2379",
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "192.168.0.137:50001"
    }
    ```

    **Note:** Please adjust `local_hostname`, `metadata_server`, and `master_server_address` to match your environment.

## Run Example

1.  Start the etcd server:

    ```bash
    etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379  >etcd.log 2>&1 &
    ```

2.  Launch multiple vLLM instances:

    ### Set Environment Variables

    ```bash
    export MODEL_PATH=/software/data/models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/
    export VLLM_MLA_DISABLE_REQUANTIZATION=1
    export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
    export VLLM_EP_SIZE=1
    export VLLM_SKIP_WARMUP=True
    export VLLM_LOGGING_LEVEL=DEBUG
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    export MAX_MODEL_LEN=8192
    ```

    ### Launch the kv_producer role instance

    ```bash
    cd /workspace/vllm
    MOONCAKE_CONFIG_PATH=./pd_distributed/mooncake.json python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --port 8100 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 -tp 1 --disable-async-output-proc --max-num-seqs 32 --enforce-eager --trust-remote-code --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'
    ```

    ### Launch the kv_consumer role instance

    ```bash
    cd /workspace/vllm
    MOONCAKE_CONFIG_PATH=./pd_distributed/mooncake.json python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --port 8200 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 -tp 1 --disable-async-output-proc --max-num-seqs 32 --enforce-eager --trust-remote-code --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
    ```

3.  Start the proxy server:

    ```bash
    cd /workspace/vllm
    python3 examples/online_serving/disagg_examples/disagg_proxy_demo.py --model $MODEL_PATH --prefill 127.0.0.1:8100 --decode 127.0.0.1:8200 --port 8123
    ```

    * `--model`: Specifies the model path, also used for the proxy server's tokenizer.
    * `--port`: Specifies the vLLM service listening port.
    * `--prefill`: Specifies the vLLM Prefill instance IP and port.
    * `--decode`: Specifies the vLLM Decode instance IP and port.

4.  Dynamically adjust Prefill and Decode instances (optional):

    ```bash
    export ADMIN_API_KEY="xxxxxxxx" # Set the admin API key

    # Add a Prefill instance
    curl -X POST "http://localhost:8123/instances/add" -H "Content-Type: application/json" -H "X-API-Key: $ADMIN_API_KEY" -d '{"type": "prefill", "instance": "localhost:8300"}'

    # Add a Decode instance
    curl -X POST "http://localhost:8123/instances/add" -H "Content-Type: application/json" -H "X-API-Key: $ADMIN_API_KEY" -d '{"type": "decode", "instance": "localhost:8301"}'

    # Get proxy status
    curl localhost:8123/status | jq
    ```

    **Note:** Replace `xxxxxxxx` with your actual API key and adjust instance addresses as needed.

## Testing

1.  Test with an OpenAI-compatible request:

    ```bash
    curl -s http://localhost:8123/v1/completions -H "Content-Type: application/json" -d '{
        "model": "${MODEL_PATH}",
        "prompt": "San Francisco is a",
        "max_tokens": 1000
    }'
    ```
    **Note:** If you are not testing on the proxy server, replace `localhost` with the proxy server's IP address.