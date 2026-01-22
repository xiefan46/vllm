#!/bin/bash
set -euo pipefail

declare -a PIDS=()

###############################################################################
# Configuration -- override via env before running
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-/workspace/logs}"
# Clear previous logs to avoid confusion
rm -rf $LOG_PATH
mkdir -p $LOG_PATH

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-2}"
GPU_P="${GPU_P:-2}"
GPU_D="${GPU_D:-3}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/workspace/ec_cache}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12000}"   # wait_for_server timeout

NUM_PROMPTS="${NUM_PROMPTS:-100}"    # number of prompts to send in benchmark

export UCX_TLS=all
export UCX_NET_DEVICES=all

###############################################################################
# Print configuration paths for debugging
###############################################################################
echo "========== Configuration =========="
echo "MODEL: $MODEL"
echo "LOG_PATH: $LOG_PATH"
echo "EC_SHARED_STORAGE_PATH: $EC_SHARED_STORAGE_PATH"
echo "ENCODE_PORT: $ENCODE_PORT"
echo "PREFILL_PORT: $PREFILL_PORT"
echo "DECODE_PORT: $DECODE_PORT"
echo "PROXY_PORT: $PROXY_PORT"
echo "GPU_E: $GPU_E, GPU_P: $GPU_P, GPU_D: $GPU_D"
echo "==================================="

###############################################################################
# Helpers
###############################################################################
# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG=$LOG_PATH/encoder_${START_TIME}.log
P_LOG=$LOG_PATH/p_${START_TIME}.log
D_LOG=$LOG_PATH/d_${START_TIME}.log
PROXY_LOG=$LOG_PATH/proxy_${START_TIME}.log

echo "GIT_ROOT: $GIT_ROOT"
echo "ENC_LOG: $ENC_LOG"
echo "P_LOG: $P_LOG"
echo "D_LOG: $D_LOG"
echo "PROXY_LOG: $PROXY_LOG"
echo "==================================="

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

wait_for_server() {
    local port=$1
    log "Waiting for server on port $port..."
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null; do
            sleep 1
        done" && { log "Server on port $port is ready!"; return 0; } || { log "TIMEOUT waiting for server on port $port"; return 1; }
}

# Cleanup function
cleanup() {
    echo "Stopping everything…"
    trap - INT TERM USR1   # prevent re-entrancy
    
    # Kill all tracked PIDs
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null
        fi
    done
    
    # Wait a moment for graceful shutdown
    sleep 2
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done
    
    # Kill the entire process group as backup
    kill -- -$$ 2>/dev/null
    
    echo "All processes stopped."
    exit 0
}

trap cleanup INT
trap cleanup USR1
trap cleanup TERM

# clear previous cache
log "Step 1: Removing previous ec cache folder: $EC_SHARED_STORAGE_PATH"
rm -rf $EC_SHARED_STORAGE_PATH
log "Step 1: Done removing previous ec cache folder"

log "Step 2: Creating ec cache folder: $EC_SHARED_STORAGE_PATH"
mkdir -p $EC_SHARED_STORAGE_PATH
log "Step 2: Done creating ec cache folder"

###############################################################################
# Encoder worker
###############################################################################
log "Step 3: Starting Encoder worker on GPU $GPU_E, port $ENCODE_PORT..."
CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.01 \
    --port "$ENCODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 128 \
    --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    >"${ENC_LOG}" 2>&1 &

PIDS+=($!)
log "Step 3: Encoder worker started with PID ${PIDS[-1]}, log: $ENC_LOG"

###############################################################################
# Prefill worker
###############################################################################
log "Step 4: Starting Prefill worker on GPU $GPU_P, port $PREFILL_PORT..."
CUDA_VISIBLE_DEVICES="$GPU_P" \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$PREFILL_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_consumer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer"
    }' \
    >"${P_LOG}" 2>&1 &

PIDS+=($!)
log "Step 4: Prefill worker started with PID ${PIDS[-1]}, log: $P_LOG"

###############################################################################
# Decode worker
###############################################################################
log "Step 5: Starting Decode worker on GPU $GPU_D, port $DECODE_PORT..."
CUDA_VISIBLE_DEVICES="$GPU_D" \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=6000 \
vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$DECODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --allowed-local-media-path ${GIT_ROOT}/tests/v1/ec_connector/integration \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer"
    }' \
    >"${D_LOG}" 2>&1 &

PIDS+=($!)
log "Step 5: Decode worker started with PID ${PIDS[-1]}, log: $D_LOG"

# Wait for workers
log "Step 6: Waiting for all workers to be ready..."
log "Step 6a: Waiting for Encoder worker (port $ENCODE_PORT)..."
wait_for_server $ENCODE_PORT
log "Step 6b: Waiting for Prefill worker (port $PREFILL_PORT)..."
wait_for_server $PREFILL_PORT
log "Step 6c: Waiting for Decode worker (port $DECODE_PORT)..."
wait_for_server $DECODE_PORT
log "Step 6: All workers are ready!"

###############################################################################
# Proxy
###############################################################################
log "Step 7: Starting Proxy on port $PROXY_PORT..."
python ${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &

PIDS+=($!)
log "Step 7: Proxy started with PID ${PIDS[-1]}, log: $PROXY_LOG"

log "Step 8: Waiting for Proxy to be ready (port $PROXY_PORT)..."
wait_for_server $PROXY_PORT
log "Step 8: Proxy is ready!"
log "========== All services are up! =========="

###############################################################################
# Benchmark
###############################################################################
log "Step 9: Running benchmark (stream)..."
vllm bench serve \
  --model               $MODEL \
  --backend             openai-chat \
  --endpoint            /v1/chat/completions \
  --dataset-name        hf \
  --dataset-path        lmarena-ai/VisionArena-Chat \
  --seed                0 \
  --num-prompts         $NUM_PROMPTS \
  --port                $PROXY_PORT

PIDS+=($!)
log "Step 9: Benchmark completed"

###############################################################################
# Single request with local image
###############################################################################
log "Step 10: Running single request with local image (non-stream)..."
curl http://127.0.0.1:${PROXY_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "'${MODEL}'",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file://'"${GIT_ROOT}"'/tests/v1/ec_connector/integration/hato.jpg"}},
        {"type": "text", "text": "What is in this image?"}
    ]}
    ]
    }'

log "Step 10: Single request completed"

# cleanup
log "Step 11: Starting cleanup..."
cleanup