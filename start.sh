#!/usr/bin/env bash
# Start Qwen3.8-27B (W4A16, DFlash2 drafter) on one CMP 170HX card.
#   ./start.sh          set the power limit (sudo), start the container, wait for /health
# Settings: .env (see .env.example) or environment variables.
# The context window follows the card: 262,144 tokens on a 64 GB card, 163,840
# on a 40 GB card (MAX_MODEL_LEN overrides).
set -euo pipefail
cd "$(dirname "$0")"
. ./versions.env
[ -f .env ] && . ./.env

IMAGE=${IMAGE:-${ENGINE_IMAGE:-$ENGINE_IMAGE_LOCAL}}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/Qwen3.8-27B-W4A16-AutoRound-fast}
DRAFT_DIR=${DRAFT_DIR:-$MODELS_DIR/Qwen3.8-27B-DFlash2-W4A16}
CACHE_DIR=${CACHE_DIR:-$PWD/cache}
GPU=${GPU:-0}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
NAME=${NAME:-cmp170hx-qwen38-27b}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.8-27b}
HOST_TUNE=${HOST_TUNE:-1}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

die() { echo "start.sh: $*" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || die "no model at $MODEL_DIR (run ./download.sh)"
[ -f "$DRAFT_DIR/config.json" ] || die "no drafter at $DRAFT_DIR (run ./download.sh)"
case "$GPU" in *,*) die "GPU must name one card, e.g. 0" ;; esac
mem_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.total --format=csv,noheader,nounits) \
  || die "nvidia-smi cannot see GPU $GPU"
if [ -z "${MAX_MODEL_LEN:-}" ]; then
  if [ "$mem_mib" -ge 60000 ]; then MAX_MODEL_LEN=262144; else MAX_MODEL_LEN=163840; fi
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
mkdir -p "$CACHE_DIR"

if [ "$HOST_TUNE" = 1 ]; then
  sudo ./host-tune.sh "$GPU"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=$GPU\"" --shm-size 8g \
  --user "$(id -u):$(id -g)" -e HOME=/cache -w /cache -e HF_HUB_OFFLINE=1 \
  -p "$HOST:$PORT:8000" \
  -e VLLM_MARLIN_INPUT_DTYPE=int8 \
  -e VLLM_MARLIN_INT8_INCLUDE_RE='mlp|linear_attn|self_attn' \
  -e VLLM_SPEC_DECODE_ATTN=1 \
  -e VLLM_SPEC_ATTN_POLICY=ga100 \
  -e VLLM_CMP_W4A8_PREFILL_MIN_M=1024 \
  -e VLLM_CMP_I8MM=1 \
  -e VLLM_CMP_PREFILL_CONTIG_ATTN=1024 \
  -e VLLM_CMP_PREFILL_INT8_ATTN=4096 \
  -e VLLM_CMP_INT8_ACT_FUSION=1 \
  -e VLLM_CMP_PIECE_BUDGET_MS=500 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$MODEL_DIR:/model:ro" \
  -v "$DRAFT_DIR:/draft:ro" \
  -v "$CACHE_DIR:/cache" \
  "$IMAGE" \
  vllm serve /model \
    --host 0.0.0.0 --port 8000 \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --prefill-chunk-with-decodes 512 \
    --gpu-memory-utilization 0.93 \
    --attention-backend FLASH_ATTN \
    --kv-cache-dtype auto \
    --mamba-ssm-cache-dtype float16 \
    --async-scheduling \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --limit-mm-per-prompt '{"image":{"count":8},"video":{"count":1}}' \
    --media-io-kwargs '{"video":{"num_frames":32}}' \
    --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152},"cap_pixels_per_frame":true}' \
    "-cc=$(tr -d ' \n' < config/compilation.json)" \
    '--speculative-config={"method":"dflash","model":"/draft","num_speculative_tokens":7}' \
  >/dev/null

echo "started $NAME ($IMAGE) on $HOST:$PORT, ${MAX_MODEL_LEN} tokens (${mem_mib} MiB card);"
echo "loading (the first start compiles kernels)..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until curl -fs "http://$HOST:$PORT/health" >/dev/null 2>&1; do
  if ! docker ps -q -f "name=^$NAME$" | grep -q .; then
    docker logs --tail 50 "$NAME" >&2 || true
    die "container exited during startup"
  fi
  [ "$(date +%s)" -lt "$deadline" ] || die "not ready after ${READY_TIMEOUT}s (docker logs $NAME)"
  sleep 5
done
echo "ready: http://$HOST:$PORT/v1 (model $SERVED_MODEL_NAME)"
