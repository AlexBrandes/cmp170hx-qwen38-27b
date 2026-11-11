#!/usr/bin/env bash
# Download the checkpoint and the drafter at the pinned revisions and assemble
# the served checkpoint, then check every file by sha256.
#   ./download.sh       (MODELS_DIR=./models by default; ~16 GB kept, ~27 GB peak)
# The served checkpoint is HyperQwen's "fast" layout (https://github.com/syv-ai/HyperQwen):
#   1. dbirks/Qwen3.8-27B-W4A16-AutoRound at MODEL_REVISION;
#   2. HyperQwen's prepare/quant_lm_head.py and prepare/quant_embed.py (copied
#      unmodified to tools/, Apache-2.0): int8 group-128 LM head and embedding
#      table, a few CPU minutes in the engine image;
#   3. syvai/qwen3.8-27b-3090-fast-variant at FAST_REVISION: the int4 LM head
#      shard, the MTP module, the draft vocabulary, config and index (they
#      replace the files step 2 rewrote, except the embedding shard).
# The result must match tools/model.sha256 file for file, or the script fails.
set -euo pipefail
cd "$(dirname "$0")"
. ./versions.env
[ -f .env ] && . ./.env
IMAGE=${IMAGE:-${ENGINE_IMAGE:-$ENGINE_IMAGE_LOCAL}}
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/Qwen3.8-27B-W4A16-AutoRound-fast}
DRAFT_DIR=${DRAFT_DIR:-$MODELS_DIR/Qwen3.8-27B-DFlash2-W4A16}
mkdir -p "$MODEL_DIR" "$DRAFT_DIR"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
run() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN="${HF_TOKEN:-}" \
          -v "$PWD/tools:/tools:ro" -v "$MODEL_DIR:/model" -v "$DRAFT_DIR:/draft" \
          --entrypoint python3 "$IMAGE" "$@"; }

if (cd "$MODEL_DIR" && sha256sum --status -c "$OLDPWD/tools/model.sha256" 2>/dev/null); then
  echo "checkpoint already assembled: $MODEL_DIR"
else
  run -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='/model',
                  allow_patterns=['*.json', '*.safetensors', '*.jinja'])
print('base checkpoint ready')"
  run /tools/quant_lm_head.py /model
  run /tools/quant_embed.py /model
  run -c "
import shutil
from huggingface_hub import hf_hub_download
for f in ['model-00007-of-00007.safetensors', 'model_extra_tensors.safetensors',
          'mtp_draft_vocab_ids.pt', 'config.json', 'model.safetensors.index.json']:
    shutil.copyfile(hf_hub_download('$FAST_REPO', f, revision='$FAST_REVISION', cache_dir='/tmp/hf'),
                    '/model/' + f)
print('fast variant files ready')"
  rm -f "$MODEL_DIR"/*.bak "$MODEL_DIR"/*.bak_embed "$MODEL_DIR"/*.bak-quant "$MODEL_DIR"/*.bak-mtp
  rm -rf "$MODEL_DIR/.cache"
  (cd "$MODEL_DIR" && sha256sum -c --quiet "$OLDPWD/tools/model.sha256") \
    || { echo "download.sh: checkpoint does not match tools/model.sha256" >&2; exit 1; }
  echo "checkpoint ready: $MODEL_DIR"
fi

run -c "
from huggingface_hub import snapshot_download
snapshot_download('$DRAFT_REPO', revision='$DRAFT_REVISION', local_dir='/draft',
                  allow_patterns=['*.json', '*.safetensors'])"
rm -rf "$DRAFT_DIR/.cache"
(cd "$DRAFT_DIR" && sha256sum -c --quiet "$OLDPWD/tools/draft.sha256") \
  || { echo "download.sh: drafter does not match tools/draft.sha256" >&2; exit 1; }
echo "drafter ready: $DRAFT_DIR"
