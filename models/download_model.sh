#!/usr/bin/env bash
# Resumable, retrying download of the Qwen3.8-27B vLLM checkpoint.
# Safe to run repeatedly; hf resumes partial files.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/../env.sh" || exit 1

[ -n "$HF" ] && [ -x "$HF" ] || { echo "hf CLI not found. Install it (pipx install 'huggingface_hub[cli]') or set HF=/path/to/hf"; exit 1; }

mkdir -p "$MODEL_DIR"
echo "[$(date +%H:%M:%S)] downloading $MODEL_REPO -> $MODEL_DIR"

attempt=0
while :; do
  attempt=$((attempt + 1))
  echo "[$(date +%H:%M:%S)] attempt #$attempt"
  if "$HF" download "$MODEL_REPO" --local-dir "$MODEL_DIR" --max-workers "${HF_WORKERS:-8}"; then
    echo "[$(date +%H:%M:%S)] COMPLETE"
    du -sh "$MODEL_DIR"
    ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l
    exit 0
  fi
  echo "[$(date +%H:%M:%S)] incomplete/failed; retry in 30s"
  sleep 30
done
