#!/usr/bin/env bash
# Report which inference backend is currently serving on the B70.
# Only one backend fits the card at a time, so normally at most one line prints.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh" || exit 1

docker_running() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$1"; }
label_from_gguf() {
  local gguf="$1"
  [ -n "$gguf" ] && basename "$gguf" .gguf || printf '%s' "$LLM_MODEL_LABEL"
}

found=0
if docker_running "$VLLM_CONTAINER"; then
  echo "vLLM XPU - docker '$VLLM_CONTAINER' on http://127.0.0.1:${VLLM_PORT} (Qwen3.8-27B dense)"
  found=1
fi
if docker_running "$LLAMA_SYCL_CONTAINER"; then
  # Read the actual GGUF out of the container command, so -dense/-moe overrides
  # are reported correctly even though env.sh selects the other model.
  gguf="$(docker inspect -f '{{range .Args}}{{println .}}{{end}}' "$LLAMA_SYCL_CONTAINER" 2>/dev/null | grep -m1 '\.gguf$')"
  echo "llama.cpp SYCL - docker '$LLAMA_SYCL_CONTAINER' on http://127.0.0.1:${LLAMA_PORT} ($(label_from_gguf "$gguf"))"
  found=1
fi

if [ "$found" -eq 0 ]; then
  echo "No backend running. env.sh selects LLM_MODEL=$LLM_MODEL ($LLM_MODEL_LABEL)."
fi
exit 0
