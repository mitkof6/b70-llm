#!/usr/bin/env bash
# Report which inference backend is currently serving on the B70.
# Only one backend fits the card at a time, so normally at most one line prints.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh"

docker_running() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$1"; }

found=0
if docker_running "$VLLM_CONTAINER"; then
  echo "vLLM XPU - docker '$VLLM_CONTAINER' on http://127.0.0.1:${VLLM_PORT}"
  found=1
fi
if docker_running "$LLAMA_SYCL_CONTAINER"; then
  echo "llama.cpp SYCL - docker '$LLAMA_SYCL_CONTAINER' on http://127.0.0.1:${LLAMA_PORT}"
  found=1
fi
if pgrep -f "$LLAMA_SERVER" >/dev/null 2>&1; then
  echo "llama.cpp Vulkan - host process 'llama-server' on http://127.0.0.1:${LLAMA_PORT}"
  found=1
fi
if [ "$found" -eq 0 ]; then
  echo "No backend running."
fi
exit 0
