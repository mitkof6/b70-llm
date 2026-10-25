#!/usr/bin/env bash
# Report whether the vLLM server is running on the B70.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh" || exit 1

if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$VLLM_CONTAINER"; then
  echo "vLLM XPU v0.32.0 - docker '$VLLM_CONTAINER' on http://127.0.0.1:${VLLM_PORT} ($LLM_MODEL_LABEL)"
else
  echo "No vLLM server running. Start with: b70-vllm"
fi
