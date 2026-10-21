#!/usr/bin/env bash
# Launch llama.cpp (SYCL / oneAPI) in Docker on the Arc Pro B70 — the fastest
# llama.cpp backend here (~34 t/s with MTP n=2, vs ~16 Vulkan; vLLM+MTP4 ~78).
#
# Uses the official ggml-org SYCL image (bundles the oneAPI 2026 runtime).
# SYCL enumerates the B70 as SYCL0. Stops the vLLM container first — only one
# backend fits the 32 GB card.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh"

[ -f "$LLAMA_GGUF" ] || { echo "GGUF not found: $LLAMA_GGUF" >&2; exit 1; }
GGUF_DIR="$(dirname "$LLAMA_GGUF")"
GGUF_NAME="$(basename "$LLAMA_GGUF")"

# Free the GPU: vLLM and llama.cpp cannot coexist.
if [ -n "$(docker ps -q --filter "name=^${VLLM_CONTAINER}$" 2>/dev/null)" ]; then
  echo "Stopping vLLM container '$VLLM_CONTAINER' to free the B70..."
  docker stop "$VLLM_CONTAINER" >/dev/null
fi
docker rm -f "$LLAMA_SYCL_CONTAINER" >/dev/null 2>&1 || true

echo "Starting llama.cpp SYCL ($LLAMA_SYCL_IMAGE), MTP n=$LLAMA_SYCL_SPEC_N_MAX -> port $LLAMA_PORT"
docker run -d --name "$LLAMA_SYCL_CONTAINER" \
  --device /dev/dri:/dev/dri \
  --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
  -p 127.0.0.1:${LLAMA_PORT}:${LLAMA_PORT} \
  -v "$GGUF_DIR":/models:ro \
  -v "$DIR/patches":/patches:ro \
  "$LLAMA_SYCL_IMAGE" \
    -m "/models/$GGUF_NAME" \
    -ngl 999 -c 262144 --cache-type-k q4_0 --cache-type-v q4_0 -fa on \
    --jinja \
    --chat-template-file /patches/qwen3.8-claude.jinja \
    --alias "$LLAMA_ALIAS" \
    --spec-type draft-mtp --spec-draft-n-max "$LLAMA_SYCL_SPEC_N_MAX" \
    --host 0.0.0.0 --port "$LLAMA_PORT" >/dev/null

echo "Waiting for model load..."
for _ in $(seq 1 90); do
  sleep 5
  if docker logs "$LLAMA_SYCL_CONTAINER" 2>&1 | grep -q 'listening on'; then
    echo "Ready on http://127.0.0.1:${LLAMA_PORT}  (logs: docker logs -f $LLAMA_SYCL_CONTAINER)"
    exit 0
  fi
done
echo "Timed out; check: docker logs $LLAMA_SYCL_CONTAINER" >&2
exit 1
