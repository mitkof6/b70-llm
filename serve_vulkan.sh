#!/usr/bin/env bash
# Launch llama.cpp (LM Studio's bundled Vulkan runtime) on the Arc Pro B70 with
# MTP speculative decoding. Stops the vLLM container first — vLLM and llama.cpp
# cannot share the 32 GB card.
#
# Measured (B70, Qwen3.8-27B Q4_K_M): ~16 t/s decode with --spec-draft-n-max 1
# (73% draft acceptance); ~11.7 t/s without MTP; vLLM XPU + MTP4 is ~78 t/s.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh"

[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found: $LLAMA_SERVER" >&2; exit 1; }
[ -f "$LLAMA_GGUF" ] || { echo "GGUF not found: $LLAMA_GGUF" >&2; exit 1; }

# Free the GPU (llama.cpp needs the full card).
if [ -n "$(docker ps -q --filter "name=^${VLLM_CONTAINER}$" 2>/dev/null)" ]; then
  echo "Stopping vLLM container '$VLLM_CONTAINER' to free the B70..."
  docker stop "$VLLM_CONTAINER" >/dev/null
fi
pkill -x llama-server 2>/dev/null || true
sleep 2

LOG="${LLM_DIR}/llama-server.log"
echo "Starting llama-server (Vulkan, MTP n=1) -> $LOG"
setsid env GGML_VK_VISIBLE_DEVICES="$GGML_VK_VISIBLE_DEVICES" "$LLAMA_SERVER" \
  -m "$LLAMA_GGUF" \
  -ngl 999 -c 262144 --cache-type-k q4_0 --cache-type-v q4_0 -fa on \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  --alias "$LLAMA_ALIAS" \
  --host 127.0.0.1 --port "$LLAMA_PORT" \
  > "$LOG" 2>&1 < /dev/null &
disown

echo "Waiting for model load (B70 pinned via GGML_VK_VISIBLE_DEVICES=$GGML_VK_VISIBLE_DEVICES)..."
for _ in $(seq 1 60); do
  sleep 5
  if grep -q 'listening on' "$LOG"; then
    echo "Ready on http://127.0.0.1:${LLAMA_PORT}  (follow: tail -f $LOG)"
    exit 0
  fi
done
echo "Timed out; check $LOG" >&2
exit 1
