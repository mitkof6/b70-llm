#!/usr/bin/env bash
# Launch vLLM XPU (vllm/vllm-openai-xpu, vLLM 0.27.2rc1) serving the MTP-capable
# INT4 checkpoint on the Arc Pro B70 - the cookbook's validated MTP4 recipe for
# this model: patch_mtp_nightly.py (BF16 draft gate, B70_MTP_BF16_DRAFT=1) +
# patch_mtp_boundary.py, text-only, prefix cache off (their cell: 83.7 t/s).
# MTP is single-stream only; concurrent requests can crash the XPU engine (#510).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh"

ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || { echo "Model not downloaded: $MODEL_DIR"; exit 1; }

docker rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 || true

docker run -d --name "$VLLM_CONTAINER" --stop-timeout 120 \
  --device /dev/dri:/dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
  --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
  --shm-size 2g \
  -v vllm-compile-cache:/root/.cache/vllm \
  -v "$MODEL_DIR":/models/qwen:ro \
  -v "$DIR/patches/patch_mtp_nightly.py":/patch_mtp.py:ro \
  -v "$DIR/patches/patch_mtp_boundary.py":/patch_boundary.py:ro \
  -p 127.0.0.1:${VLLM_PORT}:${VLLM_PORT} \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e B70_MTP_BF16_DRAFT=1 \
  --entrypoint bash \
  "$VLLM_IMAGE" -lc \
  "set -e
   python /patch_mtp.py
   python /patch_boundary.py
   exec vllm serve /models/qwen \
     --served-model-name $VLLM_SERVED_NAME \
     --host 0.0.0.0 --port $VLLM_PORT \
     --dtype float16 --kv-cache-dtype fp8 --max-model-len 131072 \
     --max-num-seqs 64 --max-num-batched-tokens 8192 \
     --gpu-memory-utilization 0.88 --no-enable-prefix-caching \
     --trust-remote-code --language-model-only \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --reasoning-parser qwen3 \
     --quantization gptq \
     --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}'"
# MTP4 recipe (cookbook; model card cell: 83.7 t/s): the draft runs BF16 only with
# B70_MTP_BF16_DRAFT=1 + patch_mtp_nightly.py; without the gate the draft builds
# quantized and acceptance collapses (~50% -> 44 t/s, measured 2026-10-17).
# --no-enable-prefix-caching matches their cell; swap to --enable-prefix-caching
# for interactive use. Lower num_speculative_tokens for a safer/slower mix.

echo "Started container '$VLLM_CONTAINER'. Follow startup with:  docker logs -f $VLLM_CONTAINER"
echo "Ready when you see 'Application startup complete' on http://127.0.0.1:${VLLM_PORT}"
