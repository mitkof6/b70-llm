#!/usr/bin/env bash
# EXPERIMENT: run the latest vLLM XPU (v0.32.0) on the Arc Pro B70.
# Same model and serve flags as serve_vllm.sh, but NO cookbook patches by
# default: we want to see what upstream v0.32.0 + vllm-xpu-kernels 0.1.15.4
# do on their own. Tuned MTP10 and prefix caching are on (~95 t/s realworld).
#
# Set VLLM_EXP_PATCHES=1 to apply the old cookbook patch set if something
# breaks (patch_mtp_nightly/boundary, APC fixes, and the v5 GDN split).
#
# Uses its own container name (vllm-b70-exp) and compile-cache volume, but
# shares the API port (8001) so opencode/Claude configs never change. It stops
# the pinned 0.27.2rc1 container first, so only one backend owns the card.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env_exp.sh" || exit 1

# vLLM needs the dense safetensors checkpoint; the MoE profile is GGUF-only.
if [ "${MODEL_VLLM_READY:-0}" != "1" ] || [ -z "${MODEL_REPO:-}" ]; then
  echo "$LLM_MODEL_LABEL has no vLLM/safetensors checkpoint configured." >&2
  echo "Use LLM_MODEL_OVERRIDE=dense ./serve_vllm_exp.sh" >&2
  exit 1
fi
[ -f "$LLAMA_CHAT_TEMPLATE" ] || { echo "Chat template not found: $LLAMA_CHAT_TEMPLATE" >&2; exit 1; }

ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || { echo "Model not downloaded: $MODEL_DIR"; exit 1; }

[ -n "$RENDER_GID" ] && [ -n "$VIDEO_GID" ] || {
  echo "Not in the 'render'/'video' groups (needed for /dev/dri in Docker)." >&2
  echo "Fix: sudo usermod -aG render,video \"$USER\", then log out and back in." >&2
  exit 1
}

# Fail early if the eGPU came back in Small BAR mode (see env_exp.sh).
b70_gpu_bar_check || exit 1

# Only one backend fits the 32 GB card: stop the pinned vLLM and SYCL first,
# then recreate the experimental container itself.
stop_other_backends "$VLLM_CONTAINER_EXP"
docker rm -f "$VLLM_CONTAINER_EXP" >/dev/null 2>&1 || true

# MTP10 is the tuned single-stream default (4 was the old recipe).
# VLLM_MTP_N_EXP=0 disables speculation. (Legacy env.sh exports VLLM_MTP_N=4,
# so the exp build uses its own knob.)
VLLM_SPEC_CONFIG=""
if [ "${VLLM_MTP_N_EXP:-10}" -gt 0 ] 2>/dev/null; then
  VLLM_SPEC_CONFIG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":${VLLM_MTP_N_EXP}}'"
fi

# Prefix caching on by default for this experiment (same as b70-vllm).
VLLM_PREFIX_FLAG="--enable-prefix-caching"
[ "${VLLM_PREFIX_CACHING:-1}" = "1" ] || VLLM_PREFIX_FLAG="--no-enable-prefix-caching"

# No patches unless explicitly requested: upstream v0.32.0 is the point here.
PATCH_CMDS=""
if [ "${VLLM_EXP_PATCHES:-0}" = "1" ]; then
  PATCH_CMDS="python /patch_mtp.py && python /patch_boundary.py"
  if [ "${VLLM_PREFIX_CACHING:-1}" = "1" ]; then
    PATCH_CMDS="$PATCH_CMDS && python /patch_fix_backward_copy.py && python /patch_fix_eagle_drop.py && python /patch_fix_accepted_sync.py"
  fi
  if [ "${VLLM_MIXED_SPLIT:-1}" = "1" ]; then
    PATCH_CMDS="$PATCH_CMDS && python /patch_gdn_mixed_split_v5.py"
  fi
fi

# Context/memory knobs: same defaults as serve_vllm.sh.
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-8192}"
# v0.32.0 profiles CUDA graphs by default, reserving ~1 GiB extra; 0.88
# leaves only ~4.94 GiB KV cache, too little for a 131072-token slot (5.23
# GiB needed). 0.94 also leaves room for MTP10 workspace/graphs.
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.94}"
export VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_EAGER_FLAG=""
[ "$VLLM_ENFORCE_EAGER" = "1" ] && VLLM_EAGER_FLAG="--enforce-eager"

docker run -d --name "$VLLM_CONTAINER_EXP" --stop-timeout 120 \
  --device /dev/dri:/dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
  --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
  --shm-size 2g \
  -v vllm-compile-cache-exp:/root/.cache/vllm \
  -v "$MODEL_DIR":/models/qwen:ro \
  -v "$DIR/patches/patch_mtp_nightly.py":/patch_mtp.py:ro \
  -v "$DIR/patches/patch_mtp_boundary.py":/patch_boundary.py:ro \
  -v "$DIR/patches/patch_fix_accepted_sync.py":/patch_fix_accepted_sync.py:ro \
  -v "$DIR/patches/patch_fix_backward_copy.py":/patch_fix_backward_copy.py:ro \
  -v "$DIR/patches/patch_fix_eagle_drop.py":/patch_fix_eagle_drop.py:ro \
  -v "$DIR/patches/patch_gdn_mixed_split_v5.py":/patch_gdn_mixed_split_v5.py:ro \
  -v "$LLAMA_CHAT_TEMPLATE":/chat-template.jinja:ro \
  -p 127.0.0.1:${VLLM_PORT_EXP}:${VLLM_PORT_EXP} \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e B70_MTP_BF16_DRAFT=1 \
  --entrypoint bash \
  "$VLLM_IMAGE_EXP" -lc \
  "set -e
    ${PATCH_CMDS:+$PATCH_CMDS && }
    exec vllm serve /models/qwen \
      --served-model-name $VLLM_SERVED_NAME_EXP \
      --host 0.0.0.0 --port $VLLM_PORT_EXP \
      --dtype $VLLM_DTYPE --kv-cache-dtype $VLLM_KV_CACHE_DTYPE \
      --max-model-len $VLLM_MAX_MODEL_LEN \
      --max-num-seqs $VLLM_MAX_NUM_SEQS \
      --max-num-batched-tokens $VLLM_MAX_BATCHED_TOKENS \
      --gpu-memory-utilization $VLLM_GPU_MEM_UTIL $VLLM_PREFIX_FLAG \
      --trust-remote-code --language-model-only \
      --chat-template /chat-template.jinja \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --quantization gptq \
      --default-chat-template-kwargs '{\"enable_thinking\": false}' \
      --generation-config vllm \
      --override-generation-config '{\"temperature\": 0.7, \"top_p\": 0.8, \"top_k\": 20}' \
      $VLLM_EAGER_FLAG $VLLM_EXTRA_ARGS $VLLM_SPEC_CONFIG"

echo "Started '$VLLM_CONTAINER_EXP' ($VLLM_IMAGE_EXP, $LLM_MODEL_LABEL)."
echo "Waiting for 'Application startup complete' (first compile can take ~15 min)."
echo "Follow with: docker logs -f $VLLM_CONTAINER_EXP"
for _ in $(seq 1 240); do
  sleep 5
  logs="$(docker logs "$VLLM_CONTAINER_EXP" 2>&1 || true)"
  if [[ "$logs" == *"Application startup complete"* ]]; then
    echo "Ready on http://127.0.0.1:${VLLM_PORT_EXP}"
    exit 0
  fi
  if [ -z "$(docker ps -q --filter "name=^${VLLM_CONTAINER_EXP}$" 2>/dev/null)" ]; then
    echo "Container '$VLLM_CONTAINER_EXP' exited; last log lines:" >&2
    docker logs --tail 60 "$VLLM_CONTAINER_EXP" 2>&1 >&2 || true
    exit 1
  fi
done
echo "Timed out after 20 min; check: docker logs -f $VLLM_CONTAINER_EXP" >&2
exit 1
