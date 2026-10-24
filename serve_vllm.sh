#!/usr/bin/env bash
# Launch vLLM XPU (vllm/vllm-openai-xpu, vLLM 0.27.2rc1) serving the MTP-capable
# INT4 checkpoint on the Arc Pro B70 - the cookbook's validated MTP4 recipe for
# this model: patch_mtp_nightly.py (BF16 draft gate, B70_MTP_BF16_DRAFT=1) +
# patch_mtp_boundary.py, text-only, prefix cache off (their cell: 83.7 t/s).
# MTP is single-stream only; concurrent requests can crash the XPU engine (#510).
#
# Prefix caching is opt-in: VLLM_PREFIX_CACHING=1. Hybrid GDN + APC forces
# mamba_cache_mode="align", which needs the cookbook's three correctness fixes
# (patch_fix_accepted_sync/backward_copy/eagle_drop.py); they are applied only
# when caching is on so the cache-off MTP4 path stays untouched.
#
# MTP + overlapping requests needs the v5 mixed-batch split: VLLM_MIXED_SPLIT=1
# (vllm-xpu-kernels#510). b70-vllm-claude-mtp sets it with MTP4 + cache.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh" || exit 1

# vLLM needs a safetensors checkpoint with vLLM kernels; the MoE profile is
# GGUF-only. Fail before stopping a working backend.
if [ "${MODEL_VLLM_READY:-0}" != "1" ] || [ -z "${MODEL_REPO:-}" ]; then
  echo "$LLM_MODEL_LABEL has no vLLM/safetensors checkpoint configured (GGUF-only)." >&2
  echo "Use ./serve_sycl.sh for it," >&2
  echo "or set LLM_MODEL=\"dense\" in env.sh (or LLM_MODEL_OVERRIDE=dense ./serve_vllm.sh)." >&2
  exit 1
fi
[ -f "$LLAMA_CHAT_TEMPLATE" ] || { echo "Chat template not found: $LLAMA_CHAT_TEMPLATE" >&2; exit 1; }

ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || { echo "Model not downloaded: $MODEL_DIR"; exit 1; }

# Docker needs the user in render/video for /dev/dri; fail before stopping
# anything, since a working backend must not be killed over a config error.
[ -n "$RENDER_GID" ] && [ -n "$VIDEO_GID" ] || {
  echo "Not in the 'render'/'video' groups (needed for /dev/dri in Docker)." >&2
  echo "Fix: sudo usermod -aG render,video \"$USER\", then log out and back in." >&2
  exit 1
}

# Only one backend fits the 32 GB card: free it from the others first.
stop_other_backends vllm
docker rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 || true

# MTP4 is single-stream only on this stack: overlapping requests can kill
# EngineCore (vllm-xpu-kernels#510). VLLM_MTP_N=0 disables speculation.
VLLM_SPEC_CONFIG=""
if [ "${VLLM_MTP_N:-4}" -gt 0 ] 2>/dev/null; then
  VLLM_SPEC_CONFIG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":${VLLM_MTP_N}}'"
fi

# Prefix caching: off by default (cookbook bench cell). On for interactive
# serving, where it skips the repeated full-context prefill (~8-20 s/turn).
# MTP + overlap needs the v5 mixed-batch split (#510).
VLLM_PREFIX_FLAG="--no-enable-prefix-caching"
APC_PATCH_CMDS=""
if [ "${VLLM_PREFIX_CACHING:-0}" = "1" ]; then
  VLLM_PREFIX_FLAG="--enable-prefix-caching"
  APC_PATCH_CMDS="python /patch_fix_backward_copy.py && python /patch_fix_eagle_drop.py && python /patch_fix_accepted_sync.py"
fi

MIXED_SPLIT_CMD=""
if [ "${VLLM_MIXED_SPLIT:-0}" = "1" ]; then
  MIXED_SPLIT_CMD="python /patch_gdn_mixed_split_v5.py"
  if [ -z "$VLLM_SPEC_CONFIG" ]; then
    echo "WARNING: VLLM_MIXED_SPLIT=1 without MTP does nothing useful." >&2
  fi
elif [ -n "$VLLM_SPEC_CONFIG" ] && [ "${VLLM_PREFIX_CACHING:-0}" = "1" ]; then
  echo "WARNING: MTP + prefix caching without VLLM_MIXED_SPLIT=1: overlapping" >&2
  echo "         requests can still kill EngineCore (vllm-xpu-kernels#510)." >&2
fi

# Context/memory knobs. Defaults are the validated single-stream recipe. For
# the full 262144-token context use b70-vllm-fullctx (tested 2026-10-24):
#   VLLM_MAX_MODEL_LEN=262144 VLLM_MAX_NUM_SEQS=1 VLLM_MTP_N=0
#   VLLM_GPU_MEM_UTIL=0.95 VLLM_MAX_BATCHED_TOKENS=4096
# That reports 9.22 GiB KV / 295,108 tokens (1.13x at 262144) and accepts a
# 158k-token prompt. Dense 27B is hybrid GDN: only 16 of 64 layers keep KV,
# 4 KV heads, head_dim 256, so fp8 KV is 32 KiB/token (8 GiB at 262144).
# MTP4 at 262144 does not fit at 0.95 (needs ~9.33 GiB KV plus draft overhead);
# use MTP off for full context, or 196k/MTP4, or SYCL MoE at 262144.
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-8192}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.88}"
export VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_EAGER_FLAG=""
[ "$VLLM_ENFORCE_EAGER" = "1" ] && VLLM_EAGER_FLAG="--enforce-eager"

docker run -d --name "$VLLM_CONTAINER" --stop-timeout 120 \
  --device /dev/dri:/dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
  --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
  --shm-size 2g \
  -v vllm-compile-cache:/root/.cache/vllm \
  -v "$MODEL_DIR":/models/qwen:ro \
  -v "$DIR/patches/patch_mtp_nightly.py":/patch_mtp.py:ro \
  -v "$DIR/patches/patch_mtp_boundary.py":/patch_boundary.py:ro \
  -v "$DIR/patches/patch_fix_accepted_sync.py":/patch_fix_accepted_sync.py:ro \
  -v "$DIR/patches/patch_fix_backward_copy.py":/patch_fix_backward_copy.py:ro \
  -v "$DIR/patches/patch_fix_eagle_drop.py":/patch_fix_eagle_drop.py:ro \
  -v "$DIR/patches/patch_gdn_mixed_split_v5.py":/patch_gdn_mixed_split_v5.py:ro \
  -v "$LLAMA_CHAT_TEMPLATE":/chat-template.jinja:ro \
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
   $MIXED_SPLIT_CMD
   $APC_PATCH_CMDS
   exec vllm serve /models/qwen \
     --served-model-name $VLLM_SERVED_NAME \
     --host 0.0.0.0 --port $VLLM_PORT \
     --dtype float16 --kv-cache-dtype $VLLM_KV_CACHE_DTYPE \
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
# MTP4 recipe (cookbook; model card cell: 83.7 t/s): the draft runs BF16 only with
# B70_MTP_BF16_DRAFT=1 + patch_mtp_nightly.py; without the gate the draft builds
# quantized and acceptance collapses (~50% -> 44 t/s, measured 2026-10-17).
# Cache is off by default (their bench cell); VLLM_PREFIX_CACHING=1 turns it on
# for interactive use. Lower num_speculative_tokens for a safer/slower mix.

echo "Started container '$VLLM_CONTAINER' ($LLM_MODEL_LABEL). Waiting for 'Application startup complete'"
echo "(first start compiles; allow up to ~15 min). Follow with: docker logs -f $VLLM_CONTAINER"
for _ in $(seq 1 180); do
  sleep 5
  logs="$(docker logs "$VLLM_CONTAINER" 2>&1 || true)"
  if [[ "$logs" == *"Application startup complete"* ]]; then
    echo "Ready on http://127.0.0.1:${VLLM_PORT}"
    exit 0
  fi
  if [ -z "$(docker ps -q --filter "name=^${VLLM_CONTAINER}$" 2>/dev/null)" ]; then
    echo "Container '$VLLM_CONTAINER' exited; last log lines:" >&2
    docker logs --tail 40 "$VLLM_CONTAINER" 2>&1 >&2 || true
    exit 1
  fi
done
echo "Timed out after 15 min; check: docker logs -f $VLLM_CONTAINER" >&2
exit 1
