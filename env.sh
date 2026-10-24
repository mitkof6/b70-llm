# Shared configuration for the Intel Arc Pro B70 + vLLM workflow.
# Source it:  source /path/to/llm/env.sh
# Relocatable: LLM_DIR is derived from this file's location.

LLM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LLM_DIR
export MODELS_DIR="${LLM_DIR}/models"

# ---------------------------------------------------------------------------
# Model selection: edit this ONE line.
#   "moe"   = Qwen3.8-35B-A3B Distill (GGUF ~22 GB, llama.cpp only)
#   "dense" = Qwen3.8-27B              (GPTQ ~19 GB for vLLM + GGUF ~17 GB)
# One-off override without editing the file:
#   LLM_MODEL_OVERRIDE=moe ./serve_sycl.sh
# The served OpenAI model id stays "qwen" either way, so client configs
# (opencode / Claude Code) never change.
# ---------------------------------------------------------------------------
LLM_MODEL="moe"
LLM_MODEL="${LLM_MODEL_OVERRIDE:-$LLM_MODEL}"
export LLM_MODEL

case "$LLM_MODEL" in
  dense)
    LLM_MODEL_LABEL="Qwen3.8-27B"
    # MTP-capable GPTQ checkpoint (its mtp.* draft head is already BF16 in-repo),
    # required for --speculative-config mtp. MTP4 measured here: 77.7 t/s median.
    MODEL_REPO="SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
    MODEL_NAME="Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
    MODEL_VLLM_READY=1
    LLAMA_GGUF="$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
    LLAMA_CHAT_TEMPLATE="${LLM_DIR}/patches/qwen3.8-claude.jinja"
    ;;
  moe)
    LLM_MODEL_LABEL="Qwen3.8-35B-A3B"
    # GGUF-only (Q4_K_M) from LM Studio; no safetensors/vLLM checkpoint here.
    MODEL_REPO=""
    MODEL_NAME="Qwen3.8-35B-A3B-Distill-GGUF"
    MODEL_VLLM_READY=0
    LLAMA_GGUF="$HOME/.lmstudio/models/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF/Qwen3.8-35B-A3B-Q4_K_M.gguf"
    LLAMA_CHAT_TEMPLATE="${LLM_DIR}/patches/qwen3.8-moe-claude.jinja"
    ;;
  *)
    echo "env.sh: LLM_MODEL must be 'moe' or 'dense' (got '$LLM_MODEL')" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

# Path override for one run (both models):
#   LLAMA_GGUF_OVERRIDE=/path/to/model.gguf ./serve_sycl.sh
LLAMA_GGUF="${LLAMA_GGUF_OVERRIDE:-$LLAMA_GGUF}"
LLAMA_CHAT_TEMPLATE="${LLAMA_CHAT_TEMPLATE_OVERRIDE:-$LLAMA_CHAT_TEMPLATE}"
export LLM_MODEL_LABEL MODEL_REPO MODEL_NAME MODEL_VLLM_READY
export MODEL_DIR="${MODELS_DIR}/${MODEL_NAME}"
export LLAMA_GGUF LLAMA_CHAT_TEMPLATE

# --- Unified client port ---
# All three backends serve the same port, so Claude Code / opencode configs
# never change when switching backend (only one backend runs at a time).
export LLM_PORT="${LLM_PORT:-8001}"

# --- vLLM container ---
# vllm/vllm-openai-xpu (vLLM 0.27.2rc1.dev77 + vllm-xpu-kernels 0.1.12.3),
# pinned to the digest benchmarked for Qwen3.8-27B + MTP on the B70.
export VLLM_IMAGE="vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
# Local archive of the image so it never needs re-pulling (docker load -i):
#   ${LLM_DIR}/images/vllm-openai-xpu-f01e24f6.tar
export VLLM_CONTAINER="vllm-b70"
export VLLM_PORT="${VLLM_PORT:-$LLM_PORT}"
export VLLM_SERVED_NAME="qwen"
# MTP draft depth. 4 is the benchmark recipe (~78 t/s) but crashes EngineCore
# when requests overlap (vllm-xpu-kernels#510). Use 0 for parallel clients,
# including Claude Code (it issues overlapping background requests).
export VLLM_MTP_N="${VLLM_MTP_N:-4}"

# --- llama.cpp (SYCL) shared settings ---
# llama.cpp and vLLM cannot share the 32 GB card; serve_sycl.sh stops vLLM first.
# $LLAMA_GGUF comes from the model profile above (override: LLAMA_GGUF_OVERRIDE=...).
export LLAMA_PORT="${LLAMA_PORT:-$LLM_PORT}"
# Both GGUFs advertise a 262144-token context. Lower it if a model is tight on VRAM.
export LLAMA_CTX="${LLAMA_CTX:-262144}"
# OpenAI model id the llama.cpp server advertises (set with --alias).
export LLAMA_ALIAS="${LLAMA_ALIAS:-qwen}"

# --- llama.cpp (SYCL / oneAPI) Docker image, fastest single-stream ---
# Bundles the oneAPI 2026 runtime; SYCL sees the B70 as SYCL0.
export LLAMA_SYCL_IMAGE="ghcr.io/ggml-org/llama.cpp:server-intel"
export LLAMA_SYCL_CONTAINER="llama-sycl"
# MTP draft depth: 2 measured best on SYCL (~45-50 t/s with SYCL graphs).
# Deeper drafts (3-4) lower MoE acceptance per draft and were slower here (40 t/s).
export LLAMA_SYCL_SPEC_N_MAX="${LLAMA_SYCL_SPEC_N_MAX:-2}"
# SYCL graph capture is off in the image by default. Measured on the B70 MoE,
# MTP n=2, 32k ctx: 26.7 -> 48.5 t/s decode (production full-ctx median ~45). Set 0 to A/B test.
export LLAMA_SYCL_ENABLE_GRAPH="${LLAMA_SYCL_ENABLE_GRAPH:-1}"
# Sparse FA is exact (it gathers every finite mask entry), not an approximation.
# It matters at full 262144 ctx: short-prompt decode 35 -> 48 t/s by compacting
# the 262k KV allocation down to the tokens actually in use. No effect on
# quality; long prompts fall back when the compacted set is > half the cache.
export LLAMA_SYCL_SPARSE_FA="${LLAMA_SYCL_SPARSE_FA:-1}"
# llama.cpp divides -c across slots: --parallel 4 turns the 262144-token
# allocation into 4 x 65536, so a single request cannot use the full context.
# Keep 1 for true 262144 single-stream context (~25.5 t/s here); set 4 for four
# 64k slots and ~43 t/s.
export LLAMA_SYCL_PARALLEL="${LLAMA_SYCL_PARALLEL:-1}"

# --- Intel Arc device ids (for docker --group-add) ---
export RENDER_GID="$(getent group render | cut -d: -f3)"
export VIDEO_GID="$(getent group video | cut -d: -f3)"

# --- Single-card invariant ---
# Only one backend fits the 32 GB card: stop every backend except $1
# (vllm | sycl) so the caller owns the GPU and the port. rm -f (not
# docker stop) on vLLM: its --stop-timeout 120 would stall a switch up to 2 min.
stop_other_backends() {
  local me="$1" c
  for c in "$VLLM_CONTAINER" "$LLAMA_SYCL_CONTAINER"; do
    [ "$c" = "$me" ] && continue
    if [ -n "$(docker ps -q --filter "name=^${c}$" 2>/dev/null)" ]; then
      echo "Stopping '$c' to free the B70..."
      docker rm -f "$c" >/dev/null 2>&1 || true
    fi
  done
}

# --- Hugging Face CLI (hf). Override with HF=/path/to/hf ---
if [ -z "${HF:-}" ]; then
  if [ -x "$HOME/.venvs/hf/bin/hf" ]; then HF="$HOME/.venvs/hf/bin/hf"
  else HF="$(command -v hf || true)"; fi
fi
export HF
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Xet high-performance mode: raises concurrency. On a flaky/throttled link the
# adaptive default can be steadier, so leave it OFF unless the link is fast.
# export HF_XET_HIGH_PERFORMANCE=1
