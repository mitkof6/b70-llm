# Shared configuration for the Intel Arc Pro B70 + vLLM workflow.
# Source it:  source /path/to/llm/env.sh
# Relocatable: LLM_DIR is derived from this file's location.

LLM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LLM_DIR
export MODELS_DIR="${LLM_DIR}/models"

# --- Model: vLLM-ready INT4 checkpoint tuned for Intel Arc Pro B70 ---
# MTP-capable GPTQ checkpoint (its mtp.* draft head is already BF16 in-repo),
# required for --speculative-config mtp. MTP4 measured on this B70: 77.7 t/s
# median (94.3% acceptance; the cookbook's cell on theirs: 83.7).
export MODEL_REPO="SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
export MODEL_NAME="Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
export MODEL_DIR="${MODELS_DIR}/${MODEL_NAME}"

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

# --- llama.cpp (LM Studio's bundled Vulkan runtime), GGUF path ---
# llama.cpp and vLLM cannot share the 32 GB card; serve_vulkan.sh stops vLLM first.
# Newest bundled llama.cpp Vulkan runtime from LM Studio (override: LLAMA_SERVER=...).
export LLAMA_SERVER="${LLAMA_SERVER:-$(ls -1 "$HOME"/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-vulkan-avx2-*/llama-server 2>/dev/null | sort -V | tail -1)}"
export LLAMA_GGUF="${LLAMA_GGUF:-$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf}"
export LLAMA_PORT="${LLAMA_PORT:-$LLM_PORT}"
# Vulkan device index of the B70 (0=iGPU UHD P630, 1=Quadro P2000, 2=BMG G31).
export GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-2}"
# OpenAI model id both llama.cpp servers advertise (set with --alias).
export LLAMA_ALIAS="${LLAMA_ALIAS:-qwen}"

# --- llama.cpp (SYCL / oneAPI) Docker image, fastest single-stream ---
# Bundles the oneAPI 2026 runtime; SYCL sees the B70 as SYCL0.
export LLAMA_SYCL_IMAGE="ghcr.io/ggml-org/llama.cpp:server-intel"
export LLAMA_SYCL_CONTAINER="llama-sycl"
# MTP draft depth: 2 measured best on SYCL (~34 t/s), 1 best on Vulkan (~16 t/s).
export LLAMA_SYCL_SPEC_N_MAX="${LLAMA_SYCL_SPEC_N_MAX:-2}"

# --- Intel Arc device ids (for docker --group-add) ---
export RENDER_GID="$(getent group render | cut -d: -f3)"
export VIDEO_GID="$(getent group video | cut -d: -f3)"

# --- Single-card invariant ---
# Only one backend fits the 32 GB card: stop every backend except $1
# (vllm | sycl | vulkan) so the caller owns the GPU and the port. rm -f (not
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
  pkill -x llama-server 2>/dev/null || true
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
