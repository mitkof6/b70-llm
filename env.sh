# Shared configuration for the Intel Arc Pro B70 + vLLM XPU workflow.
# Source it:  source /path/to/b70-llm/env.sh
# Relocatable: LLM_DIR is derived from this file's location.

LLM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LLM_DIR
export MODELS_DIR="${LLM_DIR}/models"

# Migration: the retired env.sh exported LLM_MODEL (moe/dense) and MTP4.
# Drop that stale pair so the new MTP10 default applies; a clean-shell
# VLLM_MTP_N=4 override still works.
if [ -n "${LLM_MODEL:-}" ]; then
  [ "${VLLM_MTP_N:-}" = "4" ] && unset VLLM_MTP_N
  unset LLM_MODEL
fi

# --- Model ---------------------------------------------------------------
# Qwen3.8-27B, GPTQ INT4 sym G128 with a BF16 MTP draft head. The model is
# fixed: vLLM is the only supported backend.
LLM_MODEL_LABEL="Qwen3.8-27B"
MODEL_REPO="SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
MODEL_NAME="Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
MODEL_DIR="${MODELS_DIR}/${MODEL_NAME}"
CHAT_TEMPLATE="${LLM_DIR}/patches/qwen3.8-claude.jinja"
export LLM_MODEL_LABEL MODEL_REPO MODEL_NAME MODEL_DIR CHAT_TEMPLATE

# --- API -----------------------------------------------------------------
# One OpenAI-compatible port and one served model id for opencode/Claude.
export VLLM_PORT="${VLLM_PORT:-8001}"
export VLLM_SERVED_NAME="qwen"

# --- vLLM XPU ------------------------------------------------------------
# Upstream v0.32.0 + vllm-xpu-kernels 0.1.15.4 (torch 2.14 XPU). No patches.
export VLLM_IMAGE="vllm/vllm-openai-xpu:v0.32.0"
export VLLM_CONTAINER="vllm-b70"
# MTP draft depth: 10 is the tuned single-stream default (~95 t/s realworld
# p512). Set 0 to disable speculation.
export VLLM_MTP_N="${VLLM_MTP_N:-10}"
# Model dtype. float16 is the tested default.
export VLLM_DTYPE="${VLLM_DTYPE:-float16}"

# --- Intel Arc device ids (for docker --group-add) -----------------------
export RENDER_GID="$(getent group render | cut -d: -f3)"
export VIDEO_GID="$(getent group video | cut -d: -f3)"

# --- B70 eGPU Resizable BAR preflight ------------------------------------
# The B70 is a Thunderbolt eGPU ("Razer Core"). If the TB link re-enumerates
# (hotplug, GPU reset, cable glitch) Linux cannot grow the bridge windows to
# reassign the 32 GiB BAR, so xe falls back to a 256 MiB "Small BAR device".
# Level Zero then reports zero XPU devices and vLLM dies with the confusing
# "Failed to infer device type". Detect that state up front instead.
b70_gpu_bar_check() {
  local dev="${B70_PCI_DEV:-0000:0b:00.0}" s e size mib
  [ -r "/sys/bus/pci/devices/$dev/resource" ] || return 0
  read -r s e _ < <(sed -n '3p' "/sys/bus/pci/devices/$dev/resource") || return 0
  [ -n "${s:-}" ] && [ -n "${e:-}" ] || return 0
  size=$(( e - s + 1 ))
  if [ "$s" = "0x0000000000000000" ] || [ "$size" -lt $((1 << 30)) ]; then
    mib=$(( size >> 20 ))
    echo "B70 eGPU is in Small BAR mode (BAR2 = ${mib} MiB, expected 32768 MiB)." >&2
    echo "Level Zero cannot enumerate it in this state (torch sees 0 XPUs)." >&2
    echo "This happens when the Thunderbolt eGPU re-enumerates without a reboot;" >&2
    echo "Linux cannot reallocate the 32 GiB BAR on hotplug." >&2
    echo "Recovery: reboot with the eGPU connected (or cold-replug its TB cable)." >&2
    return 1
  fi
}

# --- Hugging Face CLI (hf). Override with HF=/path/to/hf ------------------
if [ -z "${HF:-}" ]; then
  if [ -x "$HOME/.venvs/hf/bin/hf" ]; then HF="$HOME/.venvs/hf/bin/hf"
  else HF="$(command -v hf || true)"; fi
fi
export HF
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Xet high-performance mode: raises concurrency. On a flaky/throttled link the
# adaptive default can be steadier, so leave it OFF unless the link is fast.
# export HF_XET_HIGH_PERFORMANCE=1
