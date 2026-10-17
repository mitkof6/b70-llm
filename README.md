# Qwen3.8-27B on an Intel Arc Pro B70

Run and benchmark `Qwen3.8-27B` locally on an **Intel Arc Pro B70** (32 GB VRAM).
Three interchangeable backends are available, and **only one fits on the card at
a time** — each `serve_*` script stops the others first.

| Backend                          | Port | Use it for                       |
| -------------------------------- | ---- | -------------------------------- |
| llama.cpp **SYCL** (Docker)      | 8080 | Fastest llama.cpp (**~34 t/s**)     |
| llama.cpp **Vulkan** (LM Studio) | 8080 | Fast llama.cpp, no extra image      |
| **vLLM XPU** (Docker)            | 8001 | Fastest overall (**~78 t/s** with MTP4); concurrency needs MTP off |

## Hardware

- GPU: Intel Arc Pro B70 `BMG-G31` (`8086:e223`), 32 GB VRAM, `xe` driver, ReBAR on.
- The iGPU (UHD P630) and NVIDIA Quadro P2000 are not used.
- SYCL/Docker needs `--device /dev/dri` (the scripts handle group ids).

## Quick start

```bash
source ~/.bashrc            # loads aliases (b70-*) and env.sh

b70-sycl                    # fastest llama.cpp: SYCL + MTP  -> :8080
#   or
b70-vllm                    #    vLLM XPU (concurrent)         -> :8001
#   or
b70-vulkan                  #    llama.cpp Vulkan + MTP        -> :8080

b70-vulkan-bench 256 1      # benchmark whatever llama.cpp server is up
b70-vllm-bench 512 3        # benchmark vLLM (max_tokens, concurrency)
```

Stop / watch:

```bash
b70-sycl-stop   ; b70-sycl-logs      # llama.cpp SYCL (docker 'llama-sycl')
b70-vulkan-stop ; b70-vulkan-logs    # llama.cpp Vulkan (pkill llama-server)
b70-vllm-stop   ; b70-vllm-logs      # vLLM (docker 'vllm-b70')
b70-what                             # show which backend is currently live

# what the log lines mean: see LOGS.md
```

(Without aliases, run the scripts directly: `./serve_sycl.sh`, etc.)

## Requirements & migration

The folder is **relocatable** — scripts derive their paths from their own
location (`env.sh`, `bashrc.llm`), and everything they need is inside it
(patches, benchmark prompts, opencode config archive, vLLM image tar). Models
(~19 GB) are re-downloadable.

**On a new PC you need:**

- Linux with an Intel **Arc Pro B70** (`xe` kernel driver, `/dev/dri` present,
  Resizable BAR enabled in BIOS).
- **Docker** (your user able to run `docker` without sudo).
- Your user in the **`render`** and **`video`** groups (their gids are passed to
  the containers).
- **Python 3** (benchmarks use the standard library only).
- **`hf` CLI** — `pipx install "huggingface_hub[cli]"` (a venv at
  `~/.venvs/hf/bin/hf` is found too) — used by `download_model.sh`.
- **LM Studio** — only for the llama.cpp **Vulkan** backend (it provides
  `llama-server`) and the shared `Qwen3.8-27B-Q4_K_M.gguf` in
  `~/.lmstudio/models/…`; SYCL uses the same GGUF (override `LLAMA_GGUF`).
- **opencode** — optional, for the OpenAI-compatible providers.

**Migrate:**

1. Copy the folder, e.g. `rsync -a --exclude models/ ./ newpc:~/llm/`.
   (`llama-sycl/llama-b11700/` is an extracted standalone SYCL build that does
   not run — safe to exclude.)
2. Load the pinned vLLM image: `docker load -i images/vllm-openai-xpu-f01e24f6.tar`
   (otherwise Docker pulls it automatically; `env.sh` pins it by digest).
3. Add the aliases: append `source ~/llm/bashrc.llm` to `~/.bashrc`.
4. opencode: `cp ~/llm/config/opencode.global.jsonc ~/.config/opencode/opencode.jsonc`
   (it contains your OpenRouter API key — keep it private).
5. Fetch the model: `./download_model.sh` (~19 GB, resumable).
6. Start: `b70-vllm` (first compile ~4 min, later starts ~2 min) or `b70-sycl`
   (the SYCL image is pulled automatically from ghcr).
7. Verify: `b70-what`, then `./bench_vllm.sh 128 1`.

Machine-specific values can be overridden via environment variables:
`LLAMA_SERVER`, `LLAMA_GGUF`, `GGML_VK_VISIBLE_DEVICES` (the Vulkan device index
depends on PCI order — check with `llama-server --list-devices`), `HF`,
`VLLM_PORT`, `LLAMA_PORT`.

## Measured performance

Decode speed, single stream unless noted, on this B70 + Qwen3.8-27B:

| Backend / config                                | decode tok/s |
| ----------------------------------------------- | ------------ |
| **llama.cpp SYCL + MTP** `--spec-draft-n-max 2` | **34.0**     |
| llama.cpp SYCL + MTP `n=1`                      | 33.0         |
| llama.cpp SYCL, no MTP                          | 25.6         |
| llama.cpp Vulkan + MTP `--spec-draft-n-max 1`   | 16.0         |
| llama.cpp Vulkan + MTP `n=2`                    | 15.1         |
| llama.cpp Vulkan, no MTP                        | 11.7         |
| **vLLM XPU + MTP4**, C1 (patched, cache off)    | **77.7**     |
| vLLM XPU, no-spec, single stream                | 30.4         |
| vLLM XPU, no-spec, concurrency 3 (aggregate)    | 30.6         |

**Takeaways**

- Serial / single-user (e.g. an editor assistant) → **vLLM XPU + MTP4** (patched):
  ~78 tok/s vs llama.cpp SYCL ~34. llama.cpp stays useful as a no-patch-stack,
  no-extra-container fallback.
- Many parallel requests → **vLLM** (llama.cpp has no continuous batching here).
  MTP4 is single-stream only on this stack (see Gotchas).
- **MTP** (speculative decoding) is the single biggest win: vLLM 30 → 78 tok/s
  with MTP4; llama.cpp best depth is `n=1` on Vulkan / `n=2` on SYCL.
- vLLM C1 number: p512/g128 greedy, cache off, cookbook realworld harness, n=5
  median **77.7** (75.3–79.9), MTP acceptance **94.3%**. Cookbook's cell on their
  B70: 83.7 / 95.0% (230 W cap; ours reads 275 W) — ~93% reproduction. Reproduce:

  ```bash
  python3 benchmarks/b70-realworld-context-harness.py --mode context \
    --prompts benchmarks/prompts/qwen38-p512.json --target 512 --output 128 \
    --budget 8192 --reps 5 --model qwen --root http://127.0.0.1:8001 \
    --full-output-warmup --ignore-eos --outdir /tmp/harness-$(date +%s)
  ```

- Bench prompts matter hugely: one repeated filler sentence makes the model write
  meta-commentary and collapses MTP acceptance to ~50% (~49 t/s). Always bench
  with realistic prompts (the `benchmarks/` harness).

## Files

| File                  | Purpose                                                       |
| --------------------- | ------------------------------------------------------------- |
| `env.sh`              | Shared paths, image tags, device ids. Sourced by all scripts.  |
| `serve_sycl.sh`       | Start llama.cpp SYCL (Docker + MTP) on :8080.                   |
| `status.sh`           | Show which backend is currently serving.                        |
| `serve_vulkan.sh`     | Start llama.cpp Vulkan (LM Studio runtime + MTP) on :8080.      |
| `serve_vllm.sh`       | Start the vLLM XPU container on :8001.                          |
| `bench_vulkan.sh`     | Benchmark a llama.cpp server: `./bench_vulkan.sh [max_tokens] [conc]`. |
| `bench_vllm.sh`       | Benchmark vLLM: `./bench_vllm.sh [max_tokens] [concurrency]`.   |
| `bench_vllm_stream.sh` | Article-style single-stream bench (post-first-token).          |
| `download_model.sh`   | Resumable/retrying HF download of the INT4 vLLM checkpoint.     |
| `benchmarks/`         | Cookbook realworld harness, prompt generator, p512 prompts.     |
| `patches/`            | Cookbook MTP patches (BF16-draft gate + 131072 boundary).       |
| `images/`             | Docker-save archive of the pinned vLLM image (4.65 GB).         |
| `bashrc.llm`          | Aliases + env; sourced from `~/.bashrc`.                        |
| `config/`             | Archive of the opencode configs (global + former project-local). |
| `AGENTS.md`           | Notes for coding agents (this project's `instructions`).        |
| `LOGS.md`             | Line-by-line log field guide (vLLM + llama.cpp).                |

## Models

- **vLLM**: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (GPTQ INT4 g128
  symmetric, BF16 MTP heads, ~19 GB), fetched by `download_model.sh` into
  `models/`; needs the patched launch in `serve_vllm.sh`.
- **llama.cpp**: `Qwen3.8-27B-Q4_K_M.gguf` (+ `nextn` MTP head baked in) at
  `~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/`. Both Vulkan and SYCL
  use this GGUF.

## Use from opencode

The three servers are registered as OpenAI-compatible providers in the global
opencode config (`~/.config/opencode/opencode.jsonc`; archived copy in
`config/`):

- `vllm/qwen` → `http://127.0.0.1:8001/v1`
- `sycl/qwen` → `http://127.0.0.1:8080/v1`
- `vulkan/qwen` → `http://127.0.0.1:8080/v1`
  (SYCL and Vulkan share :8080; the running llama.cpp backend answers both)
- every model offers `off` / `low` / `medium` / `xhigh` thinking variants
  (`enable_thinking` + `reasoning_effort` template kwargs)

Start the backend you want, then run `opencode` from any folder. Config is not
hot-reloaded — restart opencode after changing the config. Switch model and
variant with the picker in the TUI; the one you pick answers only if its server
is running.

## Gotchas

- Only one backend fits the 32 GB card; every `serve_*` script stops the others.
- llama.cpp **Vulkan** must be pinned to the B70 with `GGML_VK_VISIBLE_DEVICES=2`
  (index 0 = Quadro P2000 → `device lost`); `serve_vulkan.sh` sets it. Do **not**
  set `VK_ICD_FILENAMES`.
- Stop llama.cpp with `pkill -x llama-server` (a `-f` match also kills the shell).
- The standalone llama.cpp SYCL tarball does **not** run here — use the
  `ghcr.io/ggml-org/llama.cpp:server-intel` image, which bundles the oneAPI
  runtime.
- The eduroam link is throttled; the vLLM image / 4.3 GB SYCL image take a while
  to pull, and `hf` downloads may take hours (they resume).
- MTP4 is **single-stream only** on this stack: overlapping requests can kill
  EngineCore (vllm-xpu-kernels#510). `serve_vllm.sh` mirrors the cookbook's bench
  config (`--no-enable-prefix-caching`); interactive serving wants cache on + the
  APC patches.
