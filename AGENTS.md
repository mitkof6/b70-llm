# Intel Arc Pro B70 inference workflow

Serving `Qwen3.8-27B` (INT4 / Q4) on an **Intel Arc Pro B70** and measuring
throughput. Three interchangeable backends, which cannot share the 32 GB card:

- **llama.cpp SYCL** (ggml `server-intel` Docker image) — fastest llama.cpp.
- **llama.cpp Vulkan** (LM Studio's bundled runtime) — fast, no extra image.
- **vLLM XPU** (`vllm/vllm-openai-xpu` image + cookbook MTP patches) — fastest
  (~78 t/s with MTP4, single-stream; MTP off for concurrency).

## Hardware

- GPU: Intel Arc Pro B70 `BMG-G31` (`8086:e223`), 32 GB VRAM, `xe` driver.
- `torch.xpu` in the container reports exactly one device (`xpu:0`) = the B70;
  the iGPU (UHD P630) and NVIDIA Quadro P2000 are not level-zero targets.
- Note: llama.cpp runs on the B70 via either **Vulkan** (LM Studio runtime) or
  **SYCL/level-zero** (ggml `server-intel` image); vLLM uses the Intel XPU
  (SYCL / level-zero) backend, not Vulkan.

## Files

| File                 | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| `env.sh`             | Shared paths, image tag, device ids. Source this.      |
| `download_model.sh`  | Resumable/retrying HF download of the INT4 checkpoint. |
| `serve_vllm.sh`      | Start the vLLM server container on the B70.            |
| `bench_vllm.sh`      | vLLM tok/s benchmark (per-request + aggregate).        |
| `bench_vllm_stream.sh` | Article-style single-stream bench (post-first-token). |
| `benchmarks/`        | Cookbook realworld harness, prompt generator, p512 prompts. |
| `patches/`           | Cookbook MTP patches + Claude Code chat template.      |
| `images/`            | `docker save` archive of the pinned vLLM image (4.65 GB). |
| `serve_vulkan.sh`    | Start llama.cpp (Vulkan + MTP) on the B70; stops vLLM. |
| `bench_vulkan.sh`    | llama.cpp tok/s benchmark (per-request + aggregate).   |
| `serve_sycl.sh`      | Start llama.cpp SYCL (Docker + MTP); stops vLLM.       |
| `status.sh`          | Show which backend is currently serving.               |
| `README.md`          | Overview + all start commands.                         |
| `LOGS.md`            | Log line-by-line field guide.                          |
| `bashrc.llm`         | Shell aliases/env; sourced from `~/.bashrc`.           |
| `config/`            | Archive of the opencode + Claude Code configs.         |

## Workflow

```bash
source ./env.sh          # or open a new shell (aliases/aliases load from bashrc)
./download_model.sh      # resumable; ~19 GB into models/ (MTP GPTQ checkpoint)
./serve_vllm.sh          # container 'vllm-b70': MTP4 + cookbook patches; waits
                         # for 'Application startup complete' on :8001
docker logs -f vllm-b70  # first start in a mode ~8-12 min; warm ~3.5-6 min
./bench_vllm.sh 512 1    # single-stream tok/s
./bench_vllm.sh 512 3    # concurrency 3
./bench_vllm_stream.sh 128 5 off greedy   # article-style single-stream bench
```

Reference decode benchmark (realworld prompts; see README §Measured performance
for the full command): `benchmarks/b70-realworld-context-harness.py`.

llama.cpp (faster single-stream; stops vLLM because only one fits the card):

```bash
./serve_sycl.sh           # SYCL + MTP n=2 (fastest) on http://127.0.0.1:8001
./serve_vulkan.sh         # Vulkan + MTP n=1 on http://127.0.0.1:8001
./bench_vulkan.sh 256 1   # single-stream decode tok/s
```

Aliases (after `source ~/.bashrc`): `b70-vllm-download`, `b70-vllm`,
`b70-vllm-claude` (vLLM with MTP off), `b70-vllm-bench`, `b70-vllm-logs`,
`b70-vllm-stop`, `b70-sycl`, `b70-sycl-logs`, `b70-sycl-stop`, `b70-vulkan`,
`b70-vulkan-bench`, `b70-vulkan-logs`, `b70-vulkan-stop`, `b70-what`.

## Backends: measured on the B70

| Backend / config                              | decode tok/s |
| --------------------------------------------- | ------------ |
| llama.cpp SYCL + MTP `--spec-draft-n-max 2`   | **34.0**     |
| llama.cpp SYCL + MTP `n=1`                    | 33.0         |
| llama.cpp SYCL, no MTP                        | 25.6         |
| llama.cpp Vulkan + MTP `--spec-draft-n-max 1` | 16.0         |
| llama.cpp Vulkan + MTP `n=2`                  | 15.1         |
| llama.cpp Vulkan, no MTP                      | 11.7         |
| vLLM XPU + MTP4, C1 (patched, cache off)      | **77.7**     |
| vLLM XPU, no-spec, C1                         | 30.4         |
| vLLM XPU, no-spec, concurrency 3 (aggregate)  | 30.6         |

- Single user / serial prompts → **vLLM XPU + MTP4** (patched) is now the
  fastest (~78 tok/s; llama.cpp SYCL ~34). Parallel requests → vLLM, MTP off.
- vLLM C1 number: p512/g128 greedy, cache off, cookbook realworld harness,
  n=5 median 77.7 (75.3-79.9), MTP acceptance 94.3%. Cookbook's cell on their
  B70: 83.7 / 95.0% (230 W cap; ours reads 275 W) — ~93% reproduction.
- Bench prompt matters hugely: repeated filler sentences collapse MTP
  acceptance to ~50%; always bench with realistic prompts (`benchmarks/`).
- Both llama.cpp backends use the Q4_K_M GGUF
  (`~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/`) + its
  `nextn` MTP head (`--spec-type draft-mtp`).
- SYCL uses the official `ghcr.io/ggml-org/llama.cpp:server-intel` image, which
  bundles the oneAPI 2026 runtime (`libsycl.so.9`, oneDNN, oneMKL). The B70 shows
  up as `SYCL0`. The standalone b11700 SYCL tarball will **not** run on the host
  (no oneAPI runtime) or in the vLLM image (it only has `libsycl.so.8`); its
  extracted copy sits in `llama-sycl/llama-b11700/` and is safe to delete.

## Model

- vLLM repo: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` — GPTQ INT4 g128
  symmetric, `mtp.*` draft tensors kept BF16, ~19 GB, 5 shards, lands in
  `models/`. Needs `--quantization gptq`, the pinned `f01e24f6` image, the two
  `patches/` files and `B70_MTP_BF16_DRAFT=1`; `--language-model-only` (text).
- llama.cpp backends use the Q4_K_M GGUF (see above), not this checkpoint.

## Gotchas

- All backends serve one unified port (`LLM_PORT`, default 8001): `VLLM_PORT`
  and `LLAMA_PORT` default to it, so Claude Code / opencode configs never
  change when switching. Only the running backend owns the port.
- Docker needs `--device /dev/dri` plus `--group-add` for `render`/`video`.
  No `--privileged` required.
- llama.cpp Vulkan: pin the B70 with `GGML_VK_VISIBLE_DEVICES=2` (index 0 is the
  Quadro P2000, which hits `device lost`; index 2 is the `BMG G31`). `serve_vulkan.sh`
  sets this. Do **not** set `VK_ICD_FILENAMES` (a wrong path → "no usable GPU").
- Stop llama.cpp with `pkill -x llama-server`; `pkill -f llama-server` also
  matches (and kills) the shell running the command.
- SYCL backend runs as container `llama-sycl` (image ~4.3 GB). Stop it with
  `docker rm -f llama-sycl`. It only sees the B70 (`SYCL0`); no device pinning
  needed.
- ReBAR is enabled (BAR2 = 32 GB); it was 256 MB in an earlier session.
- MTP4 works **single-stream only** on this stack: two-plus overlapping requests
  can kill EngineCore (vllm-xpu-kernels#510) — concurrent MTP needs the
  mixed-split v5 patch (not applied yet). `VLLM_MTP_N=0 ./serve_vllm.sh` disables
  speculation for parallel clients/agents.
- `serve_vllm.sh` currently runs `--no-enable-prefix-caching` (the cookbook's
  benchmark cell). For interactive serving the cookbook wants cache on + the
  three APC correctness patches (`patch_fix_*`).
- First start in a mode compiles torch graphs (up to ~12 min: Dynamo ~68 s +
  graph compile ~330 s). Later starts in the same mode take ~3.5–6 min (cache
  hit, ~4 s compile). MTP4 and no-MTP have separate cache keys, so each compiles
  once; keep the `vllm-compile-cache` volume and don't change compile-relevant
  flags. `--enforce-eager` starts faster but drops decode to ~4 t/s (unusable).
- Power cap here reads 275 W (`hwmon*/power1_cap`); the cookbook benchmarked at
  a 230 W cap.
- The page cache / `du` can misreport download progress; sum actual file sizes.
- If startup OOMs at `--max-model-len 262144`, lower it (e.g. 131072) in
  `serve_vllm.sh`.
- This link (eduroam Wi-Fi) is throttled; downloads may take hours. `hf`
  resumes, so re-run `download_model.sh` freely.
- The folder is relocatable: `env.sh`/`bashrc.llm` derive `LLM_DIR` from their
  own path, `LLAMA_SERVER` auto-globs the newest LM Studio Vulkan runtime, and
  `HF` falls back to `PATH`. Overridable: `LLAMA_GGUF`,
  `GGML_VK_VISIBLE_DEVICES`, `LLM_PORT` (sets the port for every backend).
- `benchmarks/b70-realworld-context-harness.py` is stdlib-only; the prompt
  generator needs `transformers` — run it inside `vllm-b70` (`docker exec`).

## Use from opencode

The B70 servers are registered as OpenAI-compatible providers in the **global**
opencode config (`~/.config/opencode/opencode.jsonc`; archived copy in
`config/`). There is no project-local `opencode.jsonc` anymore — this AGENTS.md
is picked up automatically.

- provider `vllm`, base URL `http://127.0.0.1:8001/v1`, model id `qwen`
- provider `sycl`, base URL `http://127.0.0.1:8001/v1`, model id `qwen`
- provider `vulkan`, base URL `http://127.0.0.1:8001/v1`, model id `qwen`
  (all backends share the unified port; the running one answers whichever
  provider/model id you pick)
- every `qwen` model has thinking variants `off` / `low` / `medium` / `xhigh`
  (flat `chat_template_kwargs`: `enable_thinking` + `reasoning_effort`; verified
  to reach the request body on opencode 1.18.35)

Start the backend you want first, then run `opencode` and restart it after
config changes (config is not hot-reloaded). This folder no longer forces a
default model (global default is `qwen/qwen3.5-9b`) — pick `vllm/qwen`,
`sycl/qwen` or `vulkan/qwen` plus a variant in the TUI; only the backend that is
currently running will answer.

Measured single-stream: vLLM + MTP4 ~78, llama.cpp SYCL ~34, Vulkan ~16 tok/s;
vLLM without MTP ~30, and ~30.6 tok/s aggregate at concurrency 3.

## Use from Claude Code

Claude Code uses the Anthropic Messages API. Both vLLM and llama.cpp serve
`/v1/messages` on the unified port, so one config works for every backend. The
live config is `~/.claude/settings.json` (B70 template:
`config/claude.settings.json`; previous LM Studio config:
`config/claude.settings.backup.json`):

- `ANTHROPIC_BASE_URL=http://127.0.0.1:8001` (no `/v1`; Claude appends it)
- `ANTHROPIC_AUTH_TOKEN=local` (dummy; neither server checks auth)
- `ANTHROPIC_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL` and
  `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL` are all `qwen` (settings `model`
  is `qwen` too). Do **not** use `--model sonnet`: Claude Code resolves it to
  `claude-sonnet-5`, which vLLM rejects.
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` stops the background
  session-title request, which would overlap the main request (MTP4 crash).
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072` matches vLLM's `--max-model-len`.

`patches/qwen3.8-claude.jinja` (the model's own template with hard assertions
relaxed) is loaded by both backend families. Without it, every Claude Code main
request 500s on mid-conversation `role:"system"` (llama.cpp #27367). Tool use
needs `--jinja` on llama.cpp (`serve_sycl.sh` / `serve_vulkan.sh`) and
`--enable-auto-tool-choice --tool-call-parser qwen3_coder` on vLLM
(`serve_vllm.sh`). Restart Claude Code after switching backend (config is not
hot-reloaded). Only the running backend answers; start one with
`b70-vllm-claude` / `b70-sycl` / `b70-vulkan` first. Claude Code issues
overlapping requests, so vLLM must run with `VLLM_MTP_N=0` (`b70-vllm-claude`);
MTP4 kills EngineCore even in a normal session. llama.cpp tolerates the overlap.
