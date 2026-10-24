# Qwen3.8 on an Intel Arc Pro B70

Run and benchmark **Qwen3.8-27B** (dense) or **Qwen3.8-35B-A3B** (MoE) locally on
an **Intel Arc Pro B70** (32 GB VRAM). Two interchangeable backends are
available, and **only one fits on the card at a time** — each `serve_*` script
stops the others first.

## Model selection

Edit the one line at the top of `env.sh`:

```bash
LLM_MODEL="moe"     # or "dense"
```

| `LLM_MODEL` | Model | Backends |
| ----------- | ----- | -------- |
| `moe` | Qwen3.8-35B-A3B Distill, Q4_K_M, ~22 GB | llama.cpp **SYCL** |
| `dense` | Qwen3.8-27B, GPTQ INT4 ~19 GB + Q4_K_M GGUF ~17 GB | **vLLM**, SYCL |

The MoE download is GGUF-only, so vLLM cannot serve it. The `b70-vllm*` aliases
therefore force the dense checkpoint; a plain `./serve_vllm.sh` exits with a
clear message when `LLM_MODEL="moe"`. Both models are advertised as `qwen`, so
opencode/Claude Code configs do not change. For a one-off run without editing
`env.sh`: `LLM_MODEL_OVERRIDE=moe ./serve_sycl.sh`.

| Backend                          | Port | Use it for                       |
| -------------------------------- | ---- | -------------------------------- |
| llama.cpp **SYCL** (Docker)      | 8001 | Fastest llama.cpp (**MoE ~45-50 t/s** with SYCL graphs) |
| **vLLM XPU** (Docker)            | 8001 | Fastest overall (**~78 t/s** with MTP4); overlap needs v5 or MTP off |

## Hardware

- GPU: Intel Arc Pro B70 `BMG-G31` (`8086:e223`), 32 GB VRAM, `xe` driver, ReBAR on.
- SYCL/Docker needs `--device /dev/dri` (the scripts handle group ids).

## Quick start

```bash
source ~/.bashrc            # loads aliases (b70-*) and env.sh

b70-sycl                    # llama.cpp SYCL + MTP, model from env.sh -> :8001
b70-sycl-moe                #   same, forcing the MoE model
b70-sycl-dense              #   same, forcing the dense 27B
#   or
b70-vllm                    #    vLLM XPU (dense only)        -> :8001
b70-vllm-claude             #    vLLM, MTP off + prefix cache  -> :8001
b70-vllm-fullctx            #    vLLM, full 262144 ctx, C1     -> :8001

./benchmarks/bench_vllm.sh 512 3   # benchmark vLLM (max_tokens, concurrency)
```

Stop / watch:

```bash
b70-sycl-stop   ; b70-sycl-logs      # llama.cpp SYCL (docker 'llama-sycl')
b70-vllm-stop   ; b70-vllm-logs      # vLLM (docker 'vllm-b70')
b70-what                             # show which backend is currently live
b70-model                            # show the LLM_MODEL selected in env.sh

# what the log lines mean: see LOGS.md
```

(Without aliases, run the scripts directly: `./serve_sycl.sh`, etc.)

## Requirements & migration

The folder is **relocatable** — scripts derive their paths from their own
location (`env.sh`, `bashrc.llm`), and everything they need is inside it
(patches, benchmark prompts, opencode config archive). The pinned vLLM image
tar under `images/` is optional (gitignored) — Docker pulls it if absent.
Models (~17–22 GB per model) are re-downloadable.

**On a new PC you need:**

- Linux with an Intel **Arc Pro B70** (`xe` kernel driver, `/dev/dri` present,
  Resizable BAR enabled in BIOS).
- **Docker** (your user able to run `docker` without sudo).
- Your user in the **`render`** and **`video`** groups (their gids are passed to
  the containers).
- **Python 3** (benchmarks use the standard library only).
- **`hf` CLI** — `pipx install "huggingface_hub[cli]"` (a venv at
  `~/.venvs/hf/bin/hf` is found too) — used by `download_model.sh`.
- **LM Studio** — the source of the GGUFs in `~/.lmstudio/models/…`
  (`Qwen3.8-27B-Q4_K_M.gguf`, and `Qwen3.8-35B-A3B-Q4_K_M.gguf` for the MoE);
  SYCL uses these files (override with `LLAMA_GGUF_OVERRIDE`).
- **opencode** — optional, for the OpenAI-compatible providers.

**Install the dependencies (Ubuntu — 26.04 tested):**

```bash
sudo apt update
sudo apt install -y docker.io python3 pipx intel-gpu-tools
sudo usermod -aG docker,render,video "$USER"           # takes effect next login
pipx install "huggingface_hub[cli]" && pipx ensurepath # provides `hf`
```

- No GPU driver install needed — `xe` is in-kernel; only ReBAR in BIOS matters.
- Skip `render`/`video` if they don't exist yet (`getent group render video`).
- `intel-gpu-tools` only provides the optional `intel_gpu_top` monitor; full B70
  telemetry (power/VRAM) comes from the container:
  `docker exec vllm-b70 xpu-smi stats -d 0`.
- Not in apt: **LM Studio** (source of the GGUFs — from lmstudio.ai) and
  **opencode** (`curl -fsSL https://opencode.ai/install | bash`).

**Migrate:**

1. Copy the folder, e.g. `rsync -a --exclude models/ ./ newpc:~/b70-llm/`.
   (`llama-sycl/llama-b11700/` is an extracted standalone SYCL build that does
   not run — safe to exclude.)
2. Load the pinned vLLM image: `docker load -i images/vllm-openai-xpu-f01e24f6.tar`
   (otherwise Docker pulls it automatically; `env.sh` pins it by digest).
3. Add the aliases: append `source ~/b70-llm/bashrc.llm` to `~/.bashrc`.
4. opencode: `cp ~/b70-llm/config/opencode.global.jsonc ~/.config/opencode/opencode.jsonc`
   (it contains your OpenRouter API key — keep it private).
5. Claude Code: `mkdir -p ~/.claude && cp ~/b70-llm/config/claude.settings.json ~/.claude/settings.json`.
6. Fetch the dense vLLM checkpoint: `./models/download_model.sh` (~19 GB, resumable).
   The GGUFs come from LM Studio (`~/.lmstudio/models/…`); set `LLM_MODEL="moe"`
   or `"dense"` in `env.sh` to pick one.
7. Start: `b70-vllm` (dense; first compile ~4 min, later starts ~2 min) or
   `b70-sycl` (the SYCL image is pulled automatically from ghcr).
8. Verify: `b70-what`, then `./benchmarks/bench_vllm.sh 128 1`.

Machine-specific values can be overridden via environment variables:
`LLAMA_GGUF_OVERRIDE`, `LLAMA_CHAT_TEMPLATE_OVERRIDE`, `HF`,
`VLLM_PORT`, `LLAMA_PORT`, `LLAMA_CTX`, `LLAMA_SYCL_SPEC_N_MAX`,
`LLAMA_SYCL_ENABLE_GRAPH`, `LLAMA_SYCL_SPARSE_FA`, `LLAMA_SYCL_PARALLEL`,
`VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS`, `VLLM_GPU_MEM_UTIL`,
`VLLM_KV_CACHE_DTYPE`, `VLLM_EXTRA_ARGS`.

## Measured performance

Decode speed, single stream unless noted, on this B70 + Qwen3.8-27B:

> The MoE (`Qwen3.8-35B-A3B`) numbers below were re-measured 2026-10-24 with a
> realistic 56-token prompt / 192-token greedy decode. The production SYCL
> config now enables SYCL graph capture (`GGML_SYCL_ENABLE_GRAPH=1`) and sparse
> flash attention (`GGML_SYCL_SPARSE_FA=1`): **~45-50 t/s** with the full 262144
> context allocation, up from ~27 t/s. Keep `LLAMA_SYCL_PARALLEL=1`: llama.cpp
> divides `-c` across slots, so `--parallel 4` gives each request only 65536
> tokens. First load of the 22 GB MoE still takes ~3.5-6 min
> (JIT + weight reorder, not disk), so `serve_sycl.sh` waits up to 15 min.
> Use the `benchmarks/` harness for headline numbers with realistic prompts.

| Backend / config                                | decode tok/s |
| ----------------------------------------------- | ------------ |
| **llama.cpp SYCL + MTP** `--spec-draft-n-max 2` (dense, no graphs) | **34.0** |
| llama.cpp SYCL + MTP `n=1` (dense, no graphs)   | 33.0         |
| llama.cpp SYCL, no MTP (dense, no graphs)       | 25.6         |
| **vLLM XPU + MTP4**, C1 (patched, cache off)    | **77.7**     |
| vLLM XPU + MTP4 + APC + v5, C1 (thinking on)    | 66.8         |
| vLLM XPU, no-spec, single stream                | 30.4         |
| vLLM XPU, no-spec, concurrency 3 (aggregate)    | 30.6         |

**MoE SYCL tuning (2026-10-24, MTP n=2 unless noted).** 32k context, one slot,
realistic 56-token prompt, 192-token greedy decode; median of 3 unless noted:

| Config | decode tok/s |
| ------ | ------------ |
| baseline (stock image defaults) | 26.7 |
| `GGML_SYCL_ENABLE_GRAPH=1` | **48.5** |
| `GGML_SYCL_PRIORITIZE_DMMV=1` | 31.9 |
| graph + `PRIORITIZE_DMMV=1` | 22.0 (they interfere) |
| graph + MTP `n=3` / `n=4` | 40.2 / 40.6 |
| graph + `q8_0` KV / `f16` KV | 41.6 / 30.7 |
| full 262144 ctx + graph, short prompt | 35.3 |
| full 262144 ctx + graph + `SPARSE_FA=1`, short prompt | **48.0** |
| full 262144 ctx + graph + `SPARSE_FA=1`, 52.8k-token prompt | 27.7 |

Conclusions: enable SYCL graphs; keep MTP n=2, q4_0 KV, and
`LLAMA_SYCL_PARALLEL=1` for true 262144 context (4 slots => 64k each); enable
sparse FA so the 262144 allocation does not tax short prompts. Sparse FA
is exact here — it compacts the finite causal-mask entries, and falls back when
the compacted set exceeds half the cache. `PRIORITIZE_DMMV`, deeper MTP drafts,
q8_0/f16 KV, and forcing MMQ were all slower. `SYCL_CACHE_PERSISTENT=1` crashes
this image (llama.cpp docs also advise against it); do not use it.

**Takeaways**

- Serial / single-user (e.g. an editor assistant) → **vLLM XPU + MTP4** (patched):
  ~78 tok/s vs llama.cpp SYCL ~34. llama.cpp stays useful as a no-patch-stack,
  no-extra-container fallback.
- Many parallel requests → **vLLM** (llama.cpp has no continuous batching here).
  MTP4 needs the v5 split for overlap, otherwise it is single-stream only (see
  Gotchas).
- **MTP** (speculative decoding) is the single biggest win: vLLM 30 → 78 tok/s
  with MTP4; llama.cpp's best depth is `n=2` on SYCL.
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

**Prefix caching (interactive, MTP off).** `VLLM_PREFIX_CACHING=1` enables vLLM's
APC; hybrid GDN then runs `mamba_cache_mode="align"`, so `serve_vllm.sh` also
applies the cookbook's three correctness fixes (`patch_fix_accepted_sync.py`,
`patch_fix_backward_copy.py`, `patch_fix_eagle_drop.py` — silent-corruption bugs
vllm#53919/#48375/#53505). Measured here with a 32.7k-token prompt, same request
three times, `max_tokens=1`:

| run | TTFT (+1 token) | prefix hit rate |
| --- | --- | --- |
| cold | 20.58 s | 0% |
| warm | 0.83 s | 97.9% |
| warm | 0.77 s | 97.9% |

Decode is unchanged (~30 t/s no-MTP); the win is per-turn prefill latency.
The KV pool shrinks to 6.95 GiB / 211k tokens (align padding; was 8.36 GiB /
263k), still ample for a single agent context. Multi-session concurrent reuse is
much weaker (model card: 0–38% hits).

**Prefix caching + MTP4 (overlap-safe).** `b70-vllm-claude-mtp` turns on cache
*and* the v5 mixed-batch split (`VLLM_MIXED_SPLIT=1`), which makes MTP4 survive
overlapping requests instead of killing EngineCore (#510). Measured 2026-10-22:

| Test | Result |
| ---- | ------ |
| Realworld p512/g128, thinking on, n=5 | median 66.8 t/s (65.7–68.8), acceptance 91–98% |
| `bench_vllm.sh 512 3` | 87.2 t/s aggregate (29–31 per request) |
| 3 mixed concurrent requests (32k prefill + decode + short) | all 200, no fatal, `Waiting: 0` |
| Warm TTFT, 32.7k prompt | cold 23.35 s → warm 5.03 s, 81.4% hits |
| Real Claude Code (Read, then multi-turn Write) | correct, acceptance 81–85%, no errors |

Cache-on + v5 costs ~13% decode vs cache-off (66.8 vs 77.7) and warm prefill is
~5 s / 81% rather than the MTP-off 0.8 s / 98% — still ~18 s faster than cold.
The KV pool with MTP4 + cache is 5.68 GiB / 141,882 tokens (1.08x at 131k), so a
single 32k agent context fits comfortably; very large contexts can crowd it.

Note the headline 77.7 cell was measured **with thinking on**. The server now
defaults to `enable_thinking: false`; the committed `benchmarks/prompts/` were
calibrated thinking-on, so the harness reports `expected 512, got 472` unless the
request sets `chat_template_kwargs` to match.

## Full 262144 context on vLLM

Yes, but with a tradeoff: vLLM can serve the full 262144-token context, but MTP
must be off. The default `b70-vllm` uses `--max-model-len 131072` because it is
tuned for 64 concurrent sequences plus prefix caching and MTP4. The dense 27B is
hybrid GDN: only 16 of its 64 layers keep a KV cache, with 4 KV heads and
`head_dim=256`. At fp8 KV that is **32 KiB/token**, so:

| context | fp8 KV |
| ------- | ------ |
| 131072 | 4 GiB |
| 196352 | ~6 GiB |
| 262144 | 8 GiB |

Weights are ~17.1 GiB + ~2.5 GiB peak activations + ~0.2 GiB graphs, so one
full-context stream fits, but not with MTP4's extra draft memory at a safe
utilization. Tested 2026-10-24 with `b70-vllm-fullctx` (MTP **off**,
`VLLM_GPU_MEM_UTIL=0.95`, one sequence):

- vLLM reports **9.22 GiB KV / 295,108 tokens** (1.13x at 262144).
- A **158,393-token** prompt completed successfully (1 output token).
- Short-prompt decode is **27.3 t/s** (vs 77.7 t/s at 131k with MTP4).

Use the dedicated alias:

```bash
b70-vllm-fullctx
# equivalent to:
LLM_MODEL_OVERRIDE=dense VLLM_MAX_MODEL_LEN=262144 VLLM_MAX_NUM_SEQS=1 \
VLLM_MAX_BATCHED_TOKENS=4096 VLLM_GPU_MEM_UTIL=0.95 VLLM_MTP_N=0 \
VLLM_PREFIX_CACHING=0 ./serve_vllm.sh
```

MTP4 at 262144 needs 9.33 GiB of KV plus draft overhead; at `0.92` vLLM had only
7.21 GiB available (estimated max context 196,352). Raising utilization to
0.99 to squeeze in MTP4 is untested and leaves no headroom, so it is not
recommended. If you want both long context and speculation, use ~196k with MTP4
(`VLLM_MAX_MODEL_LEN=196352`), the default 131k + MTP4, or the SYCL MoE at
262144 (MTP n=2, ~45 t/s, with `LLAMA_SYCL_PARALLEL=1`). `VLLM_MAX_NUM_SEQS=2`
needs ~16 GiB of KV and will not fit.

## Backend comparison (MoE vs dense)

| Model | Backend | Pros | Cons |
| ------- | ----- | ---- | ---- |
| MoE (`Qwen3.8-35B-A3B`) | SYCL | MTP n=2; SYCL graphs + sparse FA; true 262144 ctx via `LLAMA_SYCL_PARALLEL=1`; ~25.5 t/s full-context decode; GGUF-native. | vLLM cannot run MoE (GGUF-only); no continuous batching; single-user only. |
| MoE (`Qwen3.8-35B-A3B`) | vLLM | — | Not feasible on the B70 with the current checkpoint. |
| Dense (`Qwen3.8-27B`) | vLLM | Fastest overall (~77.7 t/s MTP4 @131k, one seq); full ctx via MTP off (27.3 t/s); prefix caching; continuous batching. | Full ctx needs MTP off; one sequence; 2 sequences need ~16 GiB KV and will not fit; startup ~8–12 min. |
| Dense (`Qwen3.8-27B`) | SYCL | Can use true 262144 ctx via `LLAMA_SYCL_PARALLEL=1`; MTP n=2; GGUF path. | Slower than vLLM; no continuous batching; full-context speed not separately measured for dense. |

## Native (no Docker) builds

Docker is currently the supported path because the pinned images bundle the
exact oneAPI/vLLM stacks that were validated on the B70. Native builds are worth
considering for two reasons: simpler process management, and AOT-compiling the
SYCL kernels for `BMG-G31` instead of paying the image's JIT cost on every start.

**llama.cpp SYCL.** Install the Intel oneAPI Toolkit **2026.1** (it includes
oneDNN; 2026.0+ removed oneDNN from Deep Learning Essentials), then:

```bash
source /opt/intel/oneapi/setvars.sh
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build-sycl -S llama.cpp \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build llama.cpp/build-sycl --config Release -j
./llama.cpp/build-sycl/bin/llama-server --list-devices
```

The runtime flags in this repo (`GGML_SYCL_ENABLE_GRAPH=1`,
`GGML_SYCL_SPARSE_FA=1`) apply unchanged. To try AOT, add the B70's
`GGML_SYCL_DEVICE_ARCH` value (see the `--offload-arch` table for your oneAPI
version); the image itself is built without AOT, which is why first load is
~3.5-6 min. This host does not have oneAPI installed and `sudo` is interactive,
so the native build has not been executed here.

**vLLM XPU.** Native is not recommended: the pinned
`vllm/vllm-openai-xpu@sha256:f01e24f6...` image is the validated
vLLM 0.27.2rc1 + `vllm-xpu-kernels` + cookbook-patch combination. A native venv
would need matching XPU PyTorch/vLLM wheels and the same patches, and small
version drift can change kernel/compile behavior. Extracting `/app/llama-server`
from the SYCL image is also not a real simplification: it still needs the
image's oneAPI runtime libraries (`libsycl`, oneMKL, oneDNN, compiler runtime)
and a wrapper to set `LD_LIBRARY_PATH`.

## Files

| File                  | Purpose                                                       |
| --------------------- | ------------------------------------------------------------- |
| `env.sh`              | Shared paths, image tags, device ids, and the `LLM_MODEL` switch. Sourced by all scripts. |
| `serve_sycl.sh`       | Start llama.cpp SYCL (Docker + MTP) on :8001.                   |
| `status.sh`           | Show which backend is currently serving.                        |
| `serve_vllm.sh`       | Start vLLM on :8001 and wait until it is ready.                  |
| `models/download_model.sh` | Resumable/retrying HF download of the dense INT4 vLLM checkpoint (`LLM_MODEL=dense` only). |
| `benchmarks/`         | Cookbook realworld harness, prompt generator, p512 prompts, and the `bench_vllm*.sh` scripts. |
| `patches/`            | Cookbook MTP + APC correctness patches + Claude Code templates (`qwen3.8-claude.jinja` for dense, `qwen3.8-moe-claude.jinja` for MoE). |
| `images/`             | Docker-save archive of the pinned vLLM image (4.65 GB).         |
| `bashrc.llm`          | Aliases + env; sourced from `~/.bashrc`.                        |
| `config/`             | Archive of opencode + Claude Code configs.                      |
| `AGENTS.md`           | Notes for coding agents (this project's `instructions`).        |
| `LOGS.md`             | Line-by-line log field guide (vLLM + llama.cpp).                |

## Models

Both models expose a `nextn` MTP draft head, so llama.cpp MTP works with either.
The OpenAI model id stays `qwen` in both cases.

- **dense (`LLM_MODEL="dense"`)**
  - **vLLM**: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (GPTQ INT4 g128
    symmetric, BF16 MTP heads, ~19 GB), fetched by `download_model.sh` into
    `models/`; needs the patched launch in `serve_vllm.sh`.
   - **llama.cpp**: `Qwen3.8-27B-Q4_K_M.gguf` (+ `nextn` MTP head baked in) at
     `~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/`. SYCL uses this GGUF.
- **MoE (`LLM_MODEL="moe"`)** — `Qwen3.8-35B-A3B-Distill-GGUF`, Q4_K_M (~22 GB)
   at `~/.lmstudio/models/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF/`. 41 blocks,
  256 experts / 8 active, 262144-token context, `nextn` MTP head. GGUF-only, so
  it runs on llama.cpp SYCL, **not vLLM**.
  `patches/qwen3.8-moe-claude.jinja` is the MoE's own chat template with only the
  Claude Code assertions relaxed (it intentionally does not inject the dense
  model's `reasoning_effort` instructions).

## Use from opencode

The two servers are registered as OpenAI-compatible providers in the global
opencode config (`~/.config/opencode/opencode.jsonc`; archived copy in
`config/`). Both use the **same** base URL, because only one backend runs
at a time:

- `vllm/qwen` → `http://127.0.0.1:8001/v1`
- `sycl/qwen` → `http://127.0.0.1:8001/v1`
- every model offers `off` / `low` / `medium` / `xhigh` thinking variants
  (`enable_thinking` + `reasoning_effort` template kwargs)

Start the backend you want, then run `opencode` from any folder. Config is not
hot-reloaded — restart opencode after changing the config. Switch model and
variant with the picker in the TUI; the running backend answers on :8001.

## Use from Claude Code

Claude Code speaks the Anthropic Messages API, which vLLM and llama.cpp both
serve on the unified port, so the same config works with any backend. The live
config is `~/.claude/settings.json` (B70 template:
`config/claude.settings.json`; the previous LM Studio config is preserved as
`config/claude.settings.backup.json`):

```jsonc
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8001",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_MODEL": "qwen",
    "ANTHROPIC_SMALL_FAST_MODEL": "qwen",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "131072"
  },
  "model": "qwen",
  "availableModels": ["qwen"]
}
```

Start it with `claude` (or `claude --model qwen`). Do **not** pass
`--model sonnet`/`opus`: Claude Code resolves those to Anthropic catalog ids
(`claude-sonnet-5`), which vLLM rejects even though the `ANTHROPIC_DEFAULT_*`
values are set. The `[claude-code:unrecognized_model]` warning is expected.

Why the extra knobs:

- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` stops the background
  session-title request. It overlaps the main request and crashed MTP4 without
  v5; with `b70-vllm-claude-mtp` it is optional, but still saves a request.
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072` matches vLLM's `--max-model-len` (SYCL
  could do 262144, but the shared config must stay within vLLM's limit).
- `patches/qwen3.8-claude.jinja` is the model's own template with the hard
  assertions relaxed: Claude Code ≥2.1.154 injects extra `role:"system"` messages
  mid-conversation and sends tool-result-only turns, both of which the stock
  Qwen3.8 template rejects (llama.cpp #27367). Both `serve_*` families load it.

Tool use needs `--jinja` on llama.cpp (set in `serve_sycl.sh`) and
`--enable-auto-tool-choice --tool-call-parser qwen3_coder`
on vLLM (set in `serve_vllm.sh`). Restart Claude Code after switching backend or
editing the config.

**vLLM + Claude Code: MTP off, or MTP4 + cache + v5.** Claude Code sends
overlapping requests (tool loops, background calls), and plain MTP4 kills
EngineCore even in a normal session. Verified 2026-10-22: `b70-vllm-claude-mtp`
(MTP4 + prefix cache + `VLLM_MIXED_SPLIT=1`) survives mixed 32k-prefill + decode
traffic and a real multi-turn Claude Code session, at ~67 t/s and ~5 s warm
prefill. `b70-vllm-claude` (MTP off, ~30 t/s, warm prefill <1 s) is the safer
fallback if anything regresses. SYCL (~34 t/s) also tolerates the
overlap and keeps MTP.

### vLLM startup time (MTP on / off)

`serve_vllm.sh` now blocks until `Application startup complete`. Measured here:

- a mode whose compile cache is cold pays torch.compile: up to ~12 min on the
  first start (Dynamo ~68 s + graph compile ~330 s);
- later starts in the **same** mode: ~3.5–6 min (compile cache hit, ~4 s; warm
  page cache helps the weight load);
- MTP4 and no-MTP use different compile keys, so each pays the cold start once;
  after that, switching modes only costs the warm restart;
- the fixed part (~3.5–5 min) is Python/XCCL init (~2.5 min), weight load
  (~27–42 s), Triton warmup (~50 s) and graph capture (~30 s);
- `--enforce-eager` skips compile but drops decode to **~4 t/s** — not usable.

Keep the `vllm-compile-cache` volume and avoid changing compile-relevant flags
(`--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`,
`--kv-cache-dtype`, MTP) or the cache key changes and the next start recompiles.
If fast restarts matter more than throughput, use SYCL (starts in ~1 min).

## Gotchas

- Only one backend fits the 32 GB card; every `serve_*` script stops the others.
- The MoE download is **GGUF-only**; vLLM has no checkpoint for it. Use
  `b70-sycl` with `LLM_MODEL="moe"`, or `b70-vllm*` (which forces
  the dense checkpoint). Plain `./serve_vllm.sh` refuses to start for MoE.
- The standalone llama.cpp SYCL tarball does **not** run here — use the
  `ghcr.io/ggml-org/llama.cpp:server-intel` image, which bundles the oneAPI
  runtime.
- The network link here is throttled; the vLLM image / 4.3 GB SYCL image take a
  while to pull, and `hf` downloads may take hours (they resume).
- MTP4 is **single-stream only without v5**: overlapping requests can kill
  EngineCore (vllm-xpu-kernels#510). `VLLM_MIXED_SPLIT=1` applies the v5
  mixed-batch split, which is verified to tolerate overlap (see
  `b70-vllm-claude-mtp`). Without it, set `VLLM_MTP_N=0` for parallel clients or
  subagents (~30 t/s, concurrency-safe).
- Prefix caching is opt-in (`VLLM_PREFIX_CACHING=1`; `b70-vllm-claude*` set it).
  Default stays off to match the cookbook bench cell. On this hybrid GDN model
  APC forces `mamba_cache_mode="align"`, hence the three `patch_fix_*` patches
  that `serve_vllm.sh` applies when cache is on.
  `b70-vllm-cache` is the MTP4 + cache combo **without** v5 and is unsafe under
  overlap; use `b70-vllm-claude-mtp` instead.
