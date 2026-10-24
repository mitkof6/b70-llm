# Qwen3.8 on an Intel Arc Pro B70

Run **Qwen3.8-27B** (dense) or **Qwen3.8-35B-A3B** (MoE) on a 32 GB Arc Pro B70.
Two backends, **only one fits at a time** — each `serve_*` stops the others.

## Pick a model

One line in `env.sh`:

```bash
LLM_MODEL="moe"     # or "dense"
```

| `LLM_MODEL` | Model | Backends |
| ----------- | ----- | -------- |
| `moe`   | Qwen3.8-35B-A3B, Q4_K_M GGUF ~22 GB | SYCL only (vLLM has no MoE checkpoint) |
| `dense` | Qwen3.8-27B, GPTQ INT4 ~19 GB + Q4_K_M GGUF | vLLM + SYCL |

One-off without editing `env.sh`: `LLM_MODEL_OVERRIDE=moe ./serve_sycl.sh`.
Both models advertise as `qwen`, so opencode/Claude configs never change.

## Start a server

```bash
source ~/.bashrc        # loads the b70-* aliases
b70-vllm                # default: fastest + overlap-safe
# or
b70-sycl                # llama.cpp, model from env.sh
```

| Alias | Backend | Config | ~decode | Notes |
| ----- | ------- | ------ | ------- | ----- |
| `b70-vllm`        | vLLM | MTP4 + prefix cache + v5 | **67 t/s** | **default** — overlap-safe, best all-rounder |
| `b70-vllm-fast`   | vLLM | MTP4, cache off            | **78 t/s** | fastest single-stream; bench numbers |
| `b70-vllm-fullctx`| vLLM | full 262k ctx, MTP off     | 27 t/s   | one long stream |
| `b70-sycl`        | SYCL | MTP n=2 + graphs + sparse FA | 45–50 t/s (MoE) / 34 t/s (dense) | true 262k ctx |
| `b70-sycl-moe` / `b70-sycl-dense` | SYCL | same, model forced | | one-off model switch |

All servers listen on **:8001**.

## Stop / watch

```bash
b70-vllm-stop   ; b70-vllm-logs     # vLLM   (docker 'vllm-b70')
b70-sycl-stop   ; b70-sycl-logs     # SYCL   (docker 'llama-sycl')
b70-what                            # which backend is live
b70-model                           # which model env.sh selects
```

Log-line meanings: [LOGS.md](LOGS.md).

## Benchmark

```bash
./benchmarks/bench_vllm.sh 512 1     # single-stream tok/s
./benchmarks/bench_vllm.sh 512 3     # concurrency 3 (aggregate)
./benchmarks/bench_vllm_stream.sh 128 5 off greedy
```

Reference decode bench (realworld prompts): `benchmarks/b70-realworld-context-harness.py`.

## Numbers

| Config | decode tok/s |
| ------ | ------------ |
| vLLM + MTP4, C1 (cache off) | **77.7** |
| vLLM + MTP4 + cache + v5, C1 | 66.8 |
| vLLM, no MTP, C1 | 30.4 |
| SYCL + MTP n=2 (dense) | 34.0 |
| SYCL + MTP n=2 + graphs + sparse FA (MoE, 262k) | 45–50 |
| vLLM full 262k ctx, MTP off | 27.3 |

MTP (speculative decoding) is the big win: vLLM 30 → 78 t/s. Bench prompts matter
hugely — one repeated filler sentence collapses MTP acceptance to ~50%.

## Migrate to a new PC

Needs: Linux + Arc Pro B70 (`xe` driver, ReBAR on), Docker, user in `render`/`video`
groups, Python 3, `hf` CLI, LM Studio (GGUF source), opencode (optional).

```bash
# 1. install (Ubuntu 26.04)
sudo apt install -y docker.io python3 pipx intel-gpu-tools
sudo usermod -aG docker,render,video "$USER"     # next login
pipx install "huggingface_hub[cli]" && pipx ensurepath
# 2. copy the folder (models/ is re-downloadable)
rsync -a --exclude models/ ./ newpc:~/b70-llm/
# 3. aliases
echo 'source ~/b70-llm/bashrc.llm' >> ~/.bashrc
# 4. opencode (archive has no API key — add your OpenRouter key if you use it)
cp ~/b70-llm/config/opencode.global.jsonc ~/.config/opencode/opencode.jsonc
# 5. Claude Code
mkdir -p ~/.claude && cp ~/b70-llm/config/claude.settings.json ~/.claude/settings.json
# 6. fetch the dense checkpoint (MoE GGUFs come from LM Studio)
./models/download_model.sh
# 7. start + verify
b70-vllm && b70-what && ./benchmarks/bench_vllm.sh 128 1
```

Machine-specific overrides: `LLAMA_GGUF_OVERRIDE`, `VLLM_PORT`, `LLAMA_CTX`,
`VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS`, `VLLM_GPU_MEM_UTIL`, `VLLM_EXTRA_ARGS`,
`LLAMA_SYCL_ENABLE_GRAPH`, `LLAMA_SYCL_SPARSE_FA`, `LLAMA_SYCL_PARALLEL`.

## Use from opencode / Claude Code

Both servers are OpenAI-compatible on :8001. opencode: providers `vllm/qwen` and
`sycl/qwen`, each with `off`/`low`/`medium`/`xhigh` thinking variants — start the
backend, then `opencode`; pick model/variant in the TUI (config is not hot-reloaded).

Claude Code speaks the Anthropic API, which both backends also serve, so the same
`~/.claude/settings.json` works for either (template: `config/claude.settings.json`).
Run `claude` (not `--model sonnet/opus`).

## Gotchas

- One backend per card; every `serve_*` stops the others.
- MoE is GGUF-only → vLLM can't serve it; `b70-vllm*` force the dense checkpoint.
- **MTP4 is single-stream without v5** — overlapping requests kill EngineCore
  (vllm-xpu-kernels#510). The default `b70-vllm` sets v5, so it is overlap-safe;
  `b70-vllm-fast` (no v5) is for single-stream only.
- vLLM cold start ~8–12 min (torch.compile), warm ~3.5–6 min; SYCL starts in ~1 min.
  Keep the `vllm-compile-cache` volume and don't change compile-relevant flags or
  it recompiles. `--enforce-eager` drops decode to ~4 t/s.
- Full 262k ctx on vLLM needs MTP off (`b70-vllm-fullctx`); 2 sequences need ~16 GiB
  KV and won't fit.

## Files

| File | Purpose |
| ---- | ------- |
| `env.sh` | Paths, image tags, device ids, `LLM_MODEL` switch (sourced by all scripts) |
| `serve_sycl.sh` / `serve_vllm.sh` | Start each backend on :8001 |
| `status.sh` | Which backend is serving |
| `models/download_model.sh` | Resumable HF download of the dense vLLM checkpoint |
| `benchmarks/` | Realworld harness, prompt generator, `bench_vllm*.sh` |
| `patches/` | Cookbook MTP + APC patches + Claude Code chat templates |
| `images/` | Pinned vLLM image tar (optional; Docker pulls if absent) |
| `bashrc.llm` | Aliases + env, sourced from `~/.bashrc` |
| `config/` | opencode + Claude Code config archive |
| `AGENTS.md` / `LOGS.md` | Agent notes / log field guide |
