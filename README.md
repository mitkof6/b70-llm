# Qwen3.8-27B on an Intel Arc Pro B70

Serve **Qwen3.8-27B** (dense, GPTQ INT4 + MTP) with **vLLM XPU v0.32.0** on a
32 GB Arc Pro B70. One model, one backend, one API: `http://127.0.0.1:8001`,
model id `qwen`.

Licensed under the GNU GPL v3 — see [LICENSE](LICENSE).

## Demo

A live Claude Code session against the local server:

<video src="docs/demo.mp4" controls width="100%"></video>

## Stack

- vLLM `vllm/vllm-openai-xpu:v0.32.0` (upstream, no cookbook patches)
- MTP n=10 + prefix caching, fp8 KV, float16
- 196608 context, 64 sequences, 4096-token prefill chunks
- Explicit 8.6 GiB KV pool + CUDA-graph estimate off
- Coding-friendly defaults: `temperature 0.2`, `top_p 0.8`, `top_k 20`

## Quick start

```bash
source ~/.bashrc            # loads the b70-* aliases
./models/download_model.sh  # first time only: ~19 GB
b70-vllm                    # start on :8001
b70-vllm-logs               # follow logs
b70-what                    # is it running?
b70-vllm-stop               # stop and remove the container
```

Directly, without aliases: `./serve_vllm.sh`.

Upgrading from the old setup? Re-source `~/.bashrc` (or open a new shell) so
the retired aliases are replaced. `env.sh` also ignores the old `LLM_MODEL` /
MTP4 export automatically.

First start compiles (~8–12 min); warm starts are ~3.5–6 min. The
`vllm-compile-cache` volume keeps the warm cache. If the eGPU enumerates in
Small BAR mode, `serve_vllm.sh` fails early with recovery instructions.

## Benchmark

```bash
./benchmarks/bench_vllm.sh 512 1      # single-stream tok/s
./benchmarks/bench_vllm.sh 512 3      # concurrency 3 (aggregate)
./benchmarks/bench_vllm_stream.sh 128 5 off greedy
```

Reference decode benchmark (realworld prompts):

```bash
python3 benchmarks/b70-realworld-context-harness.py --thinking
```

The p512 prompts are calibrated with **thinking on**; `--thinking` is required
or the stock harness fails its 512-token check. The repeated-filler prompts in
`bench_vllm.sh` collapse MTP acceptance to ~50%, so use the realworld harness
for MTP tuning.

## Numbers

| Config | decode tok/s |
| ------ | ------------ |
| v0.32.0 + MTP10 + cache, C1 (p512, thinking on) | **95.2** |
| v0.32.0 + MTP10 + cache, C3 (3 real prompts) | **156–160** aggregate |
| v0.32.0 + MTP4 + cache, C1 (p512, thinking on) | 81.8 |
| v0.32.0, no MTP, C1 | 30.4 |

MTP is the big win: ~30 → ~95 t/s. MTP10 trades acceptance (94% → 85%) for
more decode on realistic prompts; on low-acceptance synthetic filler it can be
slower than n=4.

Long-context validation: a 110k-token request completed with ~118 s prefill and
~1.6 GiB GPU memory still free.

## Context and memory

196608 context with MTP10 + prefix caching fits by reserving the KV pool
explicitly: `--kv-cache-memory-bytes 9234179686` (8.6 GiB) plus
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. vLLM reports **200,540 KV
tokens** (1.02x at 196608).

The card is seen as 28.8/30.3 GiB free on startup, so 0.94 is the practical
utilization ceiling. Auto-sizing at 0.94 gave only 6.75 GiB KV (~145k max);
0.95 + graph-estimate off + 4096-token chunks was still 0.14 GiB short.
Set `VLLM_KV_CACHE_MEMORY_BYTES=""` to return to automatic sizing.

## Use from opencode / Claude Code

opencode: provider `vllm/qwen`, with `off`/`low`/`medium`/`xhigh` thinking
variants. The default model is `vllm/qwen#off`; restart opencode after config
changes. The `cost` block in the opencode config prices local usage at the
OpenRouter Qwen3.8-27B rates so `opencode stats` shows an estimate.

Claude Code speaks the Anthropic API, which vLLM serves natively. Copy
`config/claude.settings.json` to `~/.claude/settings.json`, then run `claude`
(not `--model sonnet/opus`). Tested with Claude Code 2.1.273; a benign
`unrecognized_model` warning for `qwen` is expected.

## Migrate to a new PC

Needs: Linux + Arc Pro B70 (`xe` driver, ReBAR on), Docker, user in
`render`/`video` groups, Python 3, `hf` CLI, opencode (optional).

```bash
sudo apt install -y docker.io python3 pipx intel-gpu-tools
sudo usermod -aG docker,render,video "$USER"   # next login
pipx install "huggingface_hub[cli]" && pipx ensurepath

rsync -a --exclude models/ ./ newpc:~/b70-llm/
echo 'source ~/b70-llm/bashrc.llm' >> ~/.bashrc

# opencode: add your own OpenRouter key if you keep that provider
mkdir -p ~/.config/opencode
cp ~/b70-llm/config/opencode.global.jsonc ~/.config/opencode/opencode.jsonc

mkdir -p ~/.claude
cp ~/b70-llm/config/claude.settings.json ~/.claude/settings.json

./models/download_model.sh
b70-vllm && b70-what && ./benchmarks/bench_vllm.sh 128 1
```

Useful overrides: `VLLM_PORT`, `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS`,
`VLLM_GPU_MEM_UTIL`, `VLLM_MTP_N`, `VLLM_KV_CACHE_MEMORY_BYTES`,
`VLLM_TEMPERATURE`, `VLLM_TOP_P`, `VLLM_TOP_K`, `VLLM_EXTRA_ARGS`.

## Gotchas

- One server owns the card and the port; `serve_vllm.sh` replaces `vllm-b70`.
- `--enforce-eager` drops decode to ~4 t/s. Don't change compile-relevant flags
  casually; it invalidates the compile cache.
- MTP10 is the default. Use `VLLM_MTP_N=4` for low-acceptance filler benches,
  or `VLLM_MTP_N=0` to disable speculation.
- Prefix caching is on by default; `VLLM_PREFIX_CACHING=0` disables it.
- The explicit 8.6 GiB KV reservation is what makes 196608 fit. Raise it or
  `VLLM_GPU_MEM_UTIL` only after checking free memory.
- There is no SYCL, MoE, or pinned-vLLM fallback in this repo anymore.

## Files

| File | Purpose |
| ---- | ------- |
| `env.sh` | Paths, model, image, port, GPU preflight (sourced by all scripts) |
| `serve_vllm.sh` | Start/replace the vLLM container on :8001 |
| `status.sh` | Report whether the server is running |
| `models/download_model.sh` | Resumable HF download of the vLLM checkpoint |
| `benchmarks/` | Realworld harness, prompt generator, `bench_vllm*.sh` |
| `patches/qwen3.8-claude.jinja` | Claude Code chat template |
| `bashrc.llm` | Aliases + env, sourced from `~/.bashrc` |
| `config/` | opencode + Claude Code config templates |
| `docs/` | Demo video embedded in this README |
| `LICENSE` | GNU GPL v3 |
