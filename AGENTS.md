# Intel Arc Pro B70 inference workflow

Serving `Qwen3.8-27B` (dense) or `Qwen3.8-35B-A3B` (MoE) on an **Intel Arc Pro
B70** and measuring throughput. Pick one in `env.sh` (`LLM_MODEL="moe"` or
`"dense"`; one-off: `LLM_MODEL_OVERRIDE=moe ./serve_sycl.sh`). Two
interchangeable backends, which cannot share the 32 GB card:

- **llama.cpp SYCL** (ggml `server-intel` Docker image) — fastest llama.cpp;
  ~43 t/s for the MoE with SYCL graphs + sparse FA + MTP n=2 at 4x64k slots,
  or ~25.5 t/s with a true 262144-token slot (`b70-sycl`).
- **vLLM XPU** (`vllm/vllm-openai-xpu` image + cookbook MTP patches) — fastest
  (~78 t/s with MTP4 single-stream; ~67 t/s with prefix cache + v5, which also
  tolerates overlapping requests). **Dense only**: the MoE download is GGUF-only.

## Workflow

```bash
source ./env.sh          # or open a new shell (aliases load from bashrc)
# env.sh: LLM_MODEL="moe" (Qwen3.8-35B-A3B) or "dense" (Qwen3.8-27B)

# Dense / vLLM path:
./models/download_model.sh   # resumable; ~19 GB into models/ (MTP GPTQ checkpoint)
./serve_vllm.sh              # container 'vllm-b70': MTP4 + cookbook patches; waits
                             # for 'Application startup complete' on :8001
docker logs -f vllm-b70      # first start in a mode ~8-12 min; warm ~3.5-6 min
./benchmarks/bench_vllm.sh 512 1    # single-stream tok/s
./benchmarks/bench_vllm.sh 512 3    # concurrency 3
./benchmarks/bench_vllm_stream.sh 128 5 off greedy   # article-style single-stream bench
```

Reference decode benchmark (realworld prompts; full command in README §Benchmark):
`benchmarks/b70-realworld-context-harness.py`.

llama.cpp (faster single-stream; stops vLLM because only one fits the card).
Both models work here; the MoE GGUF is at
`~/.lmstudio/models/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF/`:

```bash
./serve_sycl.sh           # SYCL + MTP n=2 + SYCL graphs (fastest) on :8001
```

Aliases (after `source ~/.bashrc`): `b70-vllm` (default: MTP4 + cache + v5;
overlap-safe, ~67 t/s), `b70-vllm-fast` (MTP4, cache off; fastest single-stream,
~78 t/s), `b70-vllm-fullctx` (full 262144 ctx, one stream), `b70-vllm-logs`,
`b70-vllm-stop`, `b70-sycl` (true 262k, one slot),
`b70-sycl-dense`, `b70-sycl-moe`, `b70-sycl-logs`, `b70-sycl-stop`, `b70-what`,
`b70-model`. vLLM aliases force `LLM_MODEL_OVERRIDE=dense` (MoE is GGUF-only).
`b70-vllm-fullctx` uses `VLLM_MAX_MODEL_LEN=262144 VLLM_MAX_NUM_SEQS=1
VLLM_GPU_MEM_UTIL=0.95 VLLM_MTP_N=0` (full context works, MTP off);
the default `b70-vllm` stays at 131072 for 64 slots + MTP4.

## Reference

This `AGENTS.md` is picked up automatically by coding agents (there is no
`opencode.jsonc` in the folder root). Everything else — hardware, the file map,
measured performance, the model, gotchas, and how to point **opencode** /
**Claude Code** at the servers — lives in **README.md**, the single source of
truth. Log-line meanings: **LOGS.md**.
