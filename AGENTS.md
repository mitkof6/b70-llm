# Intel Arc Pro B70 inference workflow

Serve **Qwen3.8-27B** (dense, GPTQ INT4 + MTP) with **vLLM XPU v0.32.0** on an
**Intel Arc Pro B70**. The model and backend are fixed: no model selector, no
SYCL, no pinned-vLLM fallback. One OpenAI-compatible API on `:8001`, model id
`qwen`.

Current recipe: MTP n=10 + prefix caching, fp8 KV, 196608 context, 64
sequences, 4096-token prefill chunks, explicit 8.6 GiB KV pool, CUDA-graph
estimate off, coding-friendly sampling (`temperature 0.2`, `top_p 0.8`,
`top_k 20`). Realworld p512 decode: **95.2 t/s**.

## Workflow

```bash
source ./env.sh              # or open a new shell (aliases load from bashrc)
./models/download_model.sh   # first time only: ~19 GB
./serve_vllm.sh              # container 'vllm-b70' on :8001
docker logs -f vllm-b70      # first start ~8-12 min; warm ~3.5-6 min
./status.sh                  # is it running?
./benchmarks/bench_vllm.sh 512 1
./benchmarks/bench_vllm.sh 512 3
```

Reference decode benchmark (realworld prompts):
`benchmarks/b70-realworld-context-harness.py --thinking`.

Aliases (after `source ~/.bashrc`): `b70-vllm`, `b70-vllm-logs`,
`b70-vllm-stop`, `b70-what`.

Tuning knobs are environment variables: `VLLM_MTP_N`, `VLLM_GPU_MEM_UTIL`,
`VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS`, `VLLM_MAX_BATCHED_TOKENS`,
`VLLM_KV_CACHE_MEMORY_BYTES`, `VLLM_DTYPE`, `VLLM_TEMPERATURE`, `VLLM_TOP_P`,
`VLLM_TOP_K`, `VLLM_EXTRA_ARGS`. `VLLM_MTP_N=0` disables speculation;
`VLLM_PREFIX_CACHING=0` disables prefix caching.

## Constraints

- Only one server owns the card and port; `serve_vllm.sh` replaces `vllm-b70`.
- 196608 + MTP10 fits only with the explicit 8.6 GiB KV pool and graph-estimate
  off. Auto-sizing at 0.94 gives ~6.75 GiB KV and fails the 196608 check.
- `--enforce-eager` drops decode to ~4 t/s; do not adopt it casually.
- `bench_vllm.sh` uses repeated filler and collapses MTP acceptance; use the
  realworld harness for MTP tuning.
- The opencode config prices local usage at OpenRouter Qwen3.8-27B rates for
  `opencode stats`; it is a shadow estimate, not a bill.

## Reference

This `AGENTS.md` is picked up automatically by coding agents. Everything else —
hardware, file map, measured performance, gotchas, migration, and client setup —
lives in **README.md**, the single source of truth.
