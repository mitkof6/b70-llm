# Next steps

Current state: upstream vLLM 0.32.0 (`b70-vllm`) is the only server. The tuned
default is **MTP n=10 + prefix cache**, fp8 KV, 196608 context (graph estimate
off, explicit 8.6 GiB KV), 4096-token prefill chunks, and coding-friendly
sampling (`temperature 0.2`). Realworld p512 decode: **95.2 t/s** median
(thinking on). See `benchmarks/results/2026-10-25-mtp-tuning/report.md` and
`benchmarks/results/2026-10-25-rerun/report.md`.

## Resume

```bash
source ~/.bashrc
b70-vllm
b70-vllm-logs
python3 benchmarks/b70-realworld-context-harness.py --help   # --thinking
```

## Highest-value experiments

1. **MTP10 under sampling.** The harness uses `temperature=0`; agent clients
   hit the server default (now 0.2). Run p512 at n=4, n=8, and n=10 with
   sampling on and keep the best interactive setting.
2. **Real Claude Code workload.** Run a multi-turn edit/test task at n=10 and
   compare wall time and tok/s to n=4. Watch tool-call parsing, reasoning
   parsing, and long tool schemas.
3. **n=9 vs n=10 across prompt families.** The acceptance cliff between n=10
   (85%) and n=11 (81%) may be prompt-dependent; add code/tool-heavy prompts.
4. **Long-context sweep.** Measure 64k and 128k TTFT/decode; try
   `--long-prefill-token-threshold-adaptive`,
   `--long-prefill-token-threshold`, and 4096/16384 batch budgets.
5. **Concurrency with async scheduling.** C1 showed no effect; test
   `--async-scheduling` at 8–16 real prompts, and try
   `--max-num-active-seqs 8/16`.
6. **Sampling/accuracy check.** Verify n=10 vs `--no-spec` on a small
   deterministic eval and check fp8 KV quality on long contexts.

## Lower-value / cleanup

- `--dtype bfloat16` (`VLLM_DTYPE=bfloat16`) is wired but untested.
- `--optimization-level 3` may buy a few percent at the cost of a long
  recompile.
- `--stream-interval 2/4` could cut host overhead in streaming; test with
  Claude Code.
- Custom `-cc.cudagraph_capture_sizes` (e.g. 1–16 for single-stream) could cut
  graph memory/startup.

## Knobs

`VLLM_MTP_N`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_MODEL_LEN`,
`VLLM_MAX_NUM_SEQS`, `VLLM_MAX_BATCHED_TOKENS`, `VLLM_KV_CACHE_MEMORY_BYTES`,
`VLLM_DTYPE`, `VLLM_TEMPERATURE`/`VLLM_TOP_P`/`VLLM_TOP_K`, `VLLM_EXTRA_ARGS`.

## Constraints

- One server owns the 32 GB card; `serve_vllm.sh` replaces `vllm-b70`.
- MTP10 + graph-estimate off + explicit 8.6 GiB KV fits 196608; vLLM reports
  200,540 KV tokens at 1.02x. 262144 needs MTP off and `VLLM_MAX_NUM_SEQS=1`.
- vLLM cold start ~8–12 min, warm ~3.5–6 min; changing MTP depth or memory
  utilization invalidates some compile-cache entries.
