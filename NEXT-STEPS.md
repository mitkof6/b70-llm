# Next steps

State at 2026-10-25: upstream vLLM 0.32.0 (`b70-vllm-exp`) is the preferred
server on the shared API port **8001**. Tuned default is **MTP n=10 +
`--gpu-memory-utilization 0.94`**, prefix cache on, `fp8` KV, 8192 batched
tokens, `float16`, balanced scheduling, async scheduling off. Realworld p512
decode: **95.2 t/s** (median, thinking on) vs 81.8 at the old MTP4. See
`benchmarks/results/2026-10-25-mtp-tuning/report.md` and the older
`benchmarks/results/2026-10-25-v0.32.0-vs-pinned/report.md`.

## Resume

```bash
source ~/.bashrc
b70-vllm-exp                 # MTP10 + cache + 0.94, :8001
b70-vllm-exp-logs            # or b70-logs for whichever backend is live
python3 benchmarks/b70-realworld-context-harness.py --help   # --thinking
```

The tuning knobs are env vars: `VLLM_MTP_N_EXP` (exp build; legacy
`VLLM_MTP_N` belongs to the pinned script), `VLLM_GPU_MEM_UTIL`,
`VLLM_MAX_BATCHED_TOKENS`, `VLLM_DTYPE`, `VLLM_TEMPERATURE`/`VLLM_TOP_P`/
`VLLM_TOP_K`, `VLLM_EXTRA_ARGS`. The exp build now defaults to 196608 ctx and
`temperature 0.2` for coding (graph estimate off, explicit 8.6 GiB KV,
4096-token prefill chunks). The p512 harness
needs `--thinking`; the stock default renders 472 tokens and fails its check.

## Highest-value experiments

1. **MTP10 under sampling, not just greedy.** The harness sends
   `temperature=0`; agent clients hit the server default, now `temperature 0.2`
   (was 0.7). MTP acceptance still falls under sampling, shrinking the n=10
   gain. Run the p512 harness at the new 0.2 default (and at 0.7 for comparison)
   at n=4, n=8, n=10 and keep the best for interactive use. This is the biggest
   unknown in the current tuning.
2. **Real Claude Code workload.** Run a multi-turn edit/test task at n=10 and
   compare wall time and tok/s to n=4. Check tool-call parsing, reasoning
   parser, and any spec-decode edge cases with long tool schemas.
3. **n=9 vs n=10 across more prompt families.** The acceptance cliff between
   n=10 (85%) and n=11 (81%) may be prompt-dependent. The repo only has the p512
   set (5 measured prompts); add a code/tool-heavy prompt family and re-test.
4. **Long-context sweep.** 31k tokens measured ~76 t/s decode and ~24 s cold
   TTFT. Measure 64k and 128k, and try
   `--long-prefill-token-threshold-adaptive`,
   `--long-prefill-token-threshold`, and 4096/16384 batch budgets for TTFT.
   16384 did not help at 31k, but it may at 100k+.
5. **Concurrency with async scheduling.** C1 showed no effect; test
   `--async-scheduling` at 8–16 real prompts, where it is designed to fill GPU
   gaps. Also try `--max-num-active-seqs 8/16` to keep decode batches small.
6. **Sampling/accuracy check.** Speculative decoding should be exact, but verify
   n=10 vs `--no-spec` on a small deterministic eval (greedy and sampled), and
   check that `fp8` KV is not degrading quality on long contexts.

## Lower-value / cleanup

- `--dtype bfloat16` (`VLLM_DTYPE=bfloat16`) is wired but untested. Likely
  neutral on this GPTQ checkpoint; test only if accuracy work needs it.
- `--optimization-level 3` / `-O3` may buy a few percent at the cost of a long
  recompile; measure startup and decode before adopting.
- `--stream-interval 2/4` could cut host overhead in streaming; test with
  Claude Code (chunkier stream may hurt perceived latency).
- `--kv-cache-dtype fp8_inc` is Intel **Gaudi**-only, not XPU; the
  `turboquant_*`/`int8_per_token_head` variants are for capacity/accuracy, not
  expected decode speedups.
- Custom `-cc.cudagraph_capture_sizes` (e.g. focus on 1–16 for single-stream)
  could cut graph memory/startup; compare against the default 1…512 set.
- Remove the inert `B70_MTP_BF16_DRAFT=1` and `VLLM_XPU_ENABLE_XPU_GRAPH=1`
  env vars from `serve_vllm_exp.sh` (0.32.0 logs an unknown-env warning for the
  latter; the former is only read by the old cookbook patch).
- `bench_vllm.sh` filler prompts are intentionally low-acceptance; do not use
  them to tune MTP depth. The n=10 regression there (95 vs 138 aggregate at C3)
  is expected; real-prompt C3 is 156–160 t/s.

## Known constraints

- One backend owns the 32 GB card; every `serve_*` stops the others.
- MTP10 + graph-estimate off + explicit 8.6 GiB KV (`--kv-cache-memory-bytes`)
  fits 196608 ctx; vLLM reports 200,540 KV tokens at 1.02x. Auto-sizing at
  0.94 gave only 6.75 GiB (~145152 max), and 0.95 + graph off + 4096-token
  chunks was still 0.14 GiB short. 262144 needs MTP off and
  `VLLM_MAX_NUM_SEQS=1`.
- vLLM cold start ~8–12 min, warm ~3.5–6 min; changing MTP depth or memory
  utilization invalidates some compile-cache entries.
- MoE is GGUF-only; all vLLM aliases force the dense checkpoint.
