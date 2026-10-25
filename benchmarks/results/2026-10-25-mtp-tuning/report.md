# vLLM 0.32.0 MTP tuning on the Arc Pro B70

Date: 2026-10-25. Dense Qwen3.8-27B GPTQ-Int4 MTP checkpoint, upstream
`vllm/vllm-openai-xpu:v0.32.0`, no cookbook patches. One server at a time.

Baseline flags: prefix caching on, `fp8` KV, 131072 ctx, 64 seqs, 8192 batched
tokens, `float16`, greedy. Metric: the realworld p512 harness (thinking on,
128 output, 5 measured prompts) — median decode of 5 runs. `bench_vllm*` filler
prompts are not used for tuning because their low MTP acceptance distorts the
result (see below).

## Results

| Config | median | mean | min | max | MTP acc | prefill tok/s |
|---|---|---|---|---|---|---|
| n=4, mem .92 (old default) | 81.85 | 84.22 | 81.68 | 88.05 | 93.9% | 1807 |
| n=4 + `--async-scheduling` | 81.98 | 84.22 | 81.68 | 88.06 | 93.9% | 1813 |
| n=4 + `--performance-mode interactivity` | 81.76 | 84.20 | 81.54 | 88.02 | 93.9% | 1829 |
| n=4 + `--performance-mode throughput` | 81.71 | 84.17 | 81.53 | 88.06 | 93.9% | 1992 |
| n=5 | 85.53 | 88.49 | 85.30 | 93.24 | 91.9% | 1869 |
| n=6 | 88.09 | 90.48 | 87.61 | 96.68 | 90.0% | 1804 |
| n=8, mem .92 | 90.26 | 93.74 | 90.21 | 102.16 | 86.6% | 1673 |
| **n=10, mem .94** | **95.23** | **98.52** | 89.09 | 110.73 | 85.1% | 1653 |
| **n=10, mem .94 (repeat)** | **95.27** | **98.58** | 89.01 | 110.78 | 85.1% | 1562 |
| n=11, mem .94 | 89.30 | 97.02 | 88.98 | 113.57 | 81.1% | 1509 |
| n=12, mem .95 | 90.08 | 94.88 | 89.93 | 106.44 | 80.0% | 1429 |
| n=10, mem .94, batched 16384 | 95.01 | 98.36 | 88.75 | 110.87 | 85.4% | 1626 |

n=10 is reproducible (+16.4% over n=4); n=11/12 regress. n=10 needs
`--gpu-memory-utilization 0.94` — at 0.92 the extra spec workspace leaves
6.14 GiB KV, below the 6.28 GiB a 131072-token slot needs.

`--async-scheduling`, `--performance-mode`, and a 16384-token batch budget are
within noise at C1. The prefill column is endpoint input tokens / client TTFT,
not isolated engine prefill.

## Concurrency

3 concurrent requests on three different p512 prompts (thinking on, 128 out):
**156–160 aggregate tok/s** at n=10, vs 138.5 on the n=4 filler benchmark.
The filler `bench_vllm.sh 512 3` run is the exception: n=10 drops it to ~95
aggregate (per-request 32–34) because the repeated-sentence prompt collapses
acceptance to ~50%; n=4 measured 138.5 there. Use n=4 for that synthetic
workload, n=10 for real prompts.

## Long context

31,285-token prompt, 256 forced output tokens, n=10:

- cold: TTFT 23.9–25.5 s (~1.3k prefill tok/s), decode 74–76 t/s
- warm (prefix cache): TTFT 6.5 s, decode ~76 t/s

No crash or KV failure at 131072 ctx. A 16384 batch budget did not improve
cold TTFT (25.5 s vs 23.9 s), so the default 8192 stays.

## Reproduce

```bash
# Start the tuned config (also the b70-vllm-exp default now):
LLM_MODEL_OVERRIDE=dense VLLM_MTP_N_EXP=10 VLLM_GPU_MEM_UTIL=0.94 \
  VLLM_PREFIX_CACHING=1 ./serve_vllm_exp.sh

# Harness (thinking is calibrated in; --thinking is required for p512):
python3 benchmarks/b70-realworld-context-harness.py \
  --mode scheduler --prompts benchmarks/prompts/qwen38-p512.json \
  --outdir /tmp/mtp10 --model qwen --budget 8192 --reps 5 --output 128 \
  --thinking
```

Raw `results.json` files are in this directory. The older n=4 A/B run against
the pinned 0.27.2rc1 build is in `../2026-10-25-v0.32.0-vs-pinned/`.
