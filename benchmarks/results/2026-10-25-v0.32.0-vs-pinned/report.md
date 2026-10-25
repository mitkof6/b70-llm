# vLLM v0.32.0 (upstream) vs pinned 0.27.2rc1 + cookbook patches

Date: 2026-10-25. Single Intel Arc Pro B70, Qwen3.8-27B GPTQ-Int4 MTP checkpoint.
Same client, same prompts, same flags where possible. One server at a time.

> Follow-up: MTP depth was tuned later; the current default is MTP10 at
> `gpu-memory-utilization 0.94`, ~95 t/s. See
> `../2026-10-25-mtp-tuning/report.md`. This report keeps the n=4 numbers for
> the like-for-like comparison against the pinned build.

## Configurations

| | v0.32.0 (winner) | pinned 0.27.2rc1 |
|---|---|---|
| Image | `vllm/vllm-openai-xpu:v0.32.0` (sha256 `20509fab…`) | `vllm/vllm-openai-xpu@sha256:f01e24f6…` |
| vLLM / kernels | 0.32.0 / 0.1.15.4 | 0.27.2rc1.dev77 / 0.1.12.3 |
| Cookbook patches | none | mtp_nightly + boundary + 3× APC + GDN v5 split |
| MTP | n=4 | n=4 |
| Prefix caching | on | on |
| Mixed spec/non-spec split | not needed (upstream) | patch v5 |
| KV dtype / max len | fp8 / 131072 | fp8 / 131072 |
| max-num-seqs / batched tokens | 64 / 8192 | 64 / 8192 |
| gpu-memory-utilization | **0.92** (graph profiling reserves ~1 GiB) | 0.88 |

Both served the OpenAI model id `qwen`; benchmarks used greedy decoding and the
repo's default chat template. The realworld harness prompts are calibrated with
thinking **on** (512 rendered tokens); the stock harness uses the server default
(thinking off) and fails its token check, so both runs used a copy that passes
`chat_template_kwargs={"enable_thinking": true}` explicitly.

## Results

| Benchmark | v0.32.0 | pinned | delta |
|---|---|---|---|
| Realworld harness p512, 128 out, thinking on — **median decode** | **81.79 t/s** | 72.39 t/s | **+13.0%** |
| Realworld harness — mean / min / max | 84.18 / 81.70 / 87.95 | 72.11 / 70.49 / 73.02 | |
| Realworld harness — MTP draft acceptance | 511/544 = **93.9%** | 509/540 = 94.3% | equal |
| Realworld harness — prefill input rate (median) | **1,852 tok/s** | 1,361 tok/s | **+36%** |
| `bench_vllm.sh 512 1` — warm | **57.5 t/s** | 46.9 t/s | **+22.6%** |
| `bench_vllm.sh 512 1` — cold | 38.1 t/s | 34.6 t/s | +10.1% |
| `bench_vllm_stream.sh 128 5 off` — median | **54.40 t/s** | 45.91 t/s | **+18.5%** |
| `bench_vllm_stream.sh` — median TTFT | **0.307 s** | 0.344 s | -10.8% |
| `bench_vllm.sh 512 3` — aggregate | **138.5 t/s** | 92.3 t/s | **+50.1%** |
| `bench_vllm.sh 512 3` — per request | 47.1 / 46.2 / 47.7 | 30.8 / 32.7 / 32.9 | |
| Crash under overlap | none | none | |

Raw harness output: `exp-v0.32.0-harness.json`, `pinned-0.27.2rc1-harness.json`.

## Notes

- The repeated-filler benches (`bench_vllm*`) deliberately collapse MTP
  acceptance to ~50%, so they run below the realworld headline; they are still
  directly comparable between the two builds.
- The v0.32.0 run required `--gpu-memory-utilization 0.92` because 0.32.0
  profiles CUDA graphs by default (~1 GiB reservation); at 0.88 a 131072-token
  slot does not fit (4.94 GiB available vs 5.23 GiB needed).
- `B70_MTP_BF16_DRAFT=1` and `VLLM_XPU_ENABLE_XPU_GRAPH=1` are inert on 0.32.0
  (only the cookbook patch reads the former; the latter is now an unknown env
  var). They are harmless but not needed.
- Concurrency 3 (identical prompts, prefix-cache sharing) did not trigger the
  old `causal_conv1d` mixed spec/non-spec crash, so the v5 split patch is not
  required on 0.32.0 in this test.
- Long-context agent traffic during the session ran at ~69 ms/decode step with
  ~75% acceptance (~58 t/s active decode) at 40–76k context.
