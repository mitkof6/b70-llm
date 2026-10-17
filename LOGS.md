# Reading the logs

Field guide for the server logs (kept out of the README to keep it lean).

## vLLM — `docker logs vllm-b70` (`b70-vllm-logs`)

Lines are prefixed with the component talking (`(APIServer pid=1)` vs
`(EngineCore pid=96)`).

**Startup — check these in order:**

| Line | Meaning |
| ---- | ------- |
| `non-default args: {...}` | effective serve config (model, MTP4, 0.88 util, fp8 KV, cache off) |
| `Checkpoint size: … Available RAM: …` | host-RAM feasibility check |
| `[B70] MTP draft: forcing unquantized build` | BF16-draft patch marker (good sign) |
| `Model loading took 17.38 GiB memory …` | weights → VRAM (device GiB) |
| `Available KV cache memory: …` / `GPU KV cache size: … tokens` | context-memory budget |
| `Graph capturing finished … took 3.95 GiB` | first start ~4 min, later ~2 min |
| `Application startup complete.` | ready |

**Runtime — two lines every ~10 s while requests run:**

```text
Engine 000: Avg prompt throughput: 6210.8 tokens/s, Avg generation throughput: 30.9 tokens/s,
Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 39.6%, Prefix cache hit rate: 0.0%
```

- `prompt throughput` — prefill speed (spikes on a new long message; 0 while generating).
- `generation throughput` — decode speed, the t/s you feel.
- `Running / Waiting` — scheduler queue; `Waiting > 0` with MTP on = concurrency risk (#510).
- `GPU KV cache usage` — how full the context pool is; pegged near 100% = contexts barely fit.
- `Prefix cache hit rate` — 0.0 because this bench config disables caching.

```text
SpecDecoding metrics: Mean acceptance length: 2.43, Accepted: 202 tokens, Drafted: 564 tokens,
Avg Draft acceptance rate: 35.8%
```

- MTP report card. `steps/s × mean acceptance length = t/s`: higher acceptance = faster.
- `Mean acceptance length` — tokens per forward pass (1 unconditional + accepted drafts).
- `Avg Draft acceptance rate` — verified/proposed; swings with how predictable the text is.

**Requests / failures:**

- `"POST /v1/chat/completions HTTP/1.1" 200 OK` — normal.
- `EngineCore encountered a fatal error` (+ `Dumping input data` + 500s) — engine
  died (the #510 MTP-overlap crash).
- `Add 3 padding layers …` — harmless.

The same counters are Prometheus metrics on `http://127.0.0.1:8001/metrics`
(`vllm:spec_decode_*`, `vllm:num_requests_*`, `vllm:kv_cache_usage_perc`).

## llama.cpp — `docker logs llama-sycl` / `tail -f llama-server.log`

No periodic metrics: `llama-server` prints one timing block per request, ending
with `eval time … tokens per second` (decode) and `prompt eval time …` (prefill).
For comparable numbers use `b70-vulkan-bench` / `b70-vllm-bench`.
