# Intel Arc Pro B70 inference workflow

Serving `Qwen3.8-27B` (INT4 / Q4) on an **Intel Arc Pro B70** and measuring
throughput. Three interchangeable backends, which cannot share the 32 GB card:

- **llama.cpp SYCL** (ggml `server-intel` Docker image) — fastest llama.cpp.
- **llama.cpp Vulkan** (LM Studio's bundled runtime) — fast, no extra image.
- **vLLM XPU** (`vllm/vllm-openai-xpu` image + cookbook MTP patches) — fastest
  (~78 t/s with MTP4 single-stream; ~67 t/s with prefix cache + v5, which also
  tolerates overlapping requests).

## Workflow

```bash
source ./env.sh          # or open a new shell (aliases load from bashrc)
./download_model.sh      # resumable; ~19 GB into models/ (MTP GPTQ checkpoint)
./serve_vllm.sh          # container 'vllm-b70': MTP4 + cookbook patches; waits
                         # for 'Application startup complete' on :8001
docker logs -f vllm-b70  # first start in a mode ~8-12 min; warm ~3.5-6 min
./bench_vllm.sh 512 1    # single-stream tok/s
./bench_vllm.sh 512 3    # concurrency 3
./bench_vllm_stream.sh 128 5 off greedy   # article-style single-stream bench
```

Reference decode benchmark (realworld prompts; full command in README §Measured
performance): `benchmarks/b70-realworld-context-harness.py`.

llama.cpp (faster single-stream; stops vLLM because only one fits the card):

```bash
./serve_sycl.sh           # SYCL + MTP n=2 (fastest) on http://127.0.0.1:8001
./serve_vulkan.sh         # Vulkan + MTP n=1 on http://127.0.0.1:8001
./bench_vulkan.sh 256 1   # single-stream decode tok/s
```

Aliases (after `source ~/.bashrc`): `b70-vllm-download`, `b70-vllm`,
`b70-vllm-claude` (vLLM, MTP off + prefix cache on), `b70-vllm-cache` (MTP4 +
cache without v5; overlap unsafe), `b70-vllm-claude-mtp` (MTP4 + cache + v5;
overlap-safe, ~67 t/s), `b70-vllm-bench`, `b70-vllm-logs`,
`b70-vllm-stop`, `b70-sycl`, `b70-sycl-logs`, `b70-sycl-stop`, `b70-vulkan`,
`b70-vulkan-bench`, `b70-vulkan-logs`, `b70-vulkan-stop`, `b70-what`.

## Reference

This `AGENTS.md` is picked up automatically by coding agents (there is no
`opencode.jsonc` in the folder root). Everything else — hardware, the file map,
measured performance, the model, gotchas, and how to point **opencode** /
**Claude Code** at the servers — lives in **README.md**, the single source of
truth. Log-line meanings: **LOGS.md**.
