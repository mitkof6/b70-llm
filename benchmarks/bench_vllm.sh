#!/usr/bin/env bash
# Benchmark a running vLLM server: per-request and aggregate tok/s.
# Usage: ./bench_vllm.sh [max_output_tokens] [concurrency]
# NOTE: the repeated filler prompt below collapses MTP acceptance to ~50%
# (README: "Bench prompts matter hugely"), so with MTP10 on, these numbers run
# well below the harness headline (95.2 t/s on realistic prompts). For
# comparable headline numbers use benchmarks/b70-realworld-context-harness.py
# with prompts/qwen38-p512.json.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/../env.sh" || exit 1

MAXTOK="${1:-512}"
CONC="${2:-1}"
URL="http://127.0.0.1:${VLLM_PORT}/v1/chat/completions"

python3 - "$URL" "$VLLM_SERVED_NAME" "$MAXTOK" "$CONC" <<'PY'
import sys, json, time, urllib.request, concurrent.futures as cf
url, model, maxtok, conc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prompt = "Write a detailed, well-structured explanation of how a CPU executes a program. " * 20

def one(_):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": maxtok,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d["usage"]["completion_tokens"], time.time() - t0

t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=conc) as ex:
    res = list(ex.map(one, range(conc)))
wall = time.time() - t0
tok = sum(t for t, _ in res)
print(f"concurrency={conc}  max_tokens={maxtok}")
print("per-request tok/s: " + ", ".join(f"{t/dt:.1f}" for t, dt in res))
print(f"aggregate: {tok} output tokens in {wall:.2f}s = {tok/wall:.1f} tok/s")
PY
