#!/usr/bin/env bash
# Benchmark a running llama.cpp server: per-request and aggregate tok/s.
# Usage: ./bench_vulkan.sh [max_output_tokens] [concurrency]
# NOTE: the repeated filler prompt below degrades MTP acceptance to ~50%
# (README: "Bench prompts matter hugely"). The llama.cpp numbers in the README
# table were measured with this same prompt, so results stay comparable to the
# table; realistic prompts would give higher MTP numbers.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh"

MAXTOK="${1:-256}"
CONC="${2:-1}"
URL="http://127.0.0.1:${LLAMA_PORT}/completion"

python3 - "$URL" "$MAXTOK" "$CONC" <<'PY'
import sys, json, time, urllib.request, concurrent.futures as cf
url, maxtok, conc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
prompt = "Write a detailed, well-structured explanation of how a CPU executes a program. " * 20

def one(_):
    body = json.dumps({"prompt": prompt, "n_predict": maxtok,
                       "temperature": 0.0, "cache_prompt": False}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    t = d.get("timings", {})
    n = t.get("predicted_n") or (d.get("tokens_predicted", 0))
    return n, time.time() - t0, t.get("predicted_per_second", 0)

t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=conc) as ex:
    res = list(ex.map(one, range(conc)))
wall = time.time() - t0
tok = sum(n for n, _, _ in res)
print(f"concurrency={conc}  max_tokens={maxtok}")
print("per-request decode t/s: " + ", ".join(f"{ps:.1f}" for _, _, ps in res))
print(f"aggregate: {tok} output tokens in {wall:.2f}s = {tok/wall:.1f} tok/s")
PY
