#!/usr/bin/env bash
# Article-style decode benchmark for a running vLLM server (warm, single request):
# client-side post-first-token decode tok/s, matching jonathanmann.tech's B70 table
# (p~512/g128, one warmup + N runs, median).
# NOTE: the repeated filler prompt below is deliberate (it reproduces that
# article's methodology) but collapses MTP acceptance to ~50%, so with MTP10 on
# these numbers run well below the harness headline (95.2 t/s on realistic
# prompts; see benchmarks/b70-realworld-context-harness.py).
# Usage: ./bench_vllm_stream.sh [gen_tokens] [runs] [think: on|off|both] [sampling: greedy|modelcard]
#   modelcard = Qwen3.8-27B recommended presets, paired with the thinking switch
#   as in the cookbook (thinking 1.0/0.95/20/0 | non-thinking 0.7/0.80/20/1.5).
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/../env.sh" || exit 1

MAXTOK="${1:-128}"
RUNS="${2:-5}"
THINK="${3:-both}"
SAMPLING="${4:-greedy}"
URL="http://127.0.0.1:${VLLM_PORT}"

python3 - "$URL" "$VLLM_SERVED_NAME" "$MAXTOK" "$RUNS" "$THINK" "$SAMPLING" <<'PY'
import sys, json, time, statistics, urllib.request

base, model = sys.argv[1], sys.argv[2]
maxtok, runs, think = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sampling = sys.argv[6]

def http(path, payload, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def ntok(text):
    return len(json.loads(http("/tokenize", {"model": model, "prompt": text}).read())["tokens"])

filler = ("In a production inference stack, kernel selection, memory bandwidth, and scheduling interact "
          "in ways that determine end-to-end throughput for large language models running on accelerators. ")
prompt = ""
while True:
    trial = prompt + filler
    if ntok(trial) > 512:
        break
    prompt = trial
print(f"prompt tokens: {ntok(prompt)} (target ~512), gen<={maxtok}, runs={runs}, think={think}, sampling={sampling}")

def one(enable_think):
    thinking_on = enable_think is not False
    if sampling == "modelcard":
        # Qwen3.8-27B model-card presets, paired with the mode (cookbook §10):
        # thinking 1.0/0.95/20/0.0 - non-thinking 0.7/0.80/20/1.5
        samp = ({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
                if thinking_on else
                {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "presence_penalty": 1.5})
    else:
        samp = {"temperature": 0.0}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxtok, "stream": True,
            "stream_options": {"include_usage": True}, **samp}
    if enable_think is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_think}
    t0 = time.time(); tf = None; usage = None; rc = 0
    with http("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                usage = d["usage"]
            for ch in (d.get("choices") or []):
                delta = ch.get("delta") or {}
                # vLLM streams thinking as delta.reasoning; some servers use reasoning_content
                if delta.get("reasoning") or delta.get("reasoning_content"):
                    rc += 1
                if tf is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                    tf = time.time()
    te = time.time()
    if tf is None:
        raise RuntimeError("no streamed tokens seen (empty response?)")
    ct = usage["completion_tokens"] if usage else maxtok
    return ct, tf - t0, te - t0, (ct - 1) / (te - tf), rc

cases = {"on": [("think on", None)], "off": [("think off", False)],
         "both": [("think off", False), ("think on", None)]}[think]
for label, val in cases:
    one(val)  # warmup (also warms prefix cache)
    res = [one(val) for _ in range(runs)]
    decs = [r[3] for r in res]; ttfts = [r[1] for r in res]; e2es = [r[2] for r in res]
    print(f"[{label}] median decode={statistics.median(decs):.2f} tok/s  (min {min(decs):.2f} / max {max(decs):.2f})")
    print(f"  median TTFT={statistics.median(ttfts):.3f}s  median end-to-end={statistics.median(e2es):.3f}s")
    print("  runs: " + ", ".join(f"{d:.2f}" for d in decs))
    print(f"  reasoning chunks per run: {[r[4] for r in res]}")
PY
