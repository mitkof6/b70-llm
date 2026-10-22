import json, random, string, time, urllib.request

URL = "http://127.0.0.1:8001/v1/chat/completions"
METRICS = "http://127.0.0.1:8001/metrics"


def metric(name):
    with urllib.request.urlopen(METRICS, timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith(name + "{"):
                return float(line.rsplit(" ", 1)[1])
    return 0.0


def request(messages, max_tokens=1):
    body = json.dumps({
        "model": "qwen",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        payload = json.loads(r.read())
    return time.perf_counter() - t0, payload["usage"]


rng = random.Random(7)
words = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 9))) for _ in range(9000)]
filler = " ".join(words)
messages = [
    {"role": "system", "content": "You are a careful assistant. Answer with one word."},
    {"role": "user", "content": "Read this identifier list and reply OK.\n\n" + filler},
]

for i in (1, 2, 3):
    h0, q0 = metric("vllm:prefix_cache_hits_total"), metric("vllm:prefix_cache_queries_total")
    dt, usage = request(messages)
    h1, q1 = metric("vllm:prefix_cache_hits_total"), metric("vllm:prefix_cache_queries_total")
    hit = (h1 - h0) / (q1 - q0) if q1 > q0 else float("nan")
    print(f"run{i}: ttft+1tok={dt:.2f}s prompt_tokens={usage['prompt_tokens']} "
          f"cached_delta={h1 - h0:.0f}/{q1 - q0:.0f} hit_rate={hit:.1%}")
