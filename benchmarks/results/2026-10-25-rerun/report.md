# Re-run of the standard bench suite against the live `b70-vllm-exp` server

Date: 2026-10-25 (second run of the day). Server: upstream
`vllm/vllm-openai-xpu:v0.32.0`, container `vllm-b70-exp`, flags confirmed from
`docker inspect`: MTP n=10, prefix caching on, fp8 KV, 131072 ctx, 64 seqs,
8192 batched tokens, `float16`, mem-util 0.94 — the tuned default.

**Caveat:** the card was shared with live interactive use (Claude Code /
opencode sessions, sampled at temperature 0.7) during these runs. The
harness numbers below are unaffected in kind (unique entropy prefixes, cold
prefill, greedy), but wall-clock rates carry a small contention component.

## Headline: realworld p512 harness (thinking on, 128 out, 5 reps)

| Metric | This run | 2026-10-25 tuning report (n=10, mem .94) |
|---|---|---|
| median decode | **95.25 t/s** | 95.23 / 95.27 (repeat) |
| mean decode | 98.43 | 98.52 / 98.58 |
| min / max | 88.84 / 110.54 | 89.09 / 110.73 |
| MTP acceptance (harness window) | **85.1%** (579/680) | 85.1% |
| prefill input rate (median) | 1512 tok/s | 1653 tok/s |
| TTFT (median) | 0.34 s | ~0.31 s |
| cold-prefix contamination | none (hits delta 0) | — |

Per-rep decode: 95.25 / 110.54 / 102.48 / 95.07 / 88.84; per-rep acceptance
80.7 / 97.5 / 88.5 / 80.7 / 80.7%.

**The n=10 headline reproduces exactly** (95.2 vs 95.2–95.3, acceptance
85.1% = 85.1%). Prefill rate is ~9% below the dedicated-server figure,
consistent with the concurrent user load.

## Filler benches (repeated-sentence prompts; low MTP acceptance by design)

| Bench | This run | Earlier same-day numbers (n=10) |
|---|---|---|
| `bench_vllm.sh 512 1` | **39.7 t/s** | 57.5 warm / 38.1 cold (v0.32.0-vs-pinned report) |
| `bench_vllm.sh 512 3` | **95.2 aggregate** (31.7 / 33.3 / 33.8 per req) | ~95 aggregate, 32–34 per req (mtp-tuning report) |
| `bench_vllm_stream.sh 128 5 off greedy` | **38.13 t/s** median decode, TTFT 0.386 s | 54.40 t/s at n=4 (not re-run at n=10 before) |

The C3 filler result matches the recorded n=10 filler behavior exactly. The
C1 filler result matches the recorded **cold** figure (38.1), not the warm
57.5: live use evicted the filler prompt's prefix-cache entries, so this run
paid the full cold prefill. Not a regression.

## Live data point for the sampling question (NEXT-STEPS #1)

Server-wide MTP counters at the end of the run: 2,537 drafts, 25,350
proposed, 9,753 accepted = **38.5% cumulative acceptance** since server
start — vs 85.1% in the greedy harness window. The gap is the sampled
interactive traffic (temperature 0.7) plus the ~50%-acceptance filler runs
above. Directionally confirms the open hypothesis that MTP10's acceptance
drops substantially under sampling; a controlled n=4/8/10 sampling A/B is
still needed to size the effect.

## Verdict

No drift since this morning's tuning: the tuned n=10/0.94 config reproduces
its headline (95.2 t/s greedy p512, 85.1% acceptance) on a shared card.
Raw harness output: `harness-p512-think/results.json` (per-rep SSE files
included).
