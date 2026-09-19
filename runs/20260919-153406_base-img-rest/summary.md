`data/cad/bench.jsonl` · 300 held-out drawing sheets · shots 0 · retries 0 · 2026-09-19T15:34:06

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| base-4b | `Qwen/Qwen3-VL-4B-Instruct` (default) | **15.7%** [11.7%, 20.0%] | 15.0% | 34.0% | 0.202 | 0.0062 | 15.3% / 20.0% | 22.4 / 156.4 | 675 | $0.53 |

Costs: base-4b: GPU $0.85/h amortized at concurrency 24
