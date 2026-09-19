`data/cad/bench.jsonl` · 200 held-out drawing sheets · shots 0 · retries 0 · 2026-09-19T16:00:50

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| specialist | `checkpoint-1000` (default) | **68.0%** [61.5%, 74.5%] | 65.0% | 97.5% | 0.842 | 0.0035 | 70.9% / 38.9% | 29.0 / 51.1 | 208 | $0.76 |

Costs: specialist: GPU $0.85/h amortized at concurrency 12
