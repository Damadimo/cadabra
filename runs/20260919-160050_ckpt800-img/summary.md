`data/cad/bench.jsonl` · 200 held-out drawing sheets · shots 0 · retries 0 · 2026-09-19T16:00:50

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| specialist | `checkpoint-800` (default) | **64.0%** [57.0%, 70.0%] | 60.5% | 98.0% | 0.815 | 0.0039 | 66.5% / 38.9% | 29.3 / 52.3 | 202 | $0.76 |

Costs: specialist: GPU $0.85/h amortized at concurrency 12
