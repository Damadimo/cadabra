`data/cad/bench.jsonl` · 200 held-out drawing sheets · shots 0 · retries 0 · 2026-09-19T16:00:50

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| specialist | `checkpoint-600` (default) | **64.0%** [57.0%, 70.5%] | 60.5% | 99.0% | 0.803 | 0.0038 | 67.0% / 33.3% | 29.1 / 53.3 | 210 | $0.69 |

Costs: specialist: GPU $0.85/h amortized at concurrency 12
