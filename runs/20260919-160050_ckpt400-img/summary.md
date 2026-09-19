`data/cad/bench.jsonl` · 200 held-out drawing sheets · shots 0 · retries 0 · 2026-09-19T16:00:50

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| specialist | `checkpoint-400` (default) | **60.5%** [53.5%, 67.0%] | 60.0% | 98.5% | 0.760 | 0.0043 | 63.2% / 33.3% | 30.0 / 54.6 | 203 | $0.68 |

Costs: specialist: GPU $0.85/h amortized at concurrency 12
