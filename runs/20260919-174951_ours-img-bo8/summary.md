`data/cad/bench.jsonl` · 500 held-out drawing sheets · shots 0 · retries 0 · 2026-09-19T17:49:51

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| specialist | `understudy-cad-vl` (default) | **79.6%** [76.0%, 83.0%] | 72.8% | 100.0% | 0.916 | 0.0025 | 82.1% / 53.5% | 2.1 / 22.0 | 1625 | $1.23 |

Costs: specialist: GPU $6.50/h amortized at concurrency 8
