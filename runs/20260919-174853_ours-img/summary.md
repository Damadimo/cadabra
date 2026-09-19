`data/cad/bench.jsonl` · 500 held-out drawing sheets · shots 0 · retries 0 · 2026-09-19T17:48:53

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| specialist | `cadabra-vl` (default) | **70.4%** [66.2%, 74.2%] | 65.0% | 98.4% | 0.863 | 0.0028 | 72.6% / 46.5% | 1.9 / 5.0 | 203 | $0.19 |

Costs: specialist: GPU $6.50/h amortized at concurrency 32
