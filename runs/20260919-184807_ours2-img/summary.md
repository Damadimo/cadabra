`data/cad/bench.jsonl` · 500 held-out drawing sheets · shots 0 · retries 0 · 2026-09-19T18:48:07

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| specialist | `cadabra-vl` (default) | **74.8%** [71.0%, 78.6%] | 71.0% | 99.8% | 0.887 | 0.0028 | 76.8% / 53.5% | 2.0 / 4.4 | 200 | $0.22 |

Costs: specialist: GPU $6.50/h amortized at concurrency 32
