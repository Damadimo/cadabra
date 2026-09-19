`data/cad/bench.jsonl` · 200 held-out drawing sheets · shots 0 · retries 0 · 2026-09-19T14:56:17

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| specialist | `understudy-cad-vl` (default) | **61.0%** [54.0%, 67.5%] | 57.5% | 95.0% | 0.723 | 0.0040 | 63.7% / 33.3% | 19.5 / 55.4 | 230 | $2.62 |
| base-4b | `Qwen/Qwen3-VL-4B-Instruct` (default) | **13.2%** [9.1%, 17.8%] | 12.7% | 34.5% | 0.200 | 0.0065 | 13.4% / 11.1% | 18.0 / 200.4 | 601 | $2.66 |

Costs: specialist: GPU $0.85/h amortized at concurrency 16; base-4b: GPU $0.85/h amortized at concurrency 16

Left out (the API returned no answer, e.g. 402/5xx/disconnect): base-4b 3
