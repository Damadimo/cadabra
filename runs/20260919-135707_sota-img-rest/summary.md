`data/cad/bench.jsonl` · 300 held-out drawing sheets · shots 2 · retries 0 · 2026-09-19T13:57:07

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k3@high | `moonshotai/Kimi-K3` (high) | **62.3%** [56.7%, 67.3%] | 58.7% | 95.7% | 0.785 | 0.0048 | 64.0% / 44.0% | 9.1 / 89.4 | 1434 | $36.67 |
| glm-5.3-flash@high | `zai-org/GLM-5.3-Flash` (high) | **67.0%** [61.7%, 72.0%] | 63.7% | 99.3% | 0.821 | 0.0050 | 69.1% / 44.0% | 3.0 / 36.9 | 463 | $0.99 |

Costs: kimi-k3@high: list price per token; glm-5.3-flash@high: list price per token
