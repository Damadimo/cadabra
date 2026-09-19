`data/cad/bench.jsonl` · 40 held-out drawing sheets · shots 2 · retries 0 · 2026-09-19T05:05:47

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k3@high | `moonshotai/Kimi-K3` (high) | **62.5%** [47.5%, 77.5%] | 57.5% | 97.5% | 0.778 | 0.0050 | 62.2% / 66.7% | 11.0 / 49.1 | 1187 | $33.19 |
| glm-5.3-flash@high | `zai-org/GLM-5.3-Flash` (high) | **67.5%** [52.5%, 82.5%] | 67.5% | 100.0% | 0.850 | 0.0057 | 64.9% / 100.0% | 3.3 / 36.4 | 580 | $1.05 |

Costs: kimi-k3@high: list price per token; glm-5.3-flash@high: list price per token
