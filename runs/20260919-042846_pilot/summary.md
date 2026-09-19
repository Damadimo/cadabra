`data/cad/bench.jsonl` · 40 held-out specs · shots 2 · retries 0 · 2026-09-19T04:28:46

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k3@high | `moonshotai/Kimi-K3` (high) | **87.5%** [77.5%, 97.5%] | 87.5% | 95.0% | 0.876 | 0.0000 | 86.5% / 100.0% | 3.8 / 34.8 | 755 | $17.93 |
| glm-5.3@high | `zai-org/GLM-5.3` (high) | **60.0%** [45.0%, 75.0%] | 60.0% | 62.5% | 0.611 | 0.0000 | 59.5% / 66.7% | 16.6 / 621.3 | 303 | $2.87 |

Costs: kimi-k3@high: list price per token; glm-5.3@high: list price per token
