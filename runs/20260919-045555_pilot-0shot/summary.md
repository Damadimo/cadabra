`data/cad/bench.jsonl` · 40 held-out specs · shots 0 · retries 0 · 2026-09-19T04:55:55

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k3@high | `moonshotai/Kimi-K3` (high) | **95.0%** [87.5%, 100.0%] | 95.0% | 97.5% | 0.949 | 0.0000 | 94.6% / 100.0% | 7.2 / 37.2 | 1229 | $20.51 |
| glm-5.3@high | `zai-org/GLM-5.3` (high) | **87.5%** [77.5%, 97.5%] | 87.5% | 92.5% | 0.876 | 0.0000 | 86.5% / 100.0% | 63.4 / 633.6 | 1507 | $7.29 |

Costs: kimi-k3@high: list price per token; glm-5.3@high: list price per token
