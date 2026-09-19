`data/cad/bench.jsonl` · 200 held-out drawing sheets · shots 2 · retries 0 · 2026-09-19T05:16:17

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k3@high | `moonshotai/Kimi-K3` (high) | **61.0%** [54.5%, 68.0%] | 56.5% | 96.0% | 0.781 | 0.0048 | 62.6% / 38.5% | 12.2 / 62.3 | 1490 | $37.04 |
| glm-5.3-flash@high | `zai-org/GLM-5.3-Flash` (high) | **65.0%** [58.5%, 71.5%] | 62.0% | 97.5% | 0.827 | 0.0047 | 66.8% / 38.5% | 6.0 / 58.2 | 510 | $1.02 |
| glm-5.3@high | `zai-org/GLM-5.3` (high) | **17.0%** [12.0%, 22.0%] | 16.5% | 72.0% | 0.370 | 0.0087 | 17.6% / 7.7% | 31.8 / 200.2 | 688 | $6.63 |

Costs: kimi-k3@high: list price per token; glm-5.3-flash@high: list price per token; glm-5.3@high: list price per token
