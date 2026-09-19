`data/cad/bench.jsonl` · 2 held-out drawing sheets · shots 2 · retries 0 · 2026-09-19T05:04:17

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k3@high | `moonshotai/Kimi-K3` (high) | **100.0%** [100.0%, 100.0%] | 100.0% | 100.0% | 0.998 | 0.0091 | 100.0% / – | 4.4 / 6.1 | 181 | $18.12 |
| glm-5.3@high | `zai-org/GLM-5.3` (high) | **50.0%** [0.0%, 100.0%] | 50.0% | 100.0% | 0.676 | 0.0296 | 50.0% / – | 48.3 / 57.2 | 612 | $7.49 |
| glm-5.3-flash@high | `zai-org/GLM-5.3-Flash` (high) | **100.0%** [100.0%, 100.0%] | 100.0% | 100.0% | 1.000 | 0.0095 | 100.0% / – | 22.3 / 40.1 | 79 | $0.80 |

Costs: kimi-k3@high: list price per token; glm-5.3@high: list price per token; glm-5.3-flash@high: list price per token
