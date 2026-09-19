`data/cad/bench.jsonl` · 40 held-out drawing sheets · shots 2 · retries 0 · 2026-09-19T05:09:24

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k3@high | `moonshotai/Kimi-K3` (high) | **22.5%** [10.0%, 35.0%] | 20.0% | 100.0% | 0.670 | 0.0058 | 23.5% / 16.7% | 29.7 / 120.1 | 2704 | $54.92 |
| glm-5.3-flash@high | `zai-org/GLM-5.3-Flash` (high) | **22.5%** [10.0%, 35.0%] | 17.5% | 97.5% | 0.598 | 0.0062 | 26.5% / 0.0% | 7.2 / 43.3 | 844 | $1.18 |

Costs: kimi-k3@high: list price per token; glm-5.3-flash@high: list price per token
