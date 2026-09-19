`data/cad/bench.jsonl` · 65 held-out drawing sheets · shots 2 · retries 0 · 2026-09-19T14:44:35

| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k3@high | `moonshotai/Kimi-K3` (high) | **48.3%** [35.0%, 61.7%] | 41.7% | 100.0% | 0.778 | 0.0048 | 52.1% / 33.3% | 51.8 / 122.2 | 15087 | $251.67 |
| glm-5.3-flash@high | `zai-org/GLM-5.3-Flash` (high) | **45.9%** [32.8%, 57.4%] | 39.3% | 100.0% | 0.751 | 0.0053 | 49.0% / 33.3% | 11.2 / 59.5 | 3553 | $4.81 |

Costs: kimi-k3@high: list price per token; glm-5.3-flash@high: list price per token

Left out (the API returned no answer, e.g. 402/5xx/disconnect): kimi-k3@high 5, glm-5.3-flash@high 4
