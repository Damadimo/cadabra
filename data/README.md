# Data

| Path | In git? | Contents |
|---|---|---|
| `samples/` | yes | Small synthetic examples for wiring and tests. **Not** eval data. |
| `raw/` | no | Sponsor data as pulled: `{"id", "source_id", "text"}` per line (seclogs may use `"lines"`) |
| `labels/` | no | `understudy.teacher` output: record, status, label, per-check results, cost |
| `gold/gold.jsonl` | decide by licence | Hand-corrected held-out set: `{"id", "source_id", "text", "gold": {...}}`. Freeze it before training v2 |
| `corrections.jsonl` | yes | Field corrections typed into the race UI; `build_sft` upweights them |
| `demo/<task>.jsonl` | yes | Optional curated demo inputs for the UI dropdown (`{"id", "title", "text"}`) |

`source_id` ties every variant (perturbation, rewrite) to its original document, so splits and the leakage guard
work at the document level. The `gold` block holds the task's score fields plus the decision field.
