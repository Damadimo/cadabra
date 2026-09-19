# Understudy

**A 4B open model we train at the hackathon on a Baseten H100, racing Kimi K3 and GLM-5.3 on one messy real-world job: faster, far cheaper, and (the goal) at least as accurate on a held-out set we labeled ourselves.**

Frontier models are great generalists, but most production AI is one narrow job done millions of times. Understudy
takes a frontier model's judgment on a single task, keeps only the answers that pass deterministic checks, distills
them into a small model we own, serves it from a Baseten deployment, and falls back to the frontier model only when
the specialist's own checks fail.

Default task: **commercial-insurance submission triage**. Read a messy packet (broker emails, form excerpts,
statements of values, loss runs), extract the fields, resolve conflicting values, and decide accept / refer /
decline with cited guidelines. Fallback task: **security-log triage**. Adding a domain means adding one file in
`understudy/tasks/`.

## How it uses Baseten

| Step | Baseten product | What happens |
|---|---|---|
| Teacher labels | Model APIs: `zai-org/GLM-5.3`, adjudicated by `moonshotai/Kimi-K3` | Structured outputs, kept only when every deterministic check passes |
| Messy variants | Model APIs: `zai-org/GLM-5.3-Flash` | Cheap rewrites that keep every fact but change the mess |
| Training | Training Jobs on 1× H100 | LoRA SFT of `Qwen/Qwen3-4B`, answer-only loss (`training/`) |
| Serving | `baseten train checkpoint deploy` | vLLM serves the base model + our LoRA behind an OpenAI-compatible endpoint |
| Baselines | Model APIs | The same prompt and schema for every lane, so the race is fair |
| Building it | Baseten Switch | Claude Code routed to open models on Baseten |

## Pipeline

```
raw inputs ──► teacher (GLM-5.3) ──► deterministic checks ──► accepted labels ──► SFT data ──► Baseten Training (H100)
                    │ fail                                          ▲                                   │
                    ▼                                               │                                   ▼
               Kimi K3 adjudicates                         human corrections (UI)          checkpoint deploy (vLLM + LoRA)
                                                                                                         │
held-out gold set (hand-corrected, split by source, frozen) ──► evaluate: every lane, same prompt ◄──────┘
                                                                            │
                                                              race UI: live lanes + scoreboard
```

## Quickstart

```bash
uv sync                                   # Python 3.12 + deps
cp .env.example .env                      # add BASETEN_API_KEY
uv run python scripts/smoke.py            # key works, slugs are live, every lane answers (< $0.01)
uv run uvicorn app.server:app --port 8000 # race UI at http://127.0.0.1:8000
uv run pytest -q
```

No credits needed for UI work: run `uv run python scripts/mock_openai.py` and point `BASETEN_BASE_URL` and
`UNDERSTUDY_BASE_URL` at `http://127.0.0.1:8001/v1` (see `.env.example`). Mock numbers are fake.

## End-to-end run

```bash
# 1. Label: teacher + checks (+ Kimi K3 on failures), with cheap messy variants. Resumable.
uv run python -m understudy.teacher --inputs data/raw/inputs.jsonl --perturb 2 --rewrites 1

# 2. Build SFT data. Drops every source document that appears in the gold set.
uv run python -m understudy.build_sft --gold data/gold/gold.jsonl

# 3. Train on a Baseten H100 (the booth enables training access first).
#    ./training/dry_run.sh runs 2 CPU steps on a tiny Qwen3 first, to catch config and data errors for free.
./training/dry_run.sh
cd training && baseten train push --config config.py && cd ..
baseten train job logs --job-id <job_id> --tail

# 4. Deploy the LoRA checkpoint, then set UNDERSTUDY_BASE_URL / UNDERSTUDY_MODEL in .env
baseten train checkpoint deploy --job-id <job_id>

# 5. Evaluate every lane on the frozen gold set (then again at --concurrency 16)
uv run python -m understudy.evaluate --gold data/gold/gold.jsonl --lanes base-4b,specialist,cascade,kimi-k3,glm-5.3,glm-5.3-flash
```

## Evaluation rules we hold ourselves to

- **Ground truth is human-checked.** The gold set is teacher-prefilled, then corrected by hand, and frozen before the second training round.
- **No leakage.** Train/test splits are by source document, and `build_sft` drops any source that appears in the gold set.
- **Same prompt, same schema, every lane.** Frontier lanes also run at more than one reasoning effort, so speed gaps aren't just reasoning tokens.
- **Uncertainty is reported.** Accuracy comes with bootstrap 95% CIs; latency as p50/p95 at concurrency 1 and 16.
- **Cost includes the GPU.** The specialist's $/1K tasks is GPU $/hour amortized over measured throughput, not per-token pricing.
- **Dead ends are logged.** See [WORKLOG.md](WORKLOG.md).

## Results

_Filled in from `runs/<run>/summary.md` once the specialist is trained._

## Layout

```
understudy/          core library
  config.py          endpoints, pinned slugs, list prices, race lanes
  llm.py             streaming client: TTFT, tok/s, cost, retries on 429/529
  tasks/             one file per domain: schema, prompt, deterministic checks
  teacher.py         label with checks + adjudication (resumable)
  build_sft.py       labels + corrections -> TRL prompt/completion data
  evaluate.py        gold-set eval, bootstrap CIs, latency percentiles, $/1K
  cascade.py         specialist first, escalate when a check fails
training/            Baseten Training project (config.py, run.sh, train.py, deploy_config.py, dry_run.sh)
app/                 FastAPI + single-page race UI
scripts/             smoke test, mock OpenAI-compatible server
data/samples/        synthetic examples for wiring things up (not eval data)
```

Built at Hack the North 2026.
