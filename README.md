# Understudy-CAD

**A 4B open model, post-trained on a Baseten H100, that turns a part description (or a drawing sheet) into
CadQuery code. Graded by the geometry itself: every answer is executed and its solid is compared to the
reference with exact volumetric IoU.**

Frontier models are strong generalists, but CAD-as-code is a narrow, unforgiving job: a wrong axis, a
misread convention or a hallucinated API call gives a part that crashes or is simply the wrong shape.
Understudy distills that one job into a small model we own, serves it on Baseten, and races it live against
Kimi K3 and GLM-5.3 on held-out parts.

## How it works

```
CAD-Coder (Apache-2.0)  ──► audit: run every reference, check it against its own spec ──► clean splits
      spec (text) or 4-view drawing sheet (rendered from the reference solid)               │
                                                                                            ▼
Baseten Training (H100): LoRA SFT on 7.4K verified pairs ──► optional GRPO, reward = IoU of the built solid
                                                                                            │
      checkpoint deploy (vLLM + LoRA) ◄──────────────────────────────────────────────────────┘
                   │
held-out parts ──► every lane, same prompt ──► sandboxed CadQuery ──► IoU vs reference ──► scoreboard + 3D race
```

| Step | Baseten product |
|---|---|
| Frontier baselines (Kimi K3, GLM-5.3, GLM-5.3 Flash) | Model APIs |
| Fine-tune Qwen3-4B-Instruct-2507 (text) / Qwen3-VL-4B (drawing sheets) | Training Jobs on 1× H100 |
| Serve our model | `baseten train checkpoint deploy` (vLLM + LoRA) |
| Built with | Baseten Switch (Claude Code on open models) |

## Look at your data: what the audit found

`scripts/audit_cadcoder.py` runs all ~16K reference programs and checks each against the size its own spec states.

- The **official test split's references disagree with their own spec 14% of the time** (single-part rows where
  the stated size is checkable) vs **2.3% for the curated `train_high` split**. So the benchmark is held out of
  `train_high`, grouped by source part (zero overlap with training).
- References apply the specs' global rotations/translations inconsistently. The headline metric therefore aligns
  the two solids first (best IoU over the 24 axis rotations after centering). It stays size- and shape-sensitive:
  a part 10% too large scores 0.75.
- Some specs can't determine their part ("repeat for the remaining seven faces" with no coordinates), and some
  reference programs wrote files to disk. The sandbox strips export calls and runs code in a throwaway directory.

## Benchmark rules

- **Same prompt for every lane.** The system prompt spells out the dataset's conventions (e.g. sketch coordinates
  are already final size) so no model loses on a gotcha. Frontier lanes additionally get two worked examples; ours
  gets the zero-shot prompt it was trained on.
- **Success = the code runs and aligned IoU ≥ 0.9.** Also reported: IoU ≥ 0.95, run rate, mean IoU, Chamfer
  distance, results by complexity (faces, parts), latency p50/p95, output tokens and $ per 1K parts.
- **Bootstrap 95% CIs**, fixed seeds, and API connection drops are retried (infrastructure failures aren't counted
  as model failures). Reasoning effort is stated for every frontier lane.
- **Cost includes the GPU.** Our $/1K parts is the H100's hourly price amortized over measured throughput.

## Results

_Filled in from `runs/<run>/summary.md` after training (see WORKLOG.md for the pilot numbers)._

## Quickstart

```bash
uv sync                                              # Python 3.12, CadQuery, trimesh, openai
cp .env.example .env                                 # add BASETEN_API_KEY
uv run python scripts/smoke.py                       # catalog, rate limit, one held-out part per lane
uv run python scripts/audit_cadcoder.py              # optional: re-run the data audit
uv run python -m understudy.cad.splits               # rebuild splits (deterministic)
uv run python -m understudy.cad.render --split bench # render drawing sheets (image mode)
uv run uvicorn app.server:app --port 8000            # 3D race UI
uv run pytest -q
```

Benchmark any lane (named lanes or any Model API slug as `slug:effort`):

```bash
uv run python -m understudy.cad.bench --lanes moonshotai/Kimi-K3:high,zai-org/GLM-5.3:high --n 200 --tag sota
uv run python -m understudy.cad.bench --modality image --lanes moonshotai/Kimi-K3:high --n 200 --tag sota-img
UNDERSTUDY_BASE_URL=... UNDERSTUDY_MODEL=checkpoint-... \
  uv run python -m understudy.cad.bench --lanes specialist,base-4b --n 500 --shots 0 --tag ours
```

## Training on Baseten

```bash
uv run python -m understudy.cad.build_sft      # training/data: 7,429 train / 152 val + 1,200 RL prompts
./training/dry_run.sh && ./training/dry_run.sh grpo   # CPU smoke tests with a tiny Qwen3 (free)
cd training && baseten train push --config config.py  # SFT, ~30-60 min on 1x H100
baseten train checkpoint deploy --job-id <job_id>     # then set UNDERSTUDY_BASE_URL / UNDERSTUDY_MODEL
SFT_JOB_ID=<job_id> baseten train push --config config_grpo.py   # optional RL stage
```

Drawing-sheet (vision) variant: `training/vlm/` and `deploy/` (see their READMEs).

## Layout

```
understudy/cad/       geometry.py (sandbox + IoU), pool.py (worker processes), data.py, splits.py,
                      prompts.py, bench.py (benchmark), render.py (drawing sheets), build_sft.py
understudy/llm.py     streaming client: TTFT, tokens, cost, retries (429/529, dropped streams)
training/             Baseten jobs: SFT (config.py, train.py), GRPO (config_grpo.py, grpo.py), dry_run.sh
app/                  FastAPI + three.js race UI (three.js vendored for offline use)
scripts/              audit, smoke test, mock API
data/cad/             bench (500), verified official-test subset, worked examples, audit summaries
```

Built at Hack the North 2026. Data: [CAD-Coder](https://huggingface.co/datasets/gudo7208/CAD-Coder) (Apache-2.0, derived from Text2CAD/DeepCAD).
