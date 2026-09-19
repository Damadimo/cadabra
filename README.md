# Cadabra

**A 4B vision model, post-trained on a Baseten H100, that reads an engineering drawing sheet and writes the CadQuery
program that builds the part. Graded by the geometry itself: every answer is executed and its solid is compared to
the reference with exact volumetric IoU.**

Frontier models turn explicit text specs into CAD almost perfectly (Kimi K3: 95% on our held-out parts). Real CAD
work starts from drawings, though, and from a 4-view drawing sheet the same models fall to 61–65%, and to 21–37% on
parts with more than six faces. Cadabra trains a small model on exactly that job, serves it on Baseten, and races
it live against Kimi K3 and GLM-5.3 Flash on held-out parts.

## How it works

```
CAD-Coder (Apache-2.0) ─► audit: run ~83K reference programs  ─► clean splits, held out by source part
        │                                                            and deduplicated by geometry
        ▼
render every part as a drawing sheet: FRONT / TOP / RIGHT at one scale + isometric, bounding box as text
        │
        ▼
Baseten Training: LoRA SFT of Qwen3-VL-4B-Instruct on 30,260 sheets -> CadQuery, 2 epochs (1x H100, then 4x H100)
        │
        ▼
Baseten deployment: merged weights served by vLLM straight from the training job (training_checkpoints)
        │
        ▼
held-out sheet ─► sample N programs ─► render each like the input sheet, keep the best match (no answer key)
               ─► sandboxed CadQuery ─► IoU vs reference ─► scoreboard + 3D race
```

| Step | Baseten product |
|---|---|
| Frontier baselines (Kimi K3, GLM-5.3 Flash, both vision models, high reasoning effort) | Model APIs |
| Fine-tune Qwen3-VL-4B-Instruct on rendered drawing sheets | Training Jobs: 2.9 h on 1× H100, then a second epoch on 4× H100 |
| Serve our model (merged weights, or base + LoRA from any checkpoint) | Deployments that pull the training job's checkpoints, vLLM |
| Built with | Baseten Switch (Claude Code on open models) |

## Look at your data: what the audit found

`scripts/audit_cadcoder.py` runs every reference program and checks it against the size its own spec states.

- The **official test split's references disagree with their own spec 14% of the time** (single-part rows where
  the stated size is checkable) vs **2.3% for the curated `train_high` split**. So the benchmark is held out of
  `train_high`, grouped by source part (zero overlap with training), and every training part whose geometry
  (bounding box + face count) matches a benchmark part is dropped.
- References apply the specs' global rotations/translations inconsistently. The headline metric therefore aligns
  the two solids first (best IoU over the 24 axis rotations after centering). It stays size- and shape-sensitive:
  a part 10% too large scores 0.75.
- Drawing sheets sidestep the noisy text: each sheet is rendered from its own reference program, so input and
  answer always agree.
- Some reference programs wrote files to disk. The sandbox strips export calls and runs code in a throwaway directory.

## Benchmark rules

- **Same task for every lane:** one drawing sheet plus the bounding box. Frontier lanes additionally get two worked
  examples (sheet + program) and high reasoning effort; ours gets the zero-shot prompt it was trained on.
- **Output budget:** frontier lanes may use up to 16,384 output tokens (reasoning included). Kimi K3 hit that cap on
  2 of 200 parts; counting both as correct would move it from 61.0% to 62.0%.
- **Success = the code runs and aligned IoU ≥ 0.9.** Also reported: IoU ≥ 0.95, run rate, mean IoU, Chamfer
  distance, results by complexity (faces, parts), latency p50/p95, output tokens and $ per 1K parts.
- **Bootstrap 95% CIs**, fixed seeds. If the API returns no answer at all (402, 5xx, dropped connection after
  retries) the row is left out and listed, never scored as a wrong answer.
- **Best-of-N without an answer key.** Our lane may sample several programs and keep the one whose rendering best
  matches the *input* sheet and the stated bounding box (render-and-compare, `cadabra/cad/verify.py`). On 689 real
  frontier answers it separates correct from wrong with AUC 0.96, and among several answers for the same part it
  picks a correct one 98% of the time (random: 76%). Frontier lanes can use the same verifier (`--best-of`).
- **Cost includes the GPU.** Our $/1K parts is the GPU's hourly price amortized over measured throughput.

## Results

Drawing sheets, held-out parts ([`data/demo/scoreboard.json`](data/demo/scoreboard.json), charts in `docs/`):

| Model | Correct overall [95% CI] | Simple (≤ 6 faces) | Medium (7–12) | Complex (≥ 13) | Multi-part | $ / 1K parts | Latency p50 |
|---|---|---|---|---|---|---|---|
| **Cadabra 4B, best of 8 (ours)** | **80.7% [77–84]** | **95.6%** | **61.3%** | **39.0%** | **58.1%** | $1.19 | 2.3 s (p95 14 s) |
| **Cadabra 4B, 1 sample (ours)** | **74.8% [71–79]** | 92.5% | 52.1% | 25.4% | 53.5% | **$0.22** | **2.0 s (p95 4.4 s)** |
| GLM-5.3 Flash (high) | 66.2% [62–70] | 85.9% | 37.0% | 18.6% | 39.5% | $1.00 | 3.7 s (p95 37 s) |
| Kimi K3 (high) | 61.8% [58–66] | 81.5% | 30.3% | 18.6% | 39.5% | $36.82 | 10.4 s (p95 89 s) |
| Untuned Qwen3-VL-4B | 14.7% [11–18] | 19.4% | 8.4% | 1.7% | 16.3% | | |

![results by complexity](docs/results_by_tier.png)

Every lane on the same 497 held-out parts (319 simple, 119 medium, 59 complex; 43 multi-part), one H100 for ours.
Best of 8 = eight samples ranked by render-and-compare, which needs no answer key; its oracle (any of the 8 correct)
is 83.7%, so the verifier captures 96% of what sampling made available. 99.8% of single-sample programs run, 100% of
best-of-8 ones.

**Learning curve** (greedy, the frontier's 200 parts): 13% untuned → 61% at step 200 → 68% at step 1,000 → 73.5% after
one epoch → **77.0% after two**, vs GLM-5.3 Flash 65% and Kimi K3 61%; on medium/complex parts **47.7%** vs 32% / 28%.

![learning curve](docs/learning_curve.png)

**The verifier helps any model.** Best-of-4 with render-and-compare on the 65 medium/complex parts: Kimi K3 28% → 48%
($252 per 1,000 parts), GLM-5.3 Flash 32% → 46%. Text specs for comparison: Kimi K3 95.0%,
GLM-5.3 87.5% (40 parts, zero-shot). The full timeline, dead ends included, is in [WORKLOG.md](WORKLOG.md).

## Is it overfitting, or memorizing the benchmark?

- **Validation loss fell to the last step of both epochs**: 0.212 → 0.158 in epoch 1, 0.160 → 0.149 in epoch 2, with
  token accuracy rising to 94.7%. Neither pass turned upward.
- **No benchmark part is in training**: the 500 benchmark parts are held out by source part (0 shared), and every
  training part with a benchmark part's exact geometry signature was dropped (1,886 of them).
- **Near-duplicates don't explain the result** (`scripts/leakage_check.py`). 57 of 500 benchmark parts have a training
  part within 1% on every bounding-box axis with the same face and part count; none within 0.1%. Those parts are easier
  for everyone, including models that never saw our data. On the **443 parts with no near-duplicate**:

  | | Near-duplicate (57) | No near-duplicate (443) |
  |---|---|---|
  | **Ours, best of 8** | 94.7% | **78.8%** |
  | **Ours, 1 sample** | 94.7% | 72.2% |
  | GLM-5.3 Flash (high) | 84.2% | 63.9% |
  | Kimi K3 (high) | 73.7% | 60.3% |

  Our margin over the best frontier model is **+14.9 points on the clean subset**, slightly wider than the +14.5 overall.

## The data

| | Parts | What |
|---|---|---|
| CAD-Coder splits audited | 82,659 | every reference program run and checked against its own spec (`train_high` 8,177, `train_middle` 66,532, `test` 7,950) |
| **Benchmark (held out)** | **500** | from the curated `train_high`, grouped by source part; 322 simple, 119 medium, 59 complex, 43 multi-part |
| Training, `train_high` | 5,698 | the rest of `train_high` after removing benchmark sources and benchmark geometries |
| Training, `train_middle` batch 1 | 12,000 | 70% medium/complex/multi-part |
| Training, `train_middle` batch 2 | 12,868 | every remaining medium/complex part, one per geometry signature |
| **Drawing sheets used for training** | **30,260** (+306 validation) | 72% medium/complex, 35% multi-part, 549 MB of renders |
| RL prompts | 2,000 | 80% medium/complex, same held-out rules |

Each sheet is rendered from the part's own reference program, so the drawing and the answer always agree even where
the dataset's text specs are wrong. Training targets are the reference programs with trailing display comments removed;
loss is on the code only.

## Demo

```sh
uv run python -m uvicorn app.server:app --port 8000     # http://127.0.0.1:8000
uv run python deploy/test_vlm.py                        # prewarm the endpoint before judging (it scales to zero)
```

![demo](docs/demo.png)

- **Compare** (offline): pick any held-out part and see the drawing sheet, the dataset's reference solid, and each
  model's answer built and overlaid on it, with IoU, time, cost and the code. Answers come from the saved benchmark
  runs, so this works with no API calls. Filter to the parts both frontier models get wrong.
- **Live race**: the same part sent to our deployment and to Kimi K3 and GLM-5.3 Flash at once, streaming, then built
  and scored. Our lane samples 8 programs and keeps the best render-and-compare match.

## Quickstart

```bash
uv sync                                              # Python 3.12, CadQuery, VTK, openai
cp .env.example .env                                 # add BASETEN_API_KEY
uv run python scripts/smoke.py                       # catalog, rate limit, one held-out part per lane
uv run python scripts/audit_cadcoder.py              # optional: re-run the data audit
uv run python -m cadabra.cad.splits               # rebuild splits (deterministic); --extra2 for the second batch
uv run python -m cadabra.cad.render --split bench # render drawing sheets
uv run uvicorn app.server:app --port 8000            # 3D race UI
uv run pytest -q
```

Benchmark any lane (named lanes, or any Model API slug as `slug:effort`):

```bash
uv run python -m cadabra.cad.bench --modality image --lanes moonshotai/Kimi-K3:high,zai-org/GLM-5.3-Flash:high --n 200 --tag sota-img
CADABRA_BASE_URL=... uv run python -m cadabra.cad.bench --modality image --lanes specialist --n 500 --shots 0 --tag ours-img
uv run python -m cadabra.cad.bench ... --best-of 8           # sample 8, keep the best render-and-compare match
uv run python scripts/leaderboard.py --latest sota-img,ours-img --out-json data/demo/scoreboard.json
uv run python scripts/plot_results.py                          # docs/results_by_tier.png, docs/cost_vs_accuracy.png
```

## Training and serving on Baseten

```bash
uv run python -m cadabra.cad.render --split train_vlm_extra   # (and train, train_vlm_extra2) sheets for training
uv run python -m cadabra.cad.build_vlm                          # training/vlm/data: rows + images
./training/vlm/dry_run_vlm.sh                                      # 2 CPU steps with a tiny Qwen3-VL (free)
cd training/vlm && baseten train push --config config_vlm.py       # LoRA SFT on 1x H100
./scripts/deploy_vlm.sh <job_id> H100_40GB                         # merged weights -> vLLM; prints the .env lines
./scripts/deploy_vlm.sh <job_id> H100_40GB checkpoint-<N>          # or base + LoRA from any checkpoint
```

The text-spec model (Qwen3-4B, `training/`, SFT + GRPO on the IoU reward) is kept as a fallback and a cost story.

## Layout

```
cadabra/cad/       geometry.py (sandbox + IoU), pool.py (worker processes), data.py, splits.py, prompts.py,
                      render.py (drawing sheets), verify.py (render-and-compare), bench.py, build_vlm.py, build_sft.py
cadabra/llm.py     streaming client: TTFT, tokens, cost, retries (429/5xx, dropped streams)
training/vlm/         Baseten job for the drawing-sheet model (config_vlm.py, train_vlm.py, dry_run_vlm.sh)
training/             text model: SFT (config.py, train.py), GRPO (config_grpo.py, grpo.py), dry_run.sh
deploy/               vLLM configs: base model, merged fine-tune, base + LoRA
app/                  FastAPI + three.js race UI (three.js vendored for offline use)
scripts/              audit, leaderboard, charts, verifier validation, deploy, smoke test, mock API
data/cad/             bench (500), worked examples, split stats, audit summaries
runs/                 every benchmark run: results.jsonl (per part, with the code), summary.md
```

Built at Hack the North 2026. Data: [CAD-Coder](https://huggingface.co/datasets/gudo7208/CAD-Coder) (Apache-2.0, derived from Text2CAD/DeepCAD).
