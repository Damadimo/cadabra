# Devpost draft: Cadabra

_Fill the `[…]` placeholders from `runs/<ours>/summary.md` and `scripts/leaderboard.py` before submitting._

## Tagline
A 4B model we post-trained on Baseten reads engineering drawings into CAD code and beats frontier models at it,
graded by the geometry itself.

## Inspiration
Frontier models write good code, and in 2026 they turn explicit text specs into CAD almost perfectly: on our held-out
parts Kimi K3 got 95% right from a written spec. But real CAD work starts from drawings. When we gave the same parts
as a drawing sheet (front, top and right views plus an isometric), Kimi K3 fell to 62% and GLM-5.3 Flash to 66% on 500
held-out parts, and both to 19–37% on parts with more than six faces. That gap is a narrow, verifiable job: exactly
what a small specialized model should own.

## What it does
- Takes a 4-view drawing sheet and the part's bounding box, writes CadQuery (Python) code, and builds the solid.
- Samples several programs, renders each one the way the input drawing was rendered, and keeps the best match
  (render-and-compare: it checks its own work against the drawing, never against an answer key).
- A live 3D race: our model vs Kimi K3 and GLM-5.3 Flash on held-out parts, each prediction overlaid on the target,
  with time, tokens and $ per part. The scoreboard underneath has every number with 95% CIs, by part complexity.

## How we built it (on Baseten)
- **Model APIs** for the frontier baselines (Kimi K3 and GLM-5.3 Flash, both vision models), at high reasoning effort
  and with two worked examples each. Our model gets none.
- **Training Jobs on an H100**: LoRA SFT (rank 64) of Qwen3-VL-4B-Instruct on 30,260 rendered drawing sheets, two epochs (one H100, then four) from
  CAD-Coder (Apache-2.0), weighted toward the medium and complex parts where frontier models fail. Adapters on the
  language model only, loss on the code only, 1,024 visual tokens per sheet.
- **Deployment**: merged weights served by vLLM on Baseten straight from the training checkpoint (`bt://` weights);
  a second config serves base + LoRA from any intermediate checkpoint, which also gives us the untuned baseline.
- **Grader**: sandboxed CadQuery (import allowlist, no file IO, per-task process timeout) and exact OpenCascade
  boolean IoU against the reference solid, aligned over the 24 axis rotations. Success = code runs and IoU ≥ 0.9.
- **Baseten Switch** routed our Claude Code sessions to open models while we built.

## Results (held-out parts, never seen in training)
| Model | Correct overall | Medium parts (7–12 faces) | Complex (≥ 13) | $ / 1K parts | Latency p50 |
|---|---|---|---|---|---|
| **Cadabra 4B, best of 8 (ours)** | **80.7%** [77–84] | **61.3%** | **39.0%** | $1.19 | 2.3 s |
| **Cadabra 4B, 1 sample (ours)** | **74.8%** [71–79] | 52.1% | 25.4% | **$0.22** | **2.0 s** |
| GLM-5.3 Flash (high) | 66.2% [62–70] | 37.0% | 18.6% | $1.00 | 3.7 s |
| Kimi K3 (high) | 61.8% [58–66] | 30.3% | 18.6% | $36.82 | 10.4 s |
| Untuned Qwen3-VL-4B | 14.7% [11–18] | 8.4% | 1.7% | | |

[200/500] held-out parts, bootstrap 95% CIs. Rows where an API returned no answer (e.g. 402s) are left out and listed,
never scored as wrong answers.

## Challenges
- **The public data was wrong in places.** We ran all ~16K reference programs: 14% of the official test split's
  references contradict their own spec, some ignore the spec's rotations, and some even wrote files to disk. We
  rebuilt clean splits (held out by source part, deduplicated by geometry) and made the metric robust to the
  dataset's inconsistent placement.
- **Our first idea failed its own test.** We planned text-to-CAD, piloted the frontier models first, and found them at
  88–95%. We switched to drawing sheets before spending any GPU time.
- Rate limits (15 requests/min per model), multi-minute reasoning calls, and a credit limit that silently turned into
  402 errors mid-benchmark. We caught 32 failed API calls that had been scored as wrong answers, so now any row the API
  never answered is left out and listed, and a run stops at the first account-level error.

## Accomplishments
- A verifier that needs no answer key: render-and-compare separates correct from wrong frontier answers with AUC 0.96
  (689 answers) and, given several answers for one part, picks a correct one 98% of the time.
- **+14.5 points over the best frontier model** (80.7% vs 66.2%) at the same price, **+24.3 points on medium parts** and
  **+20.4 on complex ones**.
- One sample already beats both (74.8%) at **$0.22 per 1,000 parts: 4.5x cheaper than GLM-5.3 Flash and 167x cheaper
  than Kimi K3**, at 2.0 s per part vs 3.7 s and 10.4 s.
- Post-training works fast: after only 200 of 1,892 steps (11% of one epoch) our 4B model already tied Kimi K3 on the
  same parts (60.9% vs 60.9%), up from 13.2% for the untuned model.
- Our verifier makes any model better: best-of-4 with render-and-compare lifted Kimi K3 from 28% to 48% and GLM-5.3
  Flash from 32% to 46% on the hard parts. For K3 that costs $252 per 1,000 parts.
- Every number reproducible from the repo: `scripts/audit_cadcoder.py`, `cadabra/cad/bench.py`, `scripts/leaderboard.py`.

## What we learned
- Pilot the frontier before you train. The task you expect them to fail is often already solved.
- Look at your data: the benchmark's labels were a bigger risk than the model.
- Small models plus a cheap verifier can afford test-time compute that a frontier model can't at the same price.

## What's next
- GRPO on the geometry reward (the pipeline is built and dry-run tested), real scanned drawings, dimension callouts
  instead of a bounding box, and assemblies.

## Built with
Baseten (Model APIs, Training, deployments, Switch) · Qwen3-VL · TRL · PEFT · vLLM · CadQuery / OpenCascade · three.js · FastAPI
