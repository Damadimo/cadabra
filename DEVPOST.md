# Devpost draft: Understudy-CAD

_Fill the `[…]` placeholders from `runs/<ours>/summary.md` and `scripts/leaderboard.py` before submitting._

## Tagline
A 4B model we post-trained on Baseten reads engineering drawings into CAD code and beats frontier models at it,
graded by the geometry itself.

## Inspiration
Frontier models write good code, and in 2026 they turn explicit text specs into CAD almost perfectly: on our held-out
parts Kimi K3 got 95% right from a written spec. But real CAD work starts from drawings. When we gave the same parts
as a drawing sheet (front, top and right views plus an isometric), Kimi K3 fell to 61%, and to 26–32% on parts with
more than a handful of faces. That gap is a narrow, verifiable job: exactly what a small specialized model should own.

## What it does
- Takes a 4-view drawing sheet and the part's bounding box, writes CadQuery (Python) code, and builds the solid.
- Samples several programs, renders each one the way the input drawing was rendered, and keeps the best match
  (render-and-compare: it checks its own work against the drawing, never against an answer key).
- A live 3D race: our model vs Kimi K3 and GLM-5.3 Flash on held-out parts, each prediction overlaid on the target,
  with time, tokens and $ per part.

## How we built it (on Baseten)
- **Model APIs** for every frontier baseline (Kimi K3, GLM-5.3 Flash, GLM-5.3), all at high reasoning effort, all given
  two worked examples. Our model gets none.
- **Training Jobs on an H100**: LoRA SFT of Qwen3-VL-4B-Instruct on [~17k] rendered drawing sheets from CAD-Coder
  (Apache-2.0), adapters on the language model only, loss on the code only.
- **Deployment**: merged weights served by vLLM on Baseten straight from the training checkpoint (`bt://` weights).
- **Grader**: sandboxed CadQuery (import allowlist, no file IO, per-task process timeout) and exact OpenCascade
  boolean IoU against the reference solid, aligned over the 24 axis rotations. Success = code runs and IoU ≥ 0.9.
- **Baseten Switch** routed our Claude Code sessions to open models while we built.

## Results (held-out parts, never seen in training)
| Model | Correct overall | Medium parts (7–12 faces) | Complex (≥ 13) | $ / 1K parts | Latency p50 |
|---|---|---|---|---|---|
| **Understudy-CAD 4B (ours)** | [..%] | [..%] | [..%] | [$..] | [..s] |
| Kimi K3 (high) | 61.0% [55–68] | 26.1% | 31.6% | $37.04 | 12.2 s |
| GLM-5.3 Flash (high) | 65.0% [58–72] | 37.0% | 21.1% | $1.02 | 6.0 s |
| Untuned Qwen3-VL-4B | [..%] | [..%] | [..%] | | |

200 held-out parts for the frontier models, bootstrap 95% CIs. Our model was run on the same parts [and all 500].

## Challenges
- **The public data was wrong in places.** We ran all ~16K reference programs: 14% of the official test split's
  references contradict their own spec, some ignore the spec's rotations, and some even wrote files to disk. We
  rebuilt clean splits (held out by source part, deduplicated by geometry) and made the metric robust to the
  dataset's inconsistent placement.
- **Our first idea failed its own test.** We planned text-to-CAD, piloted the frontier models first, and found them at
  88–95%. We switched to drawing sheets before spending any GPU time.
- Rate limits (15 requests/min per model) and multi-minute reasoning calls meant building a resilient, resumable
  benchmark harness.

## Accomplishments
- A verifier that needs no answer key: render-and-compare separates correct from wrong frontier answers with AUC 0.96
  (689 answers) and, given several answers for one part, picks a correct one 98% of the time.
- [Our headline delta: e.g. "+X points over Kimi K3 on complex parts at 1/Y the cost"].
- Every number reproducible from the repo: `scripts/audit_cadcoder.py`, `understudy/cad/bench.py`, `scripts/leaderboard.py`.

## What we learned
- Pilot the frontier before you train. The task you expect them to fail is often already solved.
- Look at your data: the benchmark's labels were a bigger risk than the model.
- Small models plus a cheap verifier can afford test-time compute that a frontier model can't at the same price.

## What's next
- GRPO on the geometry reward (the pipeline is built and dry-run tested), real scanned drawings, dimension callouts
  instead of a bounding box, and assemblies.

## Built with
Baseten (Model APIs, Training, deployments, Switch) · Qwen3-VL · TRL · PEFT · vLLM · CadQuery / OpenCascade · three.js · FastAPI
