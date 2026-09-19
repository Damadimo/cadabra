# Build plan: Sat Sep 19 → Sun Sep 20 (EDT)

## Hard deadlines

| When | What |
|---|---|
| **Sat 12:30 PM** | Initial Devpost submitted: final team, badge IDs exactly as printed, **every prize ticked** (hard cutoff 2:00 PM) |
| **Sun 1:00 AM** | Feature freeze |
| **Sun 8:00 AM** | Code freeze + final Devpost (repo must be public, video strongly recommended) |
| Sun 9:45–11:45 AM | Sponsor judging; round-1 slot may overlap, so anyone must be able to run the demo alone |

## Done overnight (see WORKLOG.md)

- Dataset audit, clean splits (7,583 train / 500 held-out bench / verified official-test subset), sandboxed geometry grader
- SOTA pilots on Baseten Model APIs (text and drawing-sheet inputs)
- SFT + GRPO jobs for the text model, both dry-run verified on CPU
- 3D race demo, benchmark runner, tests

## Direction: drawing sheets (decided 05:20)

Measured on held-out parts (WORKLOG.md): frontier models are 88–95% correct on explicit **text** specs (no room for a
strong delta), but on **drawing sheets** (4 rendered views + bounding box) Kimi K3 drops to 61% overall (200 parts) and
**22.5% on complex parts**. The headline model is Qwen3-VL-4B fine-tuned on ~30k rendered sheets. The text model (Qwen3-4B,
`training/`) is the fallback and a cost/latency story.

Three things only you can unblock (everything else is built, dry-run tested and committed):
1. **H100 training access** at the Baseten booth (`baseten train capacity describe` must show capacity; today: none).
2. **A payment method on the workspace** (Baseten → Billing). Model deploys are refused without one, and since 06:00
   **every Model API call returns `402: please check your current payment status`**: the free credit is used up
   (~$14 of list-price calls). This also blocks the live race's frontier lanes and the remaining baselines.
3. **Devpost** by 12:30 PM with the Baseten prize ticked, and the repo made public before judging.

## Runbook once unblocked

| Step | Command | Time |
|---|---|---|
| 0. Once Model APIs work again | `BASETEN_RPM=12 uv run python -m understudy.cad.bench --modality image --lanes moonshotai/Kimi-K3:high,zai-org/GLM-5.3-Flash:high --n 0 --shots 2 --exclude 20260919-051617_sota-img --tag sota-img-rest` (other 300 parts) | ~40 min |
| 1. Data (rebuilt after the renders finish) | `uv run python -m understudy.cad.build_vlm` (~30K rows) | 5 min |
| 2. Smoke run (20 steps) | set `MAX_STEPS=20` in `training/vlm/config_vlm.py`; `cd training/vlm && baseten train push --config config_vlm.py` | ~10 min |
| 3. Full run | set `MAX_STEPS=-1`; push again; `baseten train job logs --job-id <id> --tail` | ~2–3 h |
| 4. Deploy | `./scripts/deploy_vlm.sh <job_id> H100_40GB` (prints the .env lines) | 10–20 min |
| 4b. If the job stops early or `merged` is missing | `./scripts/deploy_vlm.sh <job_id> H100_40GB checkpoint-<N>` (base + LoRA, also serves the base lane) | 10–20 min |
| 5. Benchmark ours | `uv run python -m understudy.cad.bench --modality image --lanes specialist --n 500 --shots 0 --concurrency 16 --tag ours-img` | ~10 min |
| 6. Compare | `uv run python scripts/leaderboard.py --latest sota-img,sota-img-rest,ours-img --out-json data/demo/scoreboard.json`, then `uv run python scripts/plot_results.py` | instant |
| 7. Record demo races | race UI with `record: true` on 3–4 parts (replays are the offline fallback) | 15 min |
| Optional | base model lane: `baseten model push --dir deploy/vlm_base`, set `BASE_MODEL_URL`, rerun step 5 with `--lanes base-4b` | 20 min |
| Optional | text fallback: `cd training && baseten train push --config config.py` (~40 min), deploy with `baseten train checkpoint deploy` | 1 h |

## Questions for the Baseten booth

- H100 access for training (how many, how long)? Do training and deployments draw on our credits?
- Our Model API credit ran out (402 on every call). Can event credits cover Model APIs and deployments?
- Can our account be verified / rate limits raised? We're at 15 requests/min per model, which throttles the benchmark.
- Is Loops (RL SDK) available? It supports vision LoRA on Qwen3.5.
- Any issue serving a fine-tuned vision model (Qwen3-VL) with image input?
- Is the Baseten prize judged at the booth, on Devpost, or both?

## Demo (4:10 of a 5-minute slot)

1. **0:00–0:20.** Claim: "a 4B model we post-trained on Baseten builds CAD parts that frontier models get wrong, at a fraction of the cost."
2. **0:20–1:30.** Pick a held-out part. Three lanes race; the 3D viewers overlay each prediction on the target. Point at correct/wrong pills, time and $ per part.
3. **1:30–2:30.** Scoreboard: success with CIs by complexity, latency, $ / 1K parts; base model vs ours shows what training added.
4. **2:30–3:15.** The data story: the audit found 14% of the official test references wrong; the grader runs code and compares geometry, no LLM judge.
5. **3:15–4:10.** How it was built on Baseten: Model APIs for baselines, Training Jobs on H100, checkpoint deploy, Switch.

Prepared answers: memorized the test set? (held out by source part, zero overlap) · is the prompt unfair to frontier models?
(same conventions spelled out for all; frontier gets 2 worked examples, ours gets none) · why aligned IoU? (references
apply global transforms inconsistently; size and shape still count) · cost with an idle GPU? ($6.50/h; break-even volume).

## Pitfalls

- Deprecated Sep 25: Inkling, Kimi K2.6/K2.7-Code, GLM-4.7, DeepSeek-V4-Pro. We benchmark Kimi K3 and GLM-5.3 Flash
  (both take images); GLM-5.3's model card is text-only, so it is shown for reference only.
- Training jobs default to 1 CPU / 2 GiB unless set (configs set 12–14 CPUs); RL rewards need the cores.
- Never delete a training job holding undeployed checkpoints. Scale deployments to zero when idle; prewarm before judging.
- High-effort reasoning calls can take minutes (timeouts set to 30 min); connection drops are retried, and a row with
  no answer from the API at all (402/5xx) is left out of the score and listed, never counted as a wrong answer.
