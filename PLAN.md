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

## The one decision still open: text specs or drawing sheets

Measured on held-out parts (WORKLOG.md has the numbers): frontier models are strong on explicit **text** specs, so the
text model's win is mostly cost/latency. **Drawing sheets** (4 rendered views + bounding box) are where frontier models
struggle. Take the drawing-sheet (vision) route only if both gates pass:

1. Kimi K3's success on sheets (complex parts) is clearly below what a trained model can plausibly reach.
2. A fine-tuned Qwen3-VL-4B can be served on Baseten with image input (`deploy/README.md`).

Otherwise ship the text model and lead with cost/latency plus the GLM-5.3 comparison.

## Timeline from when the booth opens

| Time | Work | Done when |
|---|---|---|
| Booth opens | Ask for H100 access + the questions below; `baseten train capacity describe` shows capacity | Capacity > 0 |
| +0:10 | Text SFT smoke run: `MAX_STEPS=50` in training/config.py, push, watch logs (~5 min) | Job completes, checkpoint synced |
| +0:20 | Full text SFT (2 epochs, ~30–60 min). In parallel on a 2nd GPU if granted: VLM SFT (`training/vlm/`) | Checkpoints synced |
| +1:30 | Deploy the checkpoint (`baseten train checkpoint deploy`), set UNDERSTUDY_BASE_URL/MODEL in .env, `scripts/smoke.py` | Endpoint answers |
| +1:45 | Benchmark ours on all 500 held-out parts: `--lanes specialist,base-4b --shots 0` (fast, no rate limit) | runs/…_ours |
| +2:15 | Error analysis on failures → optional GRPO (`config_grpo.py`, ~1–2 h) or a second SFT round | Decision logged |
| 12:30 PM | **Submit the initial Devpost with all prizes ticked** | Submitted |
| Afternoon | Final SOTA benchmark at scale (200–500 parts per frontier lane; rate-limited, run in background) | Final table |
| Evening | Demo polish, pick demo parts (data/cad/demo.json), video, README results | Freeze at 1 AM |

## Questions for the Baseten booth

- H100 access for training (how many, how long)? Do training and deployments draw on our credits?
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

- Deprecated Sep 25: Inkling, Kimi K2.6/K2.7-Code, GLM-4.7, DeepSeek-V4-Pro. We only benchmark Kimi K3, GLM-5.3, GLM-5.3 Flash.
- Training jobs default to 1 CPU / 2 GiB unless set (configs set 12–14 CPUs); RL rewards need the cores.
- Never delete a training job holding undeployed checkpoints. Scale deployments to zero when idle; prewarm before judging.
- High-effort reasoning calls can take minutes (timeouts set to 30 min); connection drops are retried, not scored as failures.
