# Worklog

Honest, timestamped: what we tried, the numbers, and the dead ends. This becomes the Devpost story.

| Time (EDT) | What | Result |
|---|---|---|
| Sat 01:50 | Scaffolded the repo (Claude Code): streaming client, eval harness, race UI | tests pass |
| Sat 03:10 | Research: which post-training task can beat frontier models by a wide, verifiable margin? Compared computer use, CAD/Blender, hard math, chip design | Picked CAD-as-code: exact geometric grader, public data, text-only deploy path |
| Sat 04:15 | **Audit:** ran all ~16K CAD-Coder reference programs and checked each against the size its own spec states | Official test split: 14% of checkable references disagree with their spec (curated `train_high`: 2.3%). Some references ignore the spec's rotations; some write files to disk |
| Sat 04:25 | Built the grader: sandboxed CadQuery (import allowlist, no dunders, no file IO, per-task process timeout) + exact OCC boolean IoU, aligned over the 24 axis rotations | Self-IoU 1.0; +20% extrusion → 0.833; +10% scale → 0.751 |
| Sat 04:26 | Splits: 500 held-out parts from `train_high` grouped by source part; 7,583 train; 2 worked examples for frontier lanes | zero source overlap |
| Sat 04:28 | **Pilot, text specs** (40 held-out parts, same prompt with the dataset's conventions spelled out + 2 worked examples) | Kimi K3 (high) **87.5%** correct; GLM-5.3 (high) 60.0%, but 15 of its 16 misses were API timeouts/disconnects |
| Sat 04:40 | Looked at Kimi K3's 5 misses | 2 real (API misuse, merged two parts), 1 misled by a "cylindrical" name on a square prism, 2 unanswerable specs ("repeat for the remaining seven faces" with no coordinates). **Verdict: explicit text-to-CAD is nearly solved for frontier models; a strong accuracy delta isn't available there** |
| Sat 04:45 | Dead end avoided: the pilot's go/no-go gate (frontier ≤ ~60%) failed for text, so we did not spend GPU time on a text-only headline | kept the text model as a fallback (cost/latency story) |
| Sat 05:05 | Rendered every part as a 4-view drawing sheet (FRONT/TOP/RIGHT orthographic at one scale + isometric) with the bounding box stated in text | 500 sheets in 33 s |
| Sat 05:06 | **Pilot, drawing sheets** (same 40 parts) | Kimi K3 **62.5%** (vs 87.5% on text), GLM-5.3 Flash 67.5% |
| Sat 05:15 | **Pilot, complex drawing sheets** (40 held-out parts with ≥ 10 faces) | Kimi K3 **22.5%**, GLM-5.3 Flash **22.5%** → the drawing-sheet task is where a specialist can win |
| Sat 05:30 | Leakage hardening: dropped 1,886 training parts whose geometry (bbox + face count) matches a bench part; pinned worked examples | 5,698 text-train parts; +12,000 complex-weighted parts from `train_middle` for the sheet model |
| Sat 05:30 | Tried to validate vision-model serving on Baseten | blocked: workspace needs a payment method to deploy models; nothing created, $0 spent |
