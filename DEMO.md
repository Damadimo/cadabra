# Demo script (Baseten track)

Target: 5 minutes of demo, then questions. Numbers below are the ones on screen.

## 0. Setup, before judges arrive
- `cd ~/cadabra && CAD_WORKERS=4 uv run python -m uvicorn app.server:app --port 8000`
- Warm the endpoint, it scales to zero: `set -a && . ./.env && set +a && uv run python deploy/test_vlm.py`
- Open `http://127.0.0.1:8000` on the Compare tab, leave it on the default part
- Have a second tab on the GitHub repo and `paper/cadabra.pdf`
- Check the wifi, the live race calls Baseten Model APIs

## 1. Open, 20 seconds
- "We post-trained a 4B model on Baseten to read engineering drawings and write the CAD program that builds the part."
- "It beats Kimi K3 and GLM-5.3 Flash on held-out parts, at a fraction of the cost."
- "Everything you see is graded by geometry, not by a language model."

## 2. The problem, 30 seconds
- Frontier models are already good at CAD from a written spec, Kimi K3 gets 95% on our parts that way
- Give them the same part as a drawing instead and K3 drops to 62%, GLM-5.3 Flash to 66%
- Real CAD work starts from drawings, so that gap is the job
- It is narrow and it is verifiable, which is exactly what a small specialised model should own

## 3. Compare view, 60 seconds
- Left tile is the input: a four view drawing sheet plus the bounding box, nothing else
- Right tile is the solid our model's program actually builds, with the reference ghosted over it in red
- Red only is volume the model missed, green only is volume it invented, that is what IoU counts
- Switch the dropdown to GLM-5.3 Flash on the same part: its bars run way past the red reference, IoU 0.49
- Hit Program to show the CadQuery code, and Parts to filter to the parts we get right and both frontier models get wrong

## 4. Live race, 60 seconds
- Switch to Live race, pick a part, hit Run
- All three models get the same sheet and bounding box at the same moment
- Ours returns in about 2 seconds, Flash takes about 4 and K3 about 10
- Each answer is executed in a sandbox and compared to the reference solid while you watch
- Point out ours samples 8 programs and picks the best match to the drawing, with no answer key

## 5. Benchmark, 45 seconds
- 497 held-out parts, identical for every lane, API failures excluded, 95% CIs on every row
- Ours best of 8: 80.7%, one sample: 74.8%, GLM-5.3 Flash: 66.2%, Kimi K3: 61.8%, untuned base: 14.7%
- The gap widens where it matters: medium parts 61.3% against 37.0% and 30.3%
- Cost per 1,000 parts: $0.22 for one sample against $1.00 for Flash and $36.82 for K3
- Untuned Qwen3-VL-4B is 14.7%, so 60 points of this came from the post-training

## 6. How we used Baseten, 45 seconds
- Model APIs for both frontier baselines, high reasoning effort, two worked examples each, ours gets none
- Training Jobs for the LoRA SFT: epoch 1 on one H100, epoch 2 on four with DDP, resumed from the saved adapter
- Served straight from the training checkpoint, no weight copy in between
- One deployment serves several LoRA checkpoints, which is how we got the learning curve and the untuned baseline
- A fourth job trained nothing and only graded seven checkpoints, one process per GPU

## 7. What did not work, 20 seconds
- The grader is an exact reward, so we ran GRPO on top of the supervised model
- Training reward rose and correct rollouts went 39% to 70%, held-out accuracy still fell
- We graded every saved checkpoint with the SFT one as a control: worse after ten steps, and it never recovers
- We shipped the supervised model and put the negative result in the paper

## 8. Close, 15 seconds
- A 4B specialist beats frontier models on the task it was trained for, 4.5x cheaper than the cheapest frontier lane and 167x cheaper than the dearest
- The whole loop is on Baseten: baselines, training, serving, evaluation
- Repo and a short paper are public

## Questions we expect
- "Is it memorising?" Zero shared source parts, zero shared geometry signatures. 57 of 500 have a near duplicate in training, and our margin is +14.9 points on the 443 without one, wider than the +14.5 overall.
- "Is best of 8 a fair comparison?" We ran the same verifier on the frontier models. It lifts Kimi K3 from 28% to 48% on hard parts, at $252 per 1,000 parts. Our one sample number beats both anyway.
- "How is a part scored correct?" The program runs in a sandbox and the solid overlaps the reference by 0.9 or more, maximised over the 24 axis rotations.
- "What does the verifier see?" Only the drawing. It renders each candidate the same way and keeps the best silhouette match, so it works with no answer key.
- "How long did training take?" Under four hours of wall clock: 2.9 on one H100, then 55 minutes on four.
