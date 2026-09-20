# Demo script, 4 minutes

Written for a screen recording, and it reads the same live. Times are cumulative and the section budgets add to 4:00.
SAY lines are what you narrate, DO lines are what is on screen.

## Before you record, not part of the 4 minutes
- Start the server: `cd ~/cadabra && CAD_WORKERS=4 uv run python -m uvicorn app.server:app --port 8000`
- Warm the endpoint, it scales to zero: `set -a && . ./.env && set +a && uv run python deploy/test_vlm.py`
- Record the browser window only, not the whole display, so the terminal and .env stay out of frame
- Open on Compare, default part, rotate on, overlay on
- Do one throwaway live race first so you know the wifi is good

## 1. Hook, 0:00 to 0:20
- DO: Compare tab, the drawing sheet filling the left tile
- SAY: "This is an engineering drawing of a mechanical part. Four views and a bounding box. That is the whole input."
- SAY: "Cadabra is a 4 billion parameter model we post-trained on Baseten. It reads that drawing and writes the CAD program that builds the part."
- SAY: "It beats Kimi K3 and GLM-5.3 Flash at this, and it costs a fraction as much."

## 2. Why this is the gap, 0:20 to 0:50
- DO: stay on the sheet, let the viewer actually read it
- SAY: "Frontier models are already good at CAD from a written spec. Kimi K3 gets 95 percent of our parts right that way."
- SAY: "Give them the same part as a drawing instead and K3 drops to 62 percent. Flash to 66."
- SAY: "Real CAD work starts from drawings, so that is the gap we went after. It is narrow, and it is checkable."

## 3. What our model produced, 0:50 to 1:35
- DO: gesture left tile, then right tile
- SAY: "Left is the input. Right is the solid that our model's program actually builds when you run it."
- SAY: "The red is the reference, the answer the dataset considers correct, ghosted on top."
- SAY: "Red on its own is volume the model missed. Green on its own is volume it invented. That is exactly what the score measures."
- DO: click Program, scroll the code for two seconds, close it
- SAY: "This is the CadQuery it wrote. We execute it in a sandbox and compare the solid, never the text. This part is 20 faces, complex tier, 0.902 overlap. Correct."

## 4. The same part, frontier models, 1:35 to 2:05
- DO: open the model dropdown, choose GLM-5.3 Flash
- SAY: "Same drawing, GLM-5.3 Flash at high reasoning effort."
- SAY: "Its bars run well past the red reference at both ends. 0.49. Wrong."
- DO: choose Kimi K3
- SAY: "Kimi K3, also wrong on the same part."
- SAY: "You can see the failure without reading a single number."

## 5. Live race, 2:05 to 2:55
- DO: Live race tab, pick a part, press Run, let it play
- SAY: "This is live, right now. All three models get the same sheet and the same bounding box at the same moment."
- SAY: "Ours is a dedicated H100 on Baseten. The other two are Baseten Model APIs, high reasoning effort, two worked examples each. Ours gets none."
- SAY: "Ours comes back in about two seconds. Flash around four. K3 around ten."
- SAY: "Each answer is executed and compared to the reference while you watch."
- SAY: "Ours samples eight programs and keeps the one that best matches the drawing, so it checks its own work with no answer key."

## 6. The numbers, 2:55 to 3:25
- DO: scroll down to the benchmark table
- SAY: "497 held-out parts. Same parts for every model, confidence intervals on every row."
- SAY: "Ours, 80.7 percent best of eight, 74.8 on a single sample. Flash 66.2. Kimi K3 61.8."
- SAY: "The untuned base model is 14.7, so nearly all of that came from the post-training."
- SAY: "And the cost. A single sample is 22 cents per thousand parts. Flash is a dollar. K3 is 37 dollars."

## 7. Built on Baseten, and one thing that failed, 3:25 to 3:50
- SAY: "The whole loop is Baseten. Model APIs for the baselines, Training Jobs for the LoRA, one H100 for the first epoch and four with DDP for the second."
- SAY: "Served straight from the training checkpoint, no weight copy in between."
- SAY: "We also tried reinforcement learning on the geometry reward. It made the model worse, so we graded every checkpoint against a control to show exactly where, and shipped the supervised one."

## 8. Close, 3:50 to 4:00
- SAY: "A 4 billion parameter specialist beating frontier models on the job it was trained for, at a fraction of the price."
- SAY: "Repo and a short paper are public."

## If you are running long, cut in this order
- Section 7, the failed RL line, saves 10 seconds
- Section 4, drop Kimi K3 and show only Flash, saves 10 seconds
- Section 2, drop the written spec comparison, saves 12 seconds

## Questions to expect afterwards
- "Is it memorising?" Zero shared source parts, zero shared geometry signatures. 57 of 500 have a near duplicate in training, and our margin is wider without them, +14.9 against +14.5.
- "Is best of eight fair?" We ran the same verifier on the frontier models. It lifts Kimi K3 from 28 to 48 percent on hard parts, at $252 per thousand. Our single sample still beats both.
- "How is a part scored correct?" The program runs in a sandbox and its solid overlaps the reference by 0.9 or more, maximised over the 24 axis rotations.
- "How long was training?" Under four hours of wall clock. 2.9 on one H100, then 55 minutes on four.
