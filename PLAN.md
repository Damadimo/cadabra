# Build plan: Sat Sep 19 → Sun Sep 20 (EDT)

## Hard deadlines

| When | What |
|---|---|
| **Sat 12:30 PM** | Initial Devpost submitted: final team, badge IDs exactly as printed, **every prize ticked** (hard cutoff 2:00 PM; prizes added later don't count) |
| **Sat 11:00 AM** | Training gate: no H100 access by now → pivot (see below) |
| **Sun 1:00 AM** | Feature freeze |
| **Sun 8:00 AM** | Code freeze + final Devpost (repo link required, video strongly recommended) |
| Sun 9:45–11:45 AM | Sponsor judging. The round-1 room slot may overlap, so **anyone on the team must be able to run the demo alone** |

## Roles

- **A: eval & data.** Gold set, checks, error analysis, metrics.
- **B: training & serving.** Baseten jobs, checkpoint deploy, cascade.
- **C: app.** Race UI, correction flow, the "take an action" step.
- **D: story.** Devpost, WORKLOG, video, pitch, sponsor tracks; helps label.

Three people: merge C and D. Two people: A+D and B+C.

## Gates

1. **03:00, domain go/no-go.** Use the dataset that gives ≥300 real inputs and a defensible way to grade them. Default: Federato (insurance). Fallback: CSE (security logs, `UNDERSTUDY_TASK=seclogs`).
2. **11:00, training access.** If there are no H100s by then, pivot to a trained-router or long-document-agent variant. The data, checks, eval harness and UI all carry over.
3. **Before v2 training: freeze the gold set.** Final numbers must not be tuned on the test set.

## Timeline

| Time | Who | Work | Done when |
|---|---|---|---|
| Sat 01:00–01:45 | All | One Baseten workspace (creator redeems the promo), verify the account, file the event rate-limit form, install the Baseten CLI, `uv`, Switch; store the HF token as Baseten secret `hf_access_token` | `scripts/smoke.py` passes |
| 01:45–03:00 | A/B/C/D | A: pull data, adapt the task file (schema, guidelines, checks). B: baselines on 20 inputs. C: UI on the mock. D: Devpost draft, prize list | **03:00 gate** |
| 03:00–05:00 | A+D, B, C | A+D: 100–150-item gold set (teacher prefill, hand-corrected, split by source). B: teacher labeling in the background. C: metrics panel | 2K+ accepted labels |
| 05:00–09:00 | B, D awake | B: scale labels to 5–10K, build SFT data. D: worklog, diagram. First person free goes to the booth when it opens | Training enabled |
| 09:00–10:30 | All | B: `MAX_STEPS=50` smoke job (~2 min), then full SFT v1 | v1 checkpoint synced |
| 10:30–12:00 | B, A | Deploy v1, evaluate at concurrency 1 and 16 | First honest table |
| 12:00–12:30 | D | **Submit the initial Devpost with all prizes ticked** | Submitted |
| 12:30–16:30 | A, C awake | A: error analysis on v1 failures, fix checks and labels. C: cascade, corrections, action step | Failure taxonomy |
| 16:30–19:30 | All | v2: retrain on v1 failures + corrections | v2 trained |
| 19:30–22:00 | All | Deploy + evaluate v2, integrate, record backup video clips | Numbers frozen |
| 22:00–01:00 | All | Stretch (GRPO on the check reward) only if every gate passed; otherwise harden | **Feature freeze** |
| Sun 01:00–08:00 | Shifts | Rehearse ×3, cut the video, finish the Devpost, clean the repo, scrub secrets | **8:00 lock** |
| 08:00–09:45 | All | Scale the deployment up (min replicas 1) and prewarm; test the demo on a phone hotspot | Demo ready |

## Questions for the Baseten booth

- How many H100s per team, for how long? Do training jobs and deployments draw on our credits?
- Can the deployment run on a MIG slice or a smaller GPU?
- Is Loops (the RL SDK) available to hackers?
- Is Hosted Tools (web search) enabled for event workspaces?
- Can our rate limits be raised for the teacher run?
- Is the Baseten prize judged at the booth, on Devpost, or both?

## Devpost checklist

- [ ] "How we use Baseten": model slugs, training job IDs, GPU-minutes, dollars
- [ ] Results table from `runs/<run>/summary.md`, with n, CIs, concurrency and reasoning effort
- [ ] What didn't work (the worklog's dead ends)
- [ ] Pipeline diagram, repo link, 2-minute video

## Demo (4:10 of a 5-minute slot)

1. **0:00–0:20.** The one-line claim and the headline number from the held-out eval.
2. **0:20–1:30.** A judge picks or edits an input (delete a field, plant a contradiction). Three lanes race. Point at TTFT, tok/s, $/doc and the checks.
3. **1:30–2:30.** Scoreboard: accuracy with CIs, p50/p95, $/1K tasks, escalation rate, the v0→v1→v2 curve, one thing that failed.
4. **2:30–3:15.** Pipeline: teachers, checks, H100 job (ID, minutes, dollars), checkpoint deploy, cascade.
5. **3:15–3:45.** The judge corrects a field and it lands in the queue that trained v2.
6. **3:45–4:10.** Close: a model we own, trained on this task's signal, served on Baseten.

Prepared answers: badly-prompted baseline? (show the few-shot baseline) · is the speed gap just reasoning? (show
low effort too) · overfit? (split by source, and the judge's live input) · trained before the event? (job
timestamps) · idle GPU cost? ($/hour and the break-even volume).

## Platform pitfalls

- Deprecated at 5 PM PT Sept 25: Inkling, Inkling Small, Kimi K2.6, Kimi K2.7 Code, GLM-4.7, DeepSeek-V4-Pro. Don't build on them.
- Never delete a training job or project that holds undeployed checkpoints; they are gone for good.
- The deployment bills $6.50/h while up. Scale to zero while developing, and prewarm before judging.
- Rate limits are per model. 429 and 529 both happen; `llm.py` backs off, and splitting load across models helps.
- `n` must be 1, and `logprobs` support varies by model.
- GLM-5.3 Fast returns a 400 for `reasoning_effort: none`; Kimi K3 defaults to max effort.
- By default, Switch falls back to Anthropic on a 429/5xx. Change that if you don't want your own Anthropic spend.
- Only quote numbers we measured, on our workload. Don't claim any provider is "the fastest".
