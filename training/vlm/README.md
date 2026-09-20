# Vision SFT: drawing sheet -> CadQuery (Qwen3-VL-4B-Instruct, LoRA, 1x H100)

| File | Role |
|---|---|
| `train_vlm.py` | TRL `SFTTrainer`, prompt/completion rows, loss on the completion only, LoRA r=16/alpha=32 on the language model's q/k/v/o/gate/up/down (vision tower and merger frozen), bf16, gradient checkpointing, no packing |
| `config_vlm.py` | Baseten job: `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`, 1x H100, 12 CPU, 96Gi, cache + checkpointing on, project `cadabra-vlm-sft` |
| `run_vlm.sh` | `pip install -r requirements_vlm.txt`, then `python train_vlm.py` |
| `requirements_vlm.txt` | Pinned versions shared by the job and the dry run |
| `dry_run_vlm.sh` | Fake 4-row dataset, 2 CPU steps with `trl-internal-testing/tiny-Qwen3VLForConditionalGeneration`, merged save, then checks that merged/ equals base + adapter |

## Run

```bash
uv run python -m cadabra.cad.build_vlm          # writes data/{train,val}.jsonl + data/images/ here
./training/vlm/dry_run_vlm.sh                      # ~30 s once uv has the packages; ends with "VLM dry run OK"
cd training/vlm && baseten train push --config config_vlm.py
baseten train job logs --job-id <job_id> --tail
```

Everything in this folder is uploaded as the job's code, including `data/`. Sheets are about 14 KB each, so 7.4k
images come to about 100 MB. For a smoke run, set `MAX_STEPS` to `"20"` in `config_vlm.py`. The merge still runs.
Take the s/step from the logs and multiply by the full step count.

The job should log these lines before step 1:
`image budget 1048576 px -> 1024 visual tokens per image`, `train: N rows, tokens/row p50=... dropped 0 ...`,
`LoRA r=16 on 252 language-model linear layers, 33.0M trainable params`, and
`loss on ... tokens of row 0, starting '```python\nimport cadquery'`.

Knobs (env vars in `config_vlm.py`): `EPOCHS` (2), `LR` (2e-4), `BATCH` (8), `GRAD_ACCUM` (2), `MAX_STEPS` (-1),
`SAVE_STEPS` (200, also the eval interval), `TIME_BUDGET_H` (0 = off; stops, saves and merges when the time is up), `LORA_RANK` (64), `LORA_ALPHA` (32), `MAX_LEN` (6144), `IMAGE_PIXELS` (1048576),
`MERGE_AT_END` (1). Also `WARMUP_STEPS`, `LOG_STEPS`, `NUM_WORKERS`, `ATTN_IMPL` and `DATA_DIR`.

## Time and cost (estimate, not measured)

Each row is 1,024 image tokens plus about 550 text tokens: a 228-token system prompt and the code (p50 225, p95 560,
max 4,336 tokens). Measured with the real tokenizer, rows are p50 1,533, p95 1,868 and max 5,644 tokens. The total
is 7.4k rows x ~1.6k tokens x 2 epochs ≈ **24M tokens**, or about 925 optimizer steps at an effective batch of 16.

At 5k–10k tokens/s (LoRA with gradient checkpointing, SDPA, no packing), training takes 40–80 min. Add 5–10 min for
pip, the first ~9 GB weight download (cached for later jobs), evals and the merge/save. Budget **1–1.5 H100-hours,
about $7–10** at $6.50/h. Check this against the smoke run.

Time scales linearly with rows: each 1k rows adds about 3.2M tokens and 6–10 min over 2 epochs. `build_vlm.py` now
adds the 12k `train_vlm_extra` parts by default (`--no-extra` skips them). With them, the set is about 17k rows:
plan for 2–3 h and $15–20, or set `EPOCHS=1`.

## Outputs (`$BT_CHECKPOINT_DIR`)

- `checkpoint-200`, `-400`, ... and `checkpoint-<last>`: the LoRA adapter (~130 MB) plus optimizer state.
- Top level: the final adapter, the processor and `train_log.json` (loss and eval-loss history).
- **`merged/`**: full bf16 weights (~9 GB), plus the base repo's own config, tokenizer, chat template and processor
  files. `preprocessor_config.json` is pinned to `IMAGE_PIXELS`. This is the vLLM model directory
  (`deploy/vlm_ft/config.yaml` pulls `rank-0/merged/` of the job with `training_checkpoints`).

```bash
baseten train checkpoint list --job-id <job_id>    # "merged" sits next to checkpoint-N; wait for it to sync
baseten train checkpoint files --job-id <job_id>   # presigned URLs, if you want to download merged/*
```

## Gotchas

- **Row format**: rows are converted at load time and the files on disk are unchanged. Every message content becomes
  a list of typed parts, because Arrow can't store a string system/assistant content alongside a list user
  content. Image paths resolve against `data/`, then against this folder. Images are decoded lazily as RGB PIL by
  TRL's collator.
- **Versions**: Qwen3-VL needs transformers >= 4.57. The pins (transformers 5.17, trl 1.13, peft 0.21, datasets
  5.0.1, accelerate 1.15) run on the image's torch 2.7.0 and satisfy transformers' floor of torch >= 2.5. Change them
  only together with a dry run. In TRL 1.13 the default loss is `chunked_nll`, which needs `lm_head` to have no
  adapter; the LoRA regex targets the language model only, so that holds.
- **merged/ config**: a plain transformers 5 `save_pretrained` writes `rope_parameters`, and transformers 4.57 fails
  to load that config. So merged/ reuses the base repo's config.json. The dry-run merged/ loads under both
  transformers 4.57.6 and 5.17, with identical logits.
- **Image budget is a train/serve contract**: min = max = 1024x1024 px. A 1024x1024 sheet gives exactly 1024 tokens.
  Other sizes are resized to the same area. Don't override `min_pixels`/`max_pixels` at serving time.
- **Dataloader workers**: 4 when `/dev/shm` is at least 8 GiB, otherwise 0, because a batch of 8 carries ~200 MB of
  float32 pixels and a 64 MB shm crashes workers. With 0 workers, PNG decoding happens on the training process
  (slower, but safe). Check `df -h /dev/shm` in the job log.
- **Dry run**: the tiny model is random, so its loss (~11.9) and generated sample mean nothing. It checks the wiring
  only.

## Optional stage 2: GRPO on the geometry reward

`grpo_vlm.py` continues the SFT adapter with GRPO: 8 samples per sheet, reward = aligned IoU of the built solid vs the
reference (+0.5 when IoU >= 0.9; crash -0.2; no code -0.5), rollouts generated with transformers (no vLLM in the image).

```sh
./training/vlm/dry_run_grpo_vlm.sh        # reward on known answers + 1 CPU step with a tiny Qwen3-VL (free)
cd training/vlm && SFT_JOB_ID=<sft job> SFT_CHECKPOINT=checkpoint-<last> baseten train push --config config_grpo_vlm.py
PROJECT=cadabra-vlm-grpo ./scripts/deploy_vlm.sh <grpo job> H100_40GB
```

Only worth it with GPU time left after SFT + deploy + benchmark: time the first steps (~1 min/step expected) and
keep it only if the benchmark improves on the same parts (`leaderboard.py --common`).

## Scoring saved checkpoints without deploying them

`eval_sweep.py` grades a list of already-saved LoRA checkpoints with the harness `bench_eval.py` uses (the same 500
held-out sheets, greedy, the same geometry checker), so the numbers sit beside the in-job evals of the runs that
produced them. Baseten mirrors every named checkpoint into `$BT_LOAD_CHECKPOINT_DIR`; the base model is loaded once
per GPU process and each adapter is attached, graded and unloaded again.

```sh
./training/vlm/dry_run_eval_sweep.sh      # two fake adapters on a tiny Qwen3-VL, CPU (free)
cd training/vlm && SWEEP="<job>:checkpoint-10,<job>:checkpoint-20,..." GPUS=4 baseten train push --config config_eval_sweep.py --team <team>
```

Results land in `$BT_CHECKPOINT_DIR/bench_eval/<job>-rank-0-<checkpoint>.json` and one `[sweep] <name> {...}` line per
checkpoint in the log. Include a checkpoint whose score you already know: it has to come back the same, or the sweep
is measuring something else. `scripts/import_job_eval.py` pulls any of these into `runs/` for the leaderboard.

