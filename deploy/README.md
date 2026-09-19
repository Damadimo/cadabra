# Serving Qwen3-VL-4B on Baseten (vLLM, OpenAI-compatible, image input)

## Status (2026-09-19, 05:06 EDT): blocked at deploy, nothing running, $0 spent

`baseten model push --dir deploy/vlm_base` failed before creating anything:

```
create model: baseten API error (HTTP 400): {"code": "VALIDATION_ERROR", "message": "You must add a payment method to deploy models."}
```

Fix: add a payment method to the "hack the north" workspace (Settings → Billing), or ask the Baseten booth to
unblock dedicated deployments for event credits. Then rerun the commands below. `baseten model list` shows
"No models found", and `baseten org billing usage` shows Dedicated $0.00.

Because nothing deployed, the image test, latency and cold start below are **not measured yet**. Both configs pass
`baseten model push --dry-run`, which checks the config schema only.

## Files

| File | What |
|---|---|
| `vlm_base/config.yaml` | Untuned `Qwen/Qwen3-VL-4B-Instruct` from Hugging Face (pinned commit, BDN-mirrored), vLLM `v0.29.0-cu129`, L4 |
| `vlm_ft/config.yaml` | Our merged fine-tune from a Baseten Training checkpoint (`bt://` weights). Placeholders: `TRAINING_PROJECT_NAME`, `TRAINING_JOB_ID` |
| `vlm_lora/config.yaml` | Base model + our LoRA adapter from any checkpoint (e.g. `checkpoint-600` if the job is stopped early). Serves both `Qwen/Qwen3-VL-4B-Instruct` (base lane) and `cadabra-vl` (ours) on one endpoint. `./scripts/deploy_vlm.sh <job_id> <gpu> <checkpoint>` fills it |
| `test_vlm.py` | Streams one chat completion with a base64 PNG and prints TTFT, total latency and tok/s |

## Push

```sh
cd ~/cadabra
baseten model push --dir deploy/vlm_base --wait --tail      # base: model "cadabra-vlm-base"
# after training: fill the bt:// line in vlm_ft/config.yaml, then
baseten train checkpoint list --job-id <TRAINING_JOB_ID>    # checkpoint ID must be "merged" and fully synced
baseten model push --dir deploy/vlm_ft --wait --tail        # ours: model "cadabra-vl"
baseten model list                                          # model IDs
```

## Endpoint and eval env

```
https://model-<model_id>.api.baseten.co/environments/production/sync/v1          # stable across redeploys
https://model-<model_id>.api.baseten.co/deployment/<deployment_id>/sync/v1       # one specific deployment
```

`/sync/<path>` goes straight to the vLLM server, so `/sync/v1/chat/completions` and `/sync/v1/models` both work.
Authenticate with `Authorization: Bearer $BASETEN_API_KEY`, which the OpenAI SDK sends as `api_key`.

```sh
set -a; . ./.env; set +a
export CADABRA_BASE_URL=https://model-<ft_model_id>.api.baseten.co/environments/production/sync/v1
export CADABRA_MODEL=cadabra-vl             # --served-model-name in vlm_ft (base: Qwen/Qwen3-VL-4B-Instruct)
export CADABRA_GPU_HOURLY=0.85                     # L4; 3.75 for H100_40GB, 6.50 for H100
uv run python deploy/test_vlm.py                      # uses /tmp/img2cad/montage_406.png if present
```

`cadabra/config.py` puts the `base-4b` lane on the same URL as `specialist`. With merged weights, base and
fine-tune are two deployments with two URLs, so that lane needs its own URL. The alternative is the LoRA option below.

## Latency and cost

| Instance (this workspace offers only T4, L4, H100 MIG, H100) | $/min | $/h | Measured TTFT / total |
|---|---|---|---|
| `L4:4x16` (L4 24 GB, 4 vCPU, 16 GiB) | 0.01414 | 0.85 | not measured (blocked) |
| `H100MIG` (`accelerator: H100_40GB`) | 0.0625 | 3.75 | not measured |
| `H100` | 0.10833 | 6.50 | not measured |

Prices come from `baseten api management /v1/instance_type_prices`. Billing runs while a replica is up, including
the scale-down delay. L4 is the cheapest GPU that fits: about 9 GB of bf16 weights plus KV cache for 8K context in 24 GB.
Decoding is memory-bandwidth bound (L4 300 GB/s, H100 3.35 TB/s), so measure on MIG or H100 before quoting race numbers.

## LoRA on Qwen3-VL (answer: yes)

vLLM v0.29.0 (`vllm/model_executor/models/qwen3_vl.py`): `Qwen3VLForConditionalGeneration` implements
`SupportsLoRA`. `--enable-lora` applies adapters to the language model. Adapters that also touch the vision tower
or merger need `--enable-tower-connector-lora` (`supports_tower_connector_lora = True`). One deployment can serve base
and fine-tune side by side, which fits the current `base-4b` / `specialist` lanes:

`vlm_lora/config.yaml` does this: base weights from the pinned HF commit at `/models/qwen3-vl-4b`, the adapter from
`bt://cadabra-vlm-sft@<job_id>/<checkpoint>` at `/models/adapter`, and `--enable-lora --max-lora-rank 64
--lora-modules cadabra-vl=<folder holding adapter_config.json>`. Our adapters only touch the language model.

Not yet run on Baseten (deploys are blocked). LoRA adds per-token overhead compared with merged weights. `baseten train checkpoint deploy` only
deploys LoRA checkpoints and is documented for LLMs, so write the config by hand for the VLM.

## Gotchas

- **The account needs a payment method before any deploy** (see above). Training GPU capacity is also not enabled yet.
- `bt://<project>[@<job_id>|latest][/<checkpoint>]`: the checkpoint is the directory name under `$BT_CHECKPOINT_DIR`.
  Baseten authenticates it automatically. Wait until the checkpoint has synced. Never delete the training job or
  project, because the deployment reads its weights from there.
  Sources: docs.baseten.co/development/model/bdn#baseten-training and docs.baseten.co/training/deployment.
- The `merged/` dir must be a complete HF dir in bf16: `save_pretrained` for the model **and** the processor.
  vLLM loads the image processor and chat template from it.
- Image budget: Qwen3-VL uses 1 token per 32×32 px. A 1024×512 montage is about 512 tokens and 1024×1024 about 1024.
  The processor allows up to 16,384 tokens per image, so with `--max-model-len 8192` send images of about 2 MP or less,
  or the request fails with a 400.
- Use the dotted `--limit-mm-per-prompt.image 4` form (as Baseten's recipes do). JSON inside `start_command` is fragile.
  `.video 0` skips video profiling.
- Use `vllm/vllm-openai:v0.29.0-cu129`, not the plain `v0.29.0` tag, which is CUDA 13.0.2 and needs a newer host driver.
- Port 8080 is reserved by Baseten's proxy, so vLLM binds 8000.
- The first push pulls the ~21 GB vLLM image and mirrors the weights, which takes minutes. Later scale-ups reuse the cache.
  Prewarm before judging.
- Stop billing when idle:
  `baseten model deployment deactivate --model-id <id> --deployment-id <dep> --yes`, or
  `baseten model deployment update-autoscaling --model-id <id> --deployment-id <dep> --min-replica 0 --scale-down-delay 60`.
  Remove completely: `baseten model delete --model-id <id> --yes`.
