"""Baseten Training job: LoRA SFT of Qwen3-VL-4B-Instruct on rendered part sheets -> CadQuery, one H100.

  ./dry_run_vlm.sh                                   # 2 CPU steps with a tiny random Qwen3-VL (catches config/data errors)
  cd training/vlm && SMOKE=1 baseten train push --config config_vlm.py --job-name smoke   # 20 steps + 32-part eval
  cd training/vlm && baseten train push --config config_vlm.py --job-name sft             # the real run
  baseten train job logs --job-id <job_id> --tail

Everything in this folder ships with the job, including data/ (train.jsonl, val.jsonl, images/).
SMOKE=1 (read here, at push time) makes a short run that exercises the image, data, save, merge and eval path.
TIME_BUDGET_H=<hours> at push time caps the training loop. The served model is $BT_CHECKPOINT_DIR/merged (README.md).
"""

import os

from truss.base.truss_config import AcceleratorSpec
from truss_train import (
    BasetenCheckpoint,
    CacheConfig,
    CheckpointingConfig,
    Compute,
    Image,
    LoadCheckpointConfig,
    Runtime,
    SecretReference,
    TrainingJob,
    TrainingProject,
)

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
SMOKE = os.environ.get("SMOKE") == "1"
# GPUS=<n> at push time: n H100s on one node, one DDP process each (run_vlm.sh uses torchrun). Per-GPU batch stays 8;
# the effective batch is 8 x n x GRAD_ACCUM (default GRAD_ACCUM: 2 on one GPU, 1 on several).
GPUS = int(os.environ.get("GPUS", "1"))
# CONTINUE_JOB=<job id> CONTINUE_CHECKPOINT=<name> at push time: keep training that adapter (e.g. a second epoch) with a
# new data order, instead of starting from the base model.
CONTINUE_JOB, CONTINUE_CHECKPOINT = os.environ.get("CONTINUE_JOB"), os.environ.get("CONTINUE_CHECKPOINT")
_continue = (
    {"load_checkpoint_config": LoadCheckpointConfig(enabled=True, checkpoints=[
        BasetenCheckpoint.from_named_checkpoint(checkpoint_name=CONTINUE_CHECKPOINT, job_id=CONTINUE_JOB)])}
    if CONTINUE_JOB and CONTINUE_CHECKPOINT else {}
)

# Qwen3-VL-4B is public: no token needed. HF_SECRET=<workspace secret name> at push time adds an authenticated download.
_hf = {"HF_TOKEN": SecretReference(name=os.environ["HF_SECRET"])} if os.environ.get("HF_SECRET") else {}

training_runtime = Runtime(
    start_commands=["chmod +x ./run_vlm.sh && ./run_vlm.sh"],
    environment_variables={
        "BASE_MODEL": "Qwen/Qwen3-VL-4B-Instruct",
        "EPOCHS": os.environ.get("EPOCHS", "1"),  # ~30k rows: one pass is ~1,900 steps at batch 16 (~2.8 h on one H100)
        "LORA_RANK": "64",  # ~130M adapter params: room for ~20M supervised code tokens
        "LORA_ALPHA": "32",
        "LR": os.environ.get("LR", "2e-4"),
        "MAX_LEN": "6144",  # rows are ~1024 image + ~600 text tokens; never truncate
        "IMAGE_PIXELS": "1048576",  # 1024*1024: a 1024x1024 sheet -> exactly 1024 visual tokens
        "BATCH": "8",
        "GRAD_ACCUM": os.environ.get("GRAD_ACCUM", "2" if GPUS == 1 else "1"),  # effective batch 8 x GPUS x GRAD_ACCUM
        "NPROC": str(GPUS),
        "DATA_SEED": os.environ.get("DATA_SEED", "42"),
        "INIT_ADAPTER": "auto" if _continue else "",
        "MAX_STEPS": "20" if SMOKE else "-1",
        "SAVE_STEPS": "10" if SMOKE else "200",
        "TIME_BUDGET_H": os.environ.get("TIME_BUDGET_H", "0"),  # e.g. 2.5 if the H100 is ours for 3 h: stops, saves, merges in time
        "MERGE_AT_END": "1",  # writes $BT_CHECKPOINT_DIR/merged for vLLM
        "EVAL_BENCH": "1",
        "EVAL_N": "32" if SMOKE else "500",  # after saving: greedy answers for the 500 held-out sheets, graded -> bench_eval/results.json (~6-10 min)
        "SAVE_ONLY_MODEL": "1",  # checkpoints hold the adapter only (no optimizer state): ~3x smaller, faster to sync/deploy
        **_hf,
    },
    cache_config=CacheConfig(enabled=True),  # persists /root/.cache (HF weights, pip) across jobs in this project
    checkpointing_config=CheckpointingConfig(enabled=True),
    **_continue,
)

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    # Jobs default to 1 CPU and 2 GiB RAM unless set; an H100 node has 16 vCPUs and ~118 GiB.
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=GPUS), cpu_count=12 * GPUS, memory=f"{96 * GPUS}Gi"),
    runtime=training_runtime,
)

training_project = TrainingProject(name="understudy-cad-vlm-sft", job=training_job)
