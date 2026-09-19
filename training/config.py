"""Baseten Training job, stage 1: LoRA SFT of Qwen3-4B-Instruct-2507 on spec→CadQuery pairs, one H100.

  uv run python -m understudy.cad.build_sft         # writes data/train.jsonl, data/val.jsonl in this folder
  ./dry_run.sh                                       # 2 CPU steps with a tiny Qwen3 (catches config/data errors)
  cd training && baseten train push --config config.py
  baseten train job logs --job-id <job_id> --tail

Everything in this folder ships with the job. Set MAX_STEPS=50 below for a ~2-minute smoke run first.
"""

from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, CheckpointingConfig, Compute, Image, Runtime, TrainingJob, TrainingProject

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "BASE_MODEL": "Qwen/Qwen3-4B-Instruct-2507",
        "EPOCHS": "2",  # beyond ~2 epochs LoRA SFT starts to overfit; add data instead
        "LORA_RANK": "16",  # must match LoRADetails(rank=...) in deploy_config.py
        "LR": "2e-4",
        "MAX_LEN": "4096",  # longest example is ~3.5K tokens; build_sft drops anything longer
        "BATCH": "8",
        "GRAD_ACCUM": "2",  # effective batch 16 -> ~465 steps per epoch
        "MAX_STEPS": "-1",  # "50" for a smoke run
        "SAVE_STEPS": "200",
    },
    cache_config=CacheConfig(enabled=True),
    checkpointing_config=CheckpointingConfig(enabled=True),
)

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    # Jobs default to 1 CPU and 2 GiB RAM unless set; an H100 node has 16 vCPUs and ~118 GiB.
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1), cpu_count=12, memory="96Gi"),
    runtime=training_runtime,
)

training_project = TrainingProject(name="understudy-cad-sft", job=training_job)
