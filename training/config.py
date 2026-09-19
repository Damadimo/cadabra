"""Baseten Training job: LoRA SFT of Qwen3-4B on one H100 (adapted from the ML Cookbook's qwen3-4b-lora-sft-trl).

  uv run python -m understudy.build_sft          # writes data/train.jsonl + data/val.jsonl in this folder
  cd training && baseten train push --config config.py
  baseten train job logs --job-id <job_id> --tail

Everything in this folder ships with the job. Knobs are env vars read by train.py; set MAX_STEPS=50 for a
two-minute smoke run before the real one.
"""

from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, CheckpointingConfig, Compute, Image, Runtime, TrainingJob, TrainingProject

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = Runtime(
    start_commands=["chmod +x ./run.sh && ./run.sh"],
    environment_variables={
        "BASE_MODEL": "Qwen/Qwen3-4B",
        "EPOCHS": "2",  # beyond ~2 epochs LoRA SFT starts to overfit; add fresh data instead
        "LORA_RANK": "16",  # must match LoRADetails(rank=...) in deploy_config.py
        "LR": "2e-4",
        "MAX_LEN": "4096",
        "MAX_STEPS": "-1",  # "50" for a smoke run
        "SAVE_STEPS": "100",
    },
    cache_config=CacheConfig(enabled=True),
    checkpointing_config=CheckpointingConfig(enabled=True),
)

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1)),
    runtime=training_runtime,
)

training_project = TrainingProject(name="understudy-sft", job=training_job)
