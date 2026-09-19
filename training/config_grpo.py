"""Baseten Training job, stage 2 (optional): GRPO with a geometry reward, continuing the stage-1 adapter.

  SFT_JOB_ID=<stage-1 job id> [SFT_CHECKPOINT=checkpoint-<step>] baseten train push --config config_grpo.py

Without SFT_CHECKPOINT it loads the latest checkpoint of that job. Rollouts come from vLLM colocated on the same
H100; rewards come from the CadQuery checker running on the node's CPUs (hence cpu_count).
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
    TrainingJob,
    TrainingProject,
)

SFT_JOB_ID = os.environ.get("SFT_JOB_ID")
SFT_CHECKPOINT = os.environ.get("SFT_CHECKPOINT")
if not SFT_JOB_ID:
    raise SystemExit("set SFT_JOB_ID to the stage-1 training job id")
source = (
    BasetenCheckpoint.from_named_checkpoint(checkpoint_name=SFT_CHECKPOINT, job_id=SFT_JOB_ID)
    if SFT_CHECKPOINT
    else BasetenCheckpoint.from_latest_checkpoint(job_id=SFT_JOB_ID)
)

training_runtime = Runtime(
    start_commands=["chmod +x ./run_grpo.sh && ./run_grpo.sh"],
    environment_variables={
        "BASE_MODEL": "Qwen/Qwen3-4B-Instruct-2507",
        "LR": "1e-5",
        "BATCH": "8",
        "GRAD_ACCUM": "4",  # 32 completions per step = 4 prompts x 8 samples
        "NUM_GENERATIONS": "8",
        "MAX_COMPLETION": "1024",
        "MAX_STEPS": "150",  # time 10 steps first and scale this to the GPU time you have
        "SAVE_STEPS": "25",
        "USE_VLLM": "1",
        "VLLM_MEM": "0.35",
        "REWARD_WORKERS": "12",
    },
    cache_config=CacheConfig(enabled=True),
    checkpointing_config=CheckpointingConfig(enabled=True),
    load_checkpoint_config=LoadCheckpointConfig(enabled=True, checkpoints=[source]),
)

training_job = TrainingJob(
    image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1), cpu_count=14, memory="100Gi"),
    runtime=training_runtime,
)

training_project = TrainingProject(name="understudy-cad-grpo", job=training_job)
