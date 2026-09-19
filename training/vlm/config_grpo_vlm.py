"""Baseten Training job, stage 2 (optional) for the drawing-sheet model: GRPO with the geometry reward.

  ./dry_run_grpo_vlm.sh                                          # 1 CPU step with a tiny Qwen3-VL (free)
  cd training/vlm && SFT_JOB_ID=<sft job id> SFT_CHECKPOINT=checkpoint-<last step> baseten train push --config config_grpo_vlm.py

Name the SFT checkpoint explicitly: the job's newest checkpoint folder may be merged/ (full weights, no adapter).
Rollouts are generated with transformers on the H100; rewards run CadQuery on the node's CPUs (hence cpu_count).
The log prints s/step after 20 steps; MAX_STEPS and TIME_BUDGET_H cap the run. deploy: PROJECT=understudy-cad-vlm-grpo
./scripts/deploy_vlm.sh <grpo job id> H100_40GB
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

SFT_JOB_ID = os.environ.get("SFT_JOB_ID")
SFT_CHECKPOINT = os.environ.get("SFT_CHECKPOINT")
if not SFT_JOB_ID or not SFT_CHECKPOINT:
    raise SystemExit("set SFT_JOB_ID and SFT_CHECKPOINT (e.g. checkpoint-1900) of the stage-1 sheet job")

# Qwen3-VL-4B is public: no token needed. HF_SECRET=<workspace secret name> at push time adds an authenticated download.
_hf = {"HF_TOKEN": SecretReference(name=os.environ["HF_SECRET"])} if os.environ.get("HF_SECRET") else {}

training_runtime = Runtime(
    start_commands=["chmod +x ./run_grpo_vlm.sh && ./run_grpo_vlm.sh"],
    environment_variables={
        "BASE_MODEL": "Qwen/Qwen3-VL-4B-Instruct",
        "IMAGE_PIXELS": "1048576",  # same budget as stage 1
        "LR": "2e-5",
        "BATCH": "8",
        "GRAD_ACCUM": "8",  # 64 rollouts per step = 8 sheets x 8 samples
        "NUM_GENERATIONS": "8",
        "MAX_COMPLETION": "1024",  # reference programs: p95 529 tokens, max 1,101
        "TEMPERATURE": "1.0",
        "SUCCESS_BONUS": "0.5",  # reward = IoU (+0.5 when IoU >= 0.9); crash -0.2; no code -0.5
        "MAX_STEPS": os.environ.get("MAX_STEPS", "60"),  # time the first steps and scale this to the GPU time left
        "SAVE_STEPS": "10",
        "REWARD_WORKERS": "12",
        "MERGE_AT_END": "1",
        "EVAL_BENCH": os.environ.get("EVAL_BENCH", "1"),  # held-out sheets, greedy, graded after saving -> bench_eval/results.json
        "EVAL_N": os.environ.get("EVAL_N", "500"),
        **_hf,
    },
    cache_config=CacheConfig(enabled=True),
    checkpointing_config=CheckpointingConfig(enabled=True),
    load_checkpoint_config=LoadCheckpointConfig(
        enabled=True, checkpoints=[BasetenCheckpoint.from_named_checkpoint(checkpoint_name=SFT_CHECKPOINT, job_id=SFT_JOB_ID)]
    ),
)

training_job = TrainingJob(
    image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1), cpu_count=14, memory="100Gi"),
    runtime=training_runtime,
)

training_project = TrainingProject(name="understudy-cad-vlm-grpo", job=training_job)
