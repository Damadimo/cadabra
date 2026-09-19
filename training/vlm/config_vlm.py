"""Baseten Training job: LoRA SFT of Qwen3-VL-4B-Instruct on rendered part sheets -> CadQuery, one H100.

  ./dry_run_vlm.sh                                   # 2 CPU steps with a tiny random Qwen3-VL (catches config/data errors)
  cd training/vlm && baseten train push --config config_vlm.py
  baseten train job logs --job-id <job_id> --tail

Everything in this folder ships with the job, including data/ (train.jsonl, val.jsonl, images/).
Set MAX_STEPS=20 below for a short smoke run first. The served model is $BT_CHECKPOINT_DIR/merged (see README.md).
"""

from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, CheckpointingConfig, Compute, Image, Runtime, TrainingJob, TrainingProject

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = Runtime(
    start_commands=["chmod +x ./run_vlm.sh && ./run_vlm.sh"],
    environment_variables={
        "BASE_MODEL": "Qwen/Qwen3-VL-4B-Instruct",
        "EPOCHS": "1",  # ~17k rows with the train_middle extras: one pass is ~1,070 steps (~1.5-2 H100-hours)
        "LORA_RANK": "16",
        "LR": "2e-4",
        "MAX_LEN": "6144",  # rows are ~1024 image + ~600 text tokens; never truncate
        "IMAGE_PIXELS": "1048576",  # 1024*1024: a 1024x1024 sheet -> exactly 1024 visual tokens
        "BATCH": "8",
        "GRAD_ACCUM": "2",  # effective batch 16
        "MAX_STEPS": "-1",  # "20" for a smoke run
        "SAVE_STEPS": "200",
        "MERGE_AT_END": "1",  # writes $BT_CHECKPOINT_DIR/merged for vLLM
    },
    cache_config=CacheConfig(enabled=True),  # persists /root/.cache (HF weights, pip) across jobs in this project
    checkpointing_config=CheckpointingConfig(enabled=True),
)

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    # Jobs default to 1 CPU and 2 GiB RAM unless set; an H100 node has 16 vCPUs and ~118 GiB.
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1), cpu_count=12, memory="96Gi"),
    runtime=training_runtime,
)

training_project = TrainingProject(name="understudy-cad-vlm-sft", job=training_job)
