"""Baseten Training job: LoRA SFT of Qwen3-VL-4B-Instruct on rendered part sheets -> CadQuery, one H100.

  ./dry_run_vlm.sh                                   # 2 CPU steps with a tiny random Qwen3-VL (catches config/data errors)
  cd training/vlm && baseten train push --config config_vlm.py
  baseten train job logs --job-id <job_id> --tail

Everything in this folder ships with the job, including data/ (train.jsonl, val.jsonl, images/).
Set MAX_STEPS=20 below for a short smoke run first. The served model is $BT_CHECKPOINT_DIR/merged (see README.md).
"""

from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, CheckpointingConfig, Compute, Image, Runtime, SecretReference, TrainingJob, TrainingProject

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

training_runtime = Runtime(
    start_commands=["chmod +x ./run_vlm.sh && ./run_vlm.sh"],
    environment_variables={
        "BASE_MODEL": "Qwen/Qwen3-VL-4B-Instruct",
        "EPOCHS": "1",  # ~30k rows with both train_middle batches: one pass is ~1,900 steps (~2-3 H100-hours)
        "LORA_RANK": "64",  # ~130M adapter params: room for ~20M supervised code tokens
        "LORA_ALPHA": "32",
        "LR": "2e-4",
        "MAX_LEN": "6144",  # rows are ~1024 image + ~600 text tokens; never truncate
        "IMAGE_PIXELS": "1048576",  # 1024*1024: a 1024x1024 sheet -> exactly 1024 visual tokens
        "BATCH": "8",
        "GRAD_ACCUM": "2",  # effective batch 16
        "MAX_STEPS": "-1",  # "20" for a smoke run
        "SAVE_STEPS": "200",
        "TIME_BUDGET_H": "0",  # e.g. "2.5" if the H100 is ours for 3 h: stops, saves and merges in time
        "MERGE_AT_END": "1",  # writes $BT_CHECKPOINT_DIR/merged for vLLM
        "EVAL_BENCH": "1",  # after saving: greedy answers for the 500 held-out sheets, graded -> bench_eval/results.json (~6-10 min)
        "SAVE_ONLY_MODEL": "1",  # checkpoints hold the adapter only (no optimizer state): ~3x smaller, faster to sync/deploy
        "HF_TOKEN": SecretReference(name="hf_access_token"),  # workspace secret: authenticated (faster) weight download
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
