"""Baseten Training job that only evaluates: score a list of saved LoRA checkpoints on the held-out sheets.

  cd training/vlm && SWEEP="wd669g3:checkpoint-10,...,qj9924w:checkpoint-946" GPUS=4 baseten train push --config config_eval_sweep.py

It trains nothing. Baseten mirrors every named checkpoint into $BT_LOAD_CHECKPOINT_DIR and run_eval_sweep.sh starts one
process per GPU, each grading its share with bench_eval.py's harness, so an RL run's checkpoint-by-checkpoint curve
costs one job instead of one deployment per checkpoint. Numbers land beside the in-job evals of the runs themselves:
same 500 sheets, same greedy decoding, same geometry checker.
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

GPUS = int(os.environ.get("GPUS", "1"))
SWEEP = os.environ.get("SWEEP")  # comma-separated <job id>:<checkpoint name>
if not SWEEP:
    raise SystemExit("set SWEEP=<job id>:<checkpoint>,... (e.g. wd669g3:checkpoint-10,wd669g3:checkpoint-20)")
CHECKPOINTS = [entry.split(":", 1) for entry in SWEEP.split(",") if entry.strip()]

_hf = {"HF_TOKEN": SecretReference(name=os.environ["HF_SECRET"])} if os.environ.get("HF_SECRET") else {}

training_runtime = Runtime(
    start_commands=["chmod +x ./run_eval_sweep.sh && ./run_eval_sweep.sh"],
    environment_variables={
        "BASE_MODEL": "Qwen/Qwen3-VL-4B-Instruct",
        "IMAGE_PIXELS": "1048576",  # the budget the adapters were trained at
        "NPROC": str(GPUS),
        "EVAL_N": os.environ.get("EVAL_N", "500"),
        "EVAL_BATCH": os.environ.get("EVAL_BATCH", "32"),
        "REWARD_WORKERS": str(max(4, 48 // GPUS)),  # CadQuery graders per GPU process
        **_hf,
    },
    cache_config=CacheConfig(enabled=True),
    checkpointing_config=CheckpointingConfig(enabled=True),  # for the bench_eval/*.json this writes
    load_checkpoint_config=LoadCheckpointConfig(
        enabled=True,
        checkpoints=[BasetenCheckpoint.from_named_checkpoint(checkpoint_name=name, job_id=job) for job, name in CHECKPOINTS],
    ),
)

training_job = TrainingJob(
    image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=GPUS), cpu_count=14 * GPUS, memory=f"{100 * GPUS}Gi"),
    runtime=training_runtime,
)

training_project = TrainingProject(name="cadabra-vlm-eval-sweep", job=training_job)
