"""Scripted checkpoint deploy (the alternative is the interactive `baseten train checkpoint deploy` wizard).

  baseten train checkpoint list --job-id <job_id>       # pick a checkpoint
  baseten train checkpoint deploy --config deploy_config.py

Needs the `hf_access_token` Baseten secret: the serving side downloads the base model itself.
The deployment serves base weights + our LoRA with vLLM behind an OpenAI-compatible endpoint:
https://model-<model_id>.api.baseten.co/environments/production/sync/v1, with model=<checkpoint name>.
"""

from truss.base import truss_config
from truss_train import definitions

TRAINING_JOB_ID = "REPLACE_ME"
CHECKPOINT = "checkpoint-REPLACE_ME"

deploy_config = definitions.DeployCheckpointsConfig(
    model_name="understudy-4b",
    checkpoint_details=definitions.CheckpointList(
        base_model_id="Qwen/Qwen3-4B",
        checkpoints=[
            definitions.LoRACheckpoint(
                training_job_id=TRAINING_JOB_ID,
                checkpoint_name=CHECKPOINT,
                lora_details=definitions.LoRADetails(rank=16),
            )
        ],
    ),
    runtime=definitions.DeployCheckpointsRuntime(
        environment_variables={"HF_TOKEN": definitions.SecretReference(name="hf_access_token")},
    ),
    compute=definitions.Compute(
        accelerator=truss_config.AcceleratorSpec(accelerator=truss_config.Accelerator.H100, count=1),
    ),
)
