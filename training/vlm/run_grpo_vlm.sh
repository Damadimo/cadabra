#!/bin/bash
set -eux

# OpenCascade/VTK need a few X/GL libraries that slim CUDA images lack (harmless if already present).
(apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglu1-mesa libxrender1 libxext6) || true

# The stage-1 pins (dry-run tested) plus the geometry checker's dependencies for the reward.
pip install -r requirements_vlm.txt cadquery trimesh scipy
python -c "import torch, transformers, trl, peft, cadquery; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, trl.__version__, peft.__version__, 'cadquery', cadquery.__version__)"
nvidia-smi || true
nproc
ls data/images | wc -l
find "${BT_LOAD_CHECKPOINT_DIR:-/nonexistent}" -name adapter_config.json | head -5 || true

if [ "${NPROC:-1}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NPROC" grpo_vlm.py   # DDP: one process per GPU, each generates its share
else
  python grpo_vlm.py
fi

# Held-out benchmark on the saved merged weights, as a separate process (its grading workers are spawned).
if [ "${EVAL_BENCH:-0}" = "1" ] && [ -f "${BT_CHECKPOINT_DIR:-./checkpoints}/merged/config.json" ]; then
  python bench_eval.py "${BT_CHECKPOINT_DIR:-./checkpoints}/merged" || echo "[bench] eval failed; the weights are saved"
fi
