#!/bin/bash
set -eux

# Same pinned set that dry_run_vlm.sh passes with. transformers>=4.57 is the floor for Qwen3-VL; no qwen-vl-utils
# needed (the transformers processor does the resizing). torch/torchvision pins match the image, so pip keeps them.
pip install -r requirements_vlm.txt
if [ "${EVAL_BENCH:-0}" = "1" ]; then
  # The in-job benchmark grades with the CadQuery checker (cadcheck/); OpenCascade needs a few GL libraries.
  (apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglu1-mesa libxrender1 libxext6) || true
  pip install cadquery trimesh scipy || echo "cadquery install failed: bench_eval will be skipped"
fi
python -c "import torch, transformers, trl, peft; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, trl.__version__, peft.__version__)"
nvidia-smi || true
df -h /dev/shm || true  # train_vlm.py uses 4 dataloader workers only when /dev/shm has >= 8 GiB
ls data/images | wc -l

if [ "${NPROC:-1}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NPROC" train_vlm.py   # DDP: one process per GPU
else
  python train_vlm.py
fi

# Held-out benchmark on the saved merged weights, as a separate process (its grading workers are spawned).
if [ "${EVAL_BENCH:-0}" = "1" ] && [ -f "${BT_CHECKPOINT_DIR:-./checkpoints}/merged/config.json" ]; then
  python bench_eval.py "${BT_CHECKPOINT_DIR:-./checkpoints}/merged" || echo "[bench] eval failed; the weights are saved"
fi
