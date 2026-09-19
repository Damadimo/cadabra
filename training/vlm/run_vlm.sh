#!/bin/bash
set -eux

# Same pinned set that dry_run_vlm.sh passes with. transformers>=4.57 is the floor for Qwen3-VL; no qwen-vl-utils
# needed (the transformers processor does the resizing). torch/torchvision pins match the image, so pip keeps them.
pip install -r requirements_vlm.txt
python -c "import torch, transformers, trl, peft; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, trl.__version__, peft.__version__)"
nvidia-smi || true
df -h /dev/shm || true  # train_vlm.py uses 4 dataloader workers only when /dev/shm has >= 8 GiB
ls data/images | wc -l

python train_vlm.py
