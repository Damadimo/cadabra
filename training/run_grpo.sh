#!/bin/bash
set -eux

# OpenCascade/VTK need a few X/GL libraries that slim CUDA images lack (harmless if already present).
(apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglu1-mesa libxrender1 libxext6) || true

pip install "trl[vllm]>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0" datasets accelerate cadquery trimesh scipy
python -c "import cadquery; print('cadquery', cadquery.__version__)"
nproc

python grpo.py
