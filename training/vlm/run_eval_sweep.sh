#!/bin/bash
set -eux

# Same image setup as the training runs: OpenCascade needs a few X/GL libraries the slim CUDA image lacks.
(apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libglu1-mesa libxrender1 libxext6) || true
pip install -r requirements_vlm.txt cadquery trimesh scipy
nvidia-smi || true
nproc
find "${BT_LOAD_CHECKPOINT_DIR:-/nonexistent}" -name adapter_config.json | sort

# One process per GPU, each taking every NPROC-th adapter.
N="${NPROC:-1}"
status=0
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES="$i" SHARD="$i" SHARDS="$N" python eval_sweep.py 2>&1 | sed "s/^/[gpu$i] /" &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || status=1; done
exit "$status"
