#!/bin/bash
# Learning curve: serve several LoRA checkpoints of one training job from ONE deployment (base weights once, one
# adapter per checkpoint), benchmark each on the frontier's parts, and print the table.
#
#   TEAM=22 ./scripts/learning_curve.sh <training_job_id> checkpoint-400,checkpoint-600,checkpoint-800 [accelerator]
#
# The model is "understudy-cad-vl-curve" (its own deployment, so it never disturbs the demo endpoint). Each checkpoint
# is served under its own name; runs are tagged ckpt<N>-img. Deactivate it afterwards:
#   baseten model deployment deactivate --model-id <id> --deployment-id <id> --yes
set -euo pipefail
JOB_ID="${1:?usage: learning_curve.sh <training_job_id> <checkpoint,checkpoint,...> [accelerator]}"
CKPTS="${2:?comma-separated checkpoint names}"
GPU="${3:-L4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FRONTIER_RUN="${FRONTIER_RUN:-20260919-051617_sota-img}"

WORK="$(mktemp -d)"
python3 - "$ROOT/deploy/vlm_lora/config.yaml" "$WORK/config.yaml" "$JOB_ID" "$CKPTS" "$GPU" <<'EOF'
import re, sys
src, dst, job, ckpts, gpu = sys.argv[1:]
s = open(src).read()
s = s.replace("model_name: understudy-cad-vl-lora", "model_name: understudy-cad-vl-curve")
# one reference per checkpoint: Baseten mirrors each reference as one bt:// volume, and several paths under a single
# reference mirrored nothing
refs = "".join(f"    - training_job_id: {job}\n      paths:\n        - rank-0/{c}/\n" for c in ckpts.split(","))
s = re.sub(r"    - training_job_id: TRAINING_JOB_ID\n      paths:\n        - rank-0/CHECKPOINT_NAME/\n", refs, s)
start = s.index("  start_command: >-")
end = s.index("  readiness_endpoint:")
s = s[:start] + """  start_command: >-
    sh -c 'MODS=""; N=0; for C in $(find /tmp/training_checkpoints -name adapter_config.json | sort); do
    D=$(dirname "$C"); MODS="$MODS $(basename "$D")=$D"; N=$((N+1)); done; echo "adapters:$MODS" &&
    exec vllm serve /models/qwen3-vl-4b
    --served-model-name Qwen/Qwen3-VL-4B-Instruct
    --enable-lora --max-lora-rank 64 --max-loras $N
    --lora-modules $MODS
    --host 0.0.0.0
    --port 8000
    --dtype bfloat16
    --max-model-len 8192
    --max-num-seqs 48
    --gpu-memory-utilization 0.90
    --limit-mm-per-prompt.image 4
    --limit-mm-per-prompt.video 0
    --enable-prefix-caching'
""" + s[end:]
s = re.sub(r"^  accelerator: .*$", f"  accelerator: {gpu}", s, flags=re.M)
open(dst, "w").write(s)
EOF
grep -n "rank-0\|accelerator:\|model_name" "$WORK/config.yaml"

echo "== pushing the curve deployment"
baseten model push --dir "$WORK" --environment production --wait --deploy-timeout 45m ${TEAM:+--team "$TEAM"}
MODEL_ID="$(baseten model list --output json | python3 -c 'import json,sys; ms=[m for m in json.load(sys.stdin)["models"] if m.get("name")=="understudy-cad-vl-curve"]; print(ms[0]["id"] if ms else "")')"
URL="https://model-$MODEL_ID.api.baseten.co/environments/production/sync/v1"
case "$GPU" in L4) PRICE=0.85 ;; H100_40GB) PRICE=3.75 ;; *) PRICE=6.50 ;; esac

echo "== benchmarking ${CKPTS//,/ } on the parts of $FRONTIER_RUN"
pids=()
for C in ${CKPTS//,/ }; do
  N="${C#checkpoint-}"
  UNDERSTUDY_BASE_URL="$URL" UNDERSTUDY_MODEL="$C" UNDERSTUDY_GPU_HOURLY="$PRICE" \
    uv run python -m understudy.cad.bench --modality image --lanes specialist --only "$FRONTIER_RUN" --n 0 --shots 0 \
    --concurrency 12 --workers 2 --tag "ckpt$N-img" > "runs/ckpt$N-img.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

for C in ${CKPTS//,/ }; do
  N="${C#checkpoint-}"
  echo "== $C"
  uv run python scripts/leaderboard.py "runs/$(ls runs | grep "_ckpt$N-img$" | tail -1)" --latest "${FRONTIER_RUN#*_}" --common | grep -E "common parts|ours" || true
done
echo "curve deployment: model $MODEL_ID (deactivate when done)"
