#!/bin/bash
# Deploy the fine-tuned drawing-sheet model from a finished Baseten Training job, then smoke-test it.
#
#   ./scripts/deploy_vlm.sh <training_job_id> [accelerator]                 # merged weights (end of training)
#   ./scripts/deploy_vlm.sh <training_job_id> [accelerator] <checkpoint>    # base + LoRA from e.g. checkpoint-600
#   accelerator: L4 (default) | H100_40GB | H100
#   TEAM=22 ./scripts/deploy_vlm.sh ...   puts the model in our Hack the North team (next to the training checkpoints)
#
# Needs a payment method on the workspace (Baseten refuses model deploys without one) and the job's
# $BT_CHECKPOINT_DIR/merged folder fully synced. Prints the .env lines for the benchmark and the race UI.
set -euo pipefail
JOB_ID="${1:?usage: deploy_vlm.sh <training_job_id> [accelerator] [checkpoint]}"
GPU="${2:-L4}"
CKPT="${3:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${PROJECT:-understudy-cad-vlm-sft}"  # PROJECT=understudy-cad-vlm-grpo for the RL stage

echo "== checkpoints of $JOB_ID (need one named 'merged', fully synced)"
baseten train checkpoint list --job-id "$JOB_ID" --output json | tee /tmp/understudy_ckpts.json | head -40 || true

WORK="$(mktemp -d)"
if [ -n "$CKPT" ]; then
  cp "$ROOT/deploy/vlm_lora/config.yaml" "$WORK/config.yaml"
  NAME="understudy-cad-vl-lora"
  sed -i '' -e "s#TRAINING_JOB_ID#$JOB_ID#" -e "s#CHECKPOINT_NAME#$CKPT#" \
            -e "s#^  accelerator: L4.*#  accelerator: $GPU#" "$WORK/config.yaml"
else
  cp "$ROOT/deploy/vlm_ft/config.yaml" "$WORK/config.yaml"
  NAME="understudy-cad-vl"
  sed -i '' -e "s#TRAINING_JOB_ID#$JOB_ID#" \
            -e "s#^  accelerator: L4.*#  accelerator: $GPU#" "$WORK/config.yaml"
fi
grep -n "training_job_id\|rank-0\|accelerator:" "$WORK/config.yaml"

echo "== pushing (first deploy pulls ~9 GB of weights; allow 10-20 min)"
baseten model push --dir "$WORK" --wait --deploy-timeout 45m ${TEAM:+--team "$TEAM"}  # TEAM=22: our event team

MODEL_ID="$(baseten model list --output json | NAME="$NAME" python3 -c 'import json,os,sys; ms=[m for m in json.load(sys.stdin)["models"] if m.get("name")==os.environ["NAME"]]; print(ms[0]["id"] if ms else "")')"
[ -n "$MODEL_ID" ] || { echo "could not find the model id: run 'baseten model list'"; exit 1; }
case "$GPU" in L4) PRICE=0.85 ;; H100_40GB) PRICE=3.75 ;; *) PRICE=6.50 ;; esac

cat <<EOF

== add to .env
UNDERSTUDY_BASE_URL=https://model-$MODEL_ID.api.baseten.co/environments/production/sync/v1
UNDERSTUDY_MODEL=understudy-cad-vl
UNDERSTUDY_GPU_HOURLY=$PRICE
UNDERSTUDY_BASE_MODEL=Qwen/Qwen3-VL-4B-Instruct   # base-4b lane (served on the same endpoint in LoRA mode)
RACE_LANES=specialist,moonshotai/Kimi-K3:high,zai-org/GLM-5.3-Flash:high

== then
uv run python deploy/test_vlm.py                                   # one sheet, latency
uv run python -m understudy.cad.bench --modality image --lanes specialist --n 500 --shots 0 --concurrency 16 --tag ours-img
EOF
