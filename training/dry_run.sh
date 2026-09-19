#!/bin/bash
# Laptop smoke test for the training scripts: a few CPU steps with a tiny random Qwen3 on real training rows.
# Catches trainer-argument and data-format errors before they cost H100 time. Run from anywhere:
#   ./training/dry_run.sh          # SFT (train.py)
#   ./training/dry_run.sh grpo     # RL (grpo.py, without vLLM)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/.dryrun"
STAGE="${1:-sft}"
[ -f "$ROOT/training/data/train.jsonl" ] || (cd "$ROOT" && uv run python -m understudy.cad.build_sft)
rm -rf "$WORK" && mkdir -p "$WORK/data"
cp "$ROOT/training/train.py" "$ROOT/training/grpo.py" "$WORK/"
cp -r "$ROOT/training/cadcheck" "$WORK/"
head -n 8 "$ROOT/training/data/train.jsonl" > "$WORK/data/train.jsonl"
head -n 2 "$ROOT/training/data/val.jsonl" > "$WORK/data/val.jsonl"
head -n 4 "$ROOT/training/data/grpo.jsonl" > "$WORK/data/grpo.jsonl"

cd "$WORK"
export CPU_DRY_RUN=1 TOKENIZERS_PARALLELISM=false BASE_MODEL=trl-internal-testing/tiny-Qwen3ForCausalLM
export MAX_STEPS=2 BATCH=2 GRAD_ACCUM=1 MAX_LEN=4096 SAVE_STEPS=1 BT_CHECKPOINT_DIR="$WORK/ckpt"
DEPS=(--with "trl>=0.20.0" --with "peft>=0.17.0" --with "transformers>=4.55.0" --with datasets --with accelerate --with torch)
if [ "$STAGE" = "grpo" ]; then
  export USE_VLLM=0 NUM_GENERATIONS=2 MAX_COMPLETION=48 REWARD_WORKERS=2 SFT_ADAPTER=
  uv run --no-project --python 3.12 "${DEPS[@]}" --with cadquery --with trimesh --with scipy python grpo.py
else
  uv run --no-project --python 3.12 "${DEPS[@]}" python train.py
fi
ls "$WORK/ckpt"
echo "Dry run ($STAGE) OK"
