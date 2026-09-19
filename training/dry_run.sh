#!/bin/bash
# Laptop smoke test for train.py: 2 steps on CPU with a tiny random Qwen3 and the synthetic samples.
# Catches trainer-argument and data-format errors before they cost H100 time. Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/.dryrun"
rm -rf "$WORK" && mkdir -p "$WORK/data"
cp "$ROOT/training/train.py" "$WORK/"

cd "$ROOT" && uv run python - "$WORK" <<'EOF'
import json, sys
from understudy.build_sft import example
from understudy.data import read_jsonl
from understudy.tasks import load_task

task = load_task("insurance")
rows = [example(task, r, r["label"]) for r in read_jsonl("data/samples/insurance_gold_SAMPLE.jsonl")]
for name, subset in (("train", rows * 2), ("val", rows[:1])):
    with open(f"{sys.argv[1]}/data/{name}.jsonl", "w") as f:
        for r in subset:
            f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")
EOF

cd "$WORK"
CPU_DRY_RUN=1 TOKENIZERS_PARALLELISM=false BASE_MODEL=trl-internal-testing/tiny-Qwen3ForCausalLM \
MAX_STEPS=2 BATCH=1 GRAD_ACCUM=1 MAX_LEN=2048 SAVE_STEPS=1 BT_CHECKPOINT_DIR="$WORK/ckpt" \
  uv run --no-project --python 3.12 --with "trl>=0.20.0" --with "peft>=0.17.0" --with "transformers>=4.55.0" \
  --with datasets --with accelerate --with torch python train.py
ls "$WORK/ckpt"
echo "Dry run OK"
