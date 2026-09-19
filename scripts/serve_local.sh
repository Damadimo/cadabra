#!/bin/bash
# Fallback when Baseten deployments are unavailable: serve the sheet model on this Mac (Apple silicon, MLX) behind the
# same OpenAI-compatible API, so the benchmark and the race UI work unchanged.
#
#   ./scripts/serve_local.sh <training_job_id>                  # downloads the job's merged/ weights, serves :8810
#   ./scripts/serve_local.sh <training_job_id> checkpoint-600   # a LoRA checkpoint: merged into the base locally first
#   ./scripts/serve_local.sh base                               # untuned Qwen/Qwen3-VL-4B-Instruct (the base-4b lane)
#
# Weights land in checkpoints/ (gitignored). bf16 4B on an M5 Pro decodes ~15-20 tok/s per stream: fine for demo races,
# slow for the full benchmark (use --only <frontier run> for the same 200 parts). Say "measured on a laptop" when quoting
# its latency; accuracy is the model's own.
set -euo pipefail
JOB="${1:?usage: serve_local.sh <training_job_id>|base [checkpoint]}"
CKPT="${2:-merged}"
PORT="${PORT:-8810}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${BASE:-Qwen/Qwen3-VL-4B-Instruct}"
cd "$ROOT"

if [ "$JOB" = "base" ]; then
  MODEL="$BASE"
else
  DEST="$ROOT/checkpoints/$JOB"
  mkdir -p "$DEST"
  echo "== downloading $CKPT of job $JOB"
  baseten train checkpoint files --job-id "$JOB" --output jsonl > "$DEST/files.jsonl"
  python3 - "$DEST" "$CKPT" <<'EOF'
import json, os, sys, urllib.request
dest, ckpt = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(os.path.join(dest, "files.jsonl")) if l.strip()]
got = 0
for r in rows:
    parts = r["relative_file_name"].strip("/").split("/")
    if ckpt not in parts:
        continue
    out = os.path.join(dest, *parts[parts.index(ckpt):])
    if os.path.exists(out) and os.path.getsize(out) == r["size_bytes"]:
        got += 1
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"  {'/'.join(parts[parts.index(ckpt):])} ({r['size_bytes'] / 1e6:.0f} MB)", flush=True)
    urllib.request.urlretrieve(r["url"], out + ".part")
    os.replace(out + ".part", out)
    got += 1
if not got:
    names = sorted({"/".join(r["relative_file_name"].split("/")[:-1]) for r in rows})
    sys.exit(f"no files for {ckpt!r}; folders in this job: {names[:20]}")
print(f"{got} files in {os.path.join(dest, ckpt)}")
EOF
  if [ "$CKPT" = "merged" ]; then
    MODEL="$DEST/merged"
  else
    MODEL="$DEST/merged-$CKPT"
    if [ ! -f "$MODEL/config.json" ]; then
      echo "== merging $CKPT into $BASE (CPU, a few minutes)"
      ADAPTER="$(dirname "$(find "$DEST/$CKPT" -name adapter_config.json | head -n1)")"
      BASE_MODEL="$BASE" uv run --no-project --python 3.12 --with-requirements "$ROOT/training/vlm/requirements_vlm.txt" python - "$ADAPTER" "$MODEL" <<'EOF'
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

sys.path.insert(0, "training/vlm")
from vlm_common import MODEL_ID, save_merged  # noqa: E402

base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cpu")
save_merged(PeftModel.from_pretrained(base, sys.argv[1]), sys.argv[2])
EOF
    fi
  fi
fi

cat <<EOF

== serving $MODEL on :$PORT. In another shell (or .env):
UNDERSTUDY_BASE_URL=http://127.0.0.1:$PORT/v1
UNDERSTUDY_MODEL=$MODEL
UNDERSTUDY_GPU_HOURLY=0   # a laptop: no GPU bill to amortize

EOF
exec uv run --no-project --python 3.12 --with mlx-vlm python -m mlx_vlm.server --model "$MODEL" --port "$PORT" --max-tokens 2048 --max-num-seqs 8
