#!/bin/bash
# Laptop smoke test for eval_sweep.py: two fresh LoRA adapters on a tiny random Qwen3-VL, graded on two fake sheets.
# Catches adapter discovery, attach/unload and output-path errors before they cost H100 time.
#   ./training/vlm/dry_run_eval_sweep.sh            # KEEP=1 keeps <repo>/.dryrun/eval_sweep
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="$ROOT/.dryrun/eval_sweep"  # outside training/vlm: a job push uploads that whole folder
rm -rf "$WORK" && mkdir -p "$WORK/data/images" "$WORK/cadcheck" "$WORK/adapters"
cp "$HERE/eval_sweep.py" "$HERE/bench_eval.py" "$HERE/vlm_common.py" "$WORK/"
cp "$ROOT/cadabra/cad/geometry.py" "$ROOT/cadabra/cad/pool.py" "$WORK/cadcheck/"
echo '"""geometry checker copy (dry run)"""' > "$WORK/cadcheck/__init__.py"
PY=(uv run --no-project --python "${DRY_PY:-3.12}" --with-requirements "$HERE/requirements_vlm.txt" --with cadquery --with trimesh --with scipy python)
cd "$WORK"

export BASE_MODEL=trl-internal-testing/tiny-Qwen3VLForConditionalGeneration
export TOKENIZERS_PARALLELISM=false REWARD_WORKERS=2 EVAL_N=2 EVAL_BATCH=2 EVAL_MAX_NEW=8
export ADAPTER_ROOT="$WORK/adapters" BT_CHECKPOINT_DIR="$WORK/ckpt"

# Two fake sheets (one solves, one does not) and two adapters mirrored the way Baseten lays named checkpoints out.
"${PY[@]}" - <<'PYEOF'
import json, os
from PIL import Image, ImageDraw
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText

SYSTEM = "You are an expert mechanical CAD engineer. Write CadQuery code for the part on the sheet. Assign it to `r`."
PARTS = [((0.75, 0.75, 0.1), 'r = cq.Workplane("XY").box(0.75, 0.75, 0.1)'),
         ((0.3, 0.3, 0.75), 'r = cq.Workplane("XY").circle(0.15).extrude(0.75)')]
rows = []
for i, (dims, code) in enumerate(PARTS):
    img = Image.new("RGB", (1024, 1024), "white")
    ImageDraw.Draw(img).rectangle([300, 300, 700, 700], fill=(120, 150, 200), outline=(20, 30, 60), width=4)
    img.save(f"data/images/part{i:04d}.png")
    rows.append({"id": f"part{i:04d}", "n_faces": 6, "n_parts": 1, "images": [f"images/part{i:04d}.png"],
                 "prompt": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Bounding box: X {}, Y {}, Z {}.".format(*dims)}]}],
                 "gold_code": f"import cadquery as cq\n\n{code}\n"})
with open("data/bench.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)

base = AutoModelForImageTextToText.from_pretrained(os.environ["BASE_MODEL"])
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=r".*language_model.*\.(q_proj|v_proj)", task_type="CAUSAL_LM")
for job, name in (("job0aaa", "checkpoint-10"), ("job0aaa", "checkpoint-20")):
    get_peft_model(base, cfg).save_pretrained(f"adapters/{job}/rank-0/{name}")
print("fake bench:", len(rows), "sheets; adapters:", sorted(os.listdir("adapters/job0aaa/rank-0")))
PYEOF

SHARDS=2 SHARD=0 "${PY[@]}" eval_sweep.py   # sharding: each process takes its share
SHARDS=2 SHARD=1 "${PY[@]}" eval_sweep.py
rm -rf ckpt/bench_eval
SHARDS=1 SHARD=0 "${PY[@]}" eval_sweep.py   # both adapters in one process: exercises attach -> grade -> unload
python3 - <<'PYEOF'
import json, os, sys
files = sorted(os.listdir("ckpt/bench_eval"))
assert files == ["job0aaa-rank-0-checkpoint-10.json", "job0aaa-rank-0-checkpoint-20.json"], files
for name in files:
    s = json.load(open(f"ckpt/bench_eval/{name}"))["summary"]
    assert s["n"] == 2 and "all" in s, s
    print(name, json.dumps(s["all"]), "run_rate", s["run_rate"])
PYEOF
cd "$ROOT"
[ "${KEEP:-0}" = "1" ] || rm -rf "$WORK"
echo "eval sweep dry run OK"
