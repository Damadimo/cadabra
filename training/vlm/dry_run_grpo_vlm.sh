#!/bin/bash
# Laptop smoke test for grpo_vlm.py: the reward on known answers, then 1 GRPO step on CPU with a tiny random Qwen3-VL
# (fresh LoRA, fake sheets). Catches data-format, reward and trainer-argument errors before they cost H100 time.
#   ./training/vlm/dry_run_grpo_vlm.sh            # KEEP=1 keeps <repo>/.dryrun/grpo_vlm
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="$ROOT/.dryrun/grpo_vlm"  # outside training/vlm: a job push uploads that whole folder
rm -rf "$WORK" && mkdir -p "$WORK/data/images" "$WORK/cadcheck"
cp "$HERE/grpo_vlm.py" "$HERE/vlm_common.py" "$WORK/"
cp "$ROOT/understudy/cad/geometry.py" "$ROOT/understudy/cad/pool.py" "$WORK/cadcheck/"
echo '"""geometry checker copy (dry run)"""' > "$WORK/cadcheck/__init__.py"
PY=(uv run --no-project --python "${DRY_PY:-3.12}" --with-requirements "$HERE/requirements_vlm.txt" --with cadquery --with trimesh --with scipy python)
cd "$WORK"

"${PY[@]}" - <<'PYEOF'
import json
from PIL import Image, ImageDraw

SYSTEM = "You are an expert mechanical CAD engineer. Write CadQuery code for the part on the sheet. Assign it to `r`."
PARTS = [
    ((0.75, 0.75, 0.1), 'r = cq.Workplane("XY").box(0.75, 0.75, 0.1).faces(">Z").workplane().hole(0.2)'),
    ((0.3, 0.3, 0.75), 'r = cq.Workplane("XY").circle(0.15).extrude(0.75)'),
]
rows = []
for i, (dims, code) in enumerate(PARTS):
    img = Image.new("RGB", (1024, 1024), "white")
    ImageDraw.Draw(img).rectangle([300, 300, 700, 700], fill=(120, 150, 200), outline=(20, 30, 60), width=4)
    img.save(f"data/images/part{i:04d}.png")
    rows.append({
        "images": [f"images/part{i:04d}.png"],
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Bounding box: X {}, Y {}, Z {}.".format(*dims)}]}],
        "gold_code": f"import cadquery as cq\n\n{code}\n",
    })
with open("data/grpo.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
print("fake grpo set:", len(rows), "prompts")
PYEOF

export CPU_DRY_RUN=1 MERGE_AT_END=1 TOKENIZERS_PARALLELISM=false SFT_ADAPTER="" REWARD_WORKERS=2
export BASE_MODEL=trl-internal-testing/tiny-Qwen3VLForConditionalGeneration
export MAX_STEPS=1 BATCH=2 GRAD_ACCUM=1 NUM_GENERATIONS=2 MAX_COMPLETION=24 SAVE_STEPS=1 BT_CHECKPOINT_DIR="$WORK/ckpt"
export GRADIENT_CHECKPOINTING=1  # the H100 setting: exercises SheetGRPOTrainer's eval-mode rollouts

# The reward on known answers: the reference itself, a wrong size, a crash, and no code at all. (A file, not stdin:
# the reward's worker processes re-import the parent script when they spawn, as on macOS.)
cat > reward_check.py <<'PYEOF'
import json

import grpo_vlm as g


def main():
    rows = [json.loads(l) for l in open("data/grpo.jsonl")]
    gold = rows[0]["gold_code"]
    answers = [
        [{"role": "assistant", "content": "```python\n" + gold + "```"}],
        [{"role": "assistant", "content": "```python\n" + gold.replace("0.75, 0.75", "0.9, 0.9") + "```"}],
        [{"role": "assistant", "content": "```python\nimport cadquery as cq\nr = cq.Workplane('XY').box(1, 1, 1).fillet(5)\n```"}],
        [{"role": "assistant", "content": "I cannot do that."}],
    ]
    rewards = g.geometry_reward(None, answers, [gold] * 4)
    success = g.success_metric(None, answers, [gold] * 4)
    print("rewards (reference, wrong size, crash, no code):", [round(r, 3) for r in rewards], "success:", success)
    assert rewards[0] >= 1.49 and 0 < rewards[1] < 0.9 and rewards[2] == -0.2 and rewards[3] == -0.5, rewards
    assert success == [1.0, 0.0, 0.0, 0.0], success
    g._pool.close()


if __name__ == "__main__":  # spawned reward workers re-import this file; only the parent runs the check
    main()
PYEOF
"${PY[@]}" reward_check.py

if [ "${DRY_NPROC:-1}" -gt 1 ]; then  # DDP code path (gloo on CPU), as run_grpo_vlm.sh does with NPROC>1
  uv run --no-project --python "${DRY_PY:-3.12}" --with-requirements "$HERE/requirements_vlm.txt" --with cadquery --with trimesh --with scipy \
    torchrun --nnodes=1 --nproc_per_node="$DRY_NPROC" --master_addr=127.0.0.1 --master_port=29512 grpo_vlm.py
else
  "${PY[@]}" grpo_vlm.py
fi
python3 -c "import json,sys; h=json.load(open(sys.argv[1])); print('grpo_log.json:', [{k: round(v, 4) if isinstance(v, float) else v for k, v in e.items() if k in ('step','loss','reward','rewards/geometry_reward/mean','rewards/success_metric/mean')} for e in h])" "$WORK/ckpt/grpo_log.json"
ls "$WORK/ckpt" "$WORK/ckpt/merged"
[ "${KEEP:-0}" = "1" ] || rm -rf "$WORK"
echo "GRPO VLM dry run OK"
