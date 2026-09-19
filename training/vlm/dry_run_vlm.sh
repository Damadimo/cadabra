#!/bin/bash
# Laptop smoke test for train_vlm.py: 2 CPU steps with a tiny random Qwen3-VL on a fake 4-row dataset (PIL-drawn
# 1024x1024 sheets, realistic prompt/completion text), including the merged save. Then it reloads merged/ and checks
# that it matches base + adapter. Catches data-format, processor and trainer-argument errors before they cost H100 time.
#   ./training/vlm/dry_run_vlm.sh            # KEEP=1 keeps <repo>/.dryrun/vlm (data + checkpoints) for inspection
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$HERE/../../.dryrun/vlm"  # outside training/vlm: a job push uploads that whole folder
rm -rf "$WORK" && mkdir -p "$WORK/data/images"
cp "$HERE/train_vlm.py" "$HERE/vlm_common.py" "$HERE/bench_eval.py" "$WORK/"
mkdir -p "$WORK/cadcheck" && cp "$HERE/../../understudy/cad/geometry.py" "$HERE/../../understudy/cad/pool.py" "$WORK/cadcheck/"
echo '"""geometry checker copy (dry run)"""' > "$WORK/cadcheck/__init__.py"
PY=(uv run --no-project --python "${DRY_PY:-3.12}" --with-requirements "$HERE/requirements_vlm.txt" --with cadquery --with trimesh --with scipy python)
cd "$WORK"

# Fake dataset in the coordinator's on-disk format: 4 train rows, 2 val rows.
"${PY[@]}" - <<'EOF'
import json
from PIL import Image, ImageDraw

SYSTEM = (
    "You are an expert mechanical CAD engineer. The image is a 2x2 sheet of one part rendered from four views "
    "(top-left: front, top-right: top, bottom-left: right, bottom-right: isometric). Write CadQuery (Python) code "
    "that builds exactly this part.\n\nOutput rules:\n"
    "- Reply with the complete program in one ```python code block and nothing else.\n"
    "- Use `import cadquery as cq` (and `math` if needed); import nothing else.\n"
    "- Assign the final solid to a variable named `r`. Do not export files or call show_object."
)
PARTS = [
    ((0.75, 0.03, 0.12), 'r = cq.Workplane("XY").box(0.75, 0.03, 0.12)'),
    ((0.75, 0.75, 0.1), 'r = (\n    cq.Workplane("XY")\n    .box(0.75, 0.75, 0.1)\n    .faces(">Z")\n    .workplane()\n    .hole(0.2)\n)'),
    ((0.3, 0.3, 0.75), 'r = cq.Workplane("XY").circle(0.15).extrude(0.75)'),
    ((0.6, 0.4, 0.05), 'r = (\n    cq.Workplane("XY")\n    .rect(0.6, 0.4)\n    .extrude(0.05)\n    .faces(">Z")\n    .workplane()\n    .rarray(0.4, 0.2, 2, 2)\n    .hole(0.05)\n)'),
    ((0.5, 0.25, 0.5), 'base = cq.Workplane("XY").box(0.5, 0.25, 0.1).translate((0, 0, 0.05))\nwall = cq.Workplane("XY").box(0.1, 0.25, 0.5).translate((-0.2, 0, 0.25))\nr = base.union(wall)'),
    ((0.2, 0.2, 0.2), 'r = cq.Workplane("XY").box(0.2, 0.2, 0.2).edges("|Z").fillet(0.02)'),
]


def sheet(dims, path):
    img = Image.new("RGB", (1024, 1024), "white")
    d = ImageDraw.Draw(img)
    d.line([(512, 0), (512, 1023)], fill=(160, 160, 160), width=3)
    d.line([(0, 512), (1023, 512)], fill=(160, 160, 160), width=3)
    x, y, z = dims
    s = 380 / max(dims)
    for (ox, oy), (w, h) in zip([(0, 0), (512, 0), (0, 512)], [(x, z), (x, y), (y, z)]):
        cx, cy, hw, hh = ox + 256, oy + 256, max(w * s / 2, 3), max(h * s / 2, 3)
        d.rectangle([cx - hw, cy - hh, cx + hw, cy + hh], fill=(120, 150, 200), outline=(20, 30, 60), width=4)
    d.polygon([(620, 820), (760, 740), (900, 820), (760, 900)], fill=(150, 180, 220), outline=(20, 30, 60))
    d.ellipse([730, 790, 790, 850], outline=(20, 30, 60), width=4)
    img.save(path)


rows = []
for i, (dims, code) in enumerate(PARTS):
    sheet(dims, f"data/images/part{i:04d}.png")
    rows.append({
        "images": [f"images/part{i:04d}.png"],
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Bounding box: X {}, Y {}, Z {} (units). Write the CadQuery program.".format(*dims)},
            ]},
        ],
        "completion": [{"role": "assistant", "content": f"```python\nimport cadquery as cq\n\n{code}\n```"}],
    })
for name, part in (("train", rows[:4]), ("val", rows[4:])):
    with open(f"data/{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in part)
with open("data/bench.jsonl", "w") as f:  # in-job benchmark eval (bench_eval.py): prompt + reference, no completion
    for i, r in enumerate(rows[4:]):
        code = r["completion"][0]["content"].split("```python\n")[1].rsplit("```", 1)[0]
        f.write(json.dumps({"id": f"bench:{i}", "images": r["images"], "prompt": r["prompt"], "gold_code": code, "n_faces": 6, "n_parts": 1}) + "\n")
print("fake dataset:", len(rows[:4]), "train rows,", len(rows[4:]), "val rows")
EOF

# LR is high on purpose: after 2 steps the adapter must visibly change the logits for the merge check below.
export CPU_DRY_RUN=1 MERGE_AT_END=1 TOKENIZERS_PARALLELISM=false
export BASE_MODEL=trl-internal-testing/tiny-Qwen3VLForConditionalGeneration
export MAX_STEPS=2 BATCH=2 GRAD_ACCUM=1 SAVE_STEPS=1 LOG_STEPS=1 LR=5e-3 BT_CHECKPOINT_DIR="$WORK/ckpt"
export EVAL_BENCH=1 EVAL_N=2 EVAL_BATCH=2 EVAL_MAX_NEW=16 REWARD_WORKERS=2
"${PY[@]}" train_vlm.py
"${PY[@]}" bench_eval.py "$WORK/ckpt/merged"  # as run_vlm.sh does after training

# merged/ must load on its own, keep the pinned image budget, and equal base + LoRA adapter.
"${PY[@]}" - <<'EOF'
import json, os
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ckpt = os.environ["BT_CHECKPOINT_DIR"]
merged_dir = os.path.join(ckpt, "merged")
last = max((d for d in os.listdir(ckpt) if d.startswith("checkpoint-")), key=lambda d: int(d.split("-")[1]))
row = json.loads(open("data/val.jsonl").readline())
proc = AutoProcessor.from_pretrained(merged_dir)
messages = row["prompt"]
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image = Image.open(os.path.join("data", row["images"][0])).convert("RGB")
inputs = proc(text=[text], images=[image], return_tensors="pt")

merged = AutoModelForImageTextToText.from_pretrained(merged_dir, dtype=torch.float32).eval()
n_image = int((inputs["input_ids"] == merged.config.image_token_id).sum())
assert n_image == 1024, f"expected 1024 visual tokens from merged/ processor, got {n_image}"
base = AutoModelForImageTextToText.from_pretrained(os.environ["BASE_MODEL"], dtype=torch.float32).eval()
with torch.no_grad():
    base_logits = base(**inputs).logits
    adapted_logits = PeftModel.from_pretrained(base, os.path.join(ckpt, last)).eval()(**inputs).logits
    merged_logits = merged(**inputs).logits
d_merge = (merged_logits - adapted_logits).abs().max().item()
d_base = (merged_logits - base_logits).abs().max().item()
assert d_merge < 1e-4 < d_base, (d_merge, d_base)
out = merged.generate(**inputs, max_new_tokens=6, do_sample=False)
print(f"merged/ check OK: {n_image} visual tokens; |merged - (base+{last})| = {d_merge:.1e}, |merged - base| = {d_base:.1e}; "
      f"greedy sample {proc.batch_decode(out[:, inputs['input_ids'].shape[1]:])[0]!r}")
EOF

ls "$WORK/ckpt" "$WORK/ckpt/merged"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('bench_eval summary:', d['summary']); assert d['summary']['n'] == 2" "$WORK/ckpt/bench_eval/results.json"
python3 -c "import json,sys; h=json.load(open(sys.argv[1])); print('train_log.json:', [{k: round(v, 4) for k, v in e.items() if k in ('step','loss','eval_loss','train_loss')} for e in h])" "$WORK/ckpt/train_log.json"
du -sh "$WORK"
[ "${KEEP:-0}" = "1" ] || rm -rf "$WORK"
echo "VLM dry run OK"
