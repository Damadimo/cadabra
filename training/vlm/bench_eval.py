"""Score the trained model on the held-out benchmark sheets inside the training job (greedy, one sample per part).

Run by run_vlm.sh / run_grpo_vlm.sh after training exits (EVAL_BENCH=1, data/bench.jsonl present), as its own process,
on the saved merged/ weights, so it can never cost them. Writes $BT_CHECKPOINT_DIR/bench_eval/results.json
(summary + per-part rows with the code; synced like a checkpoint, so `baseten train checkpoint files` lists it) and
prints the summary to the job log. The same geometry checker and success rule as
understudy/cad/bench.py (code runs and aligned IoU >= 0.9). It is a safety net and an early read; the official numbers
come from the deployed model via understudy/cad/bench.py.

  python bench_eval.py <model dir or HF id>          # e.g. $BT_CHECKPOINT_DIR/merged, or the base model for comparison
"""

import json
import os
import sys
import time

import torch
from PIL import Image as PILImage

from vlm_common import DATA_DIR, IMAGE_PIXELS, as_parts, image_path

EVAL_N = int(os.getenv("EVAL_N", "500"))
EVAL_BATCH = int(os.getenv("EVAL_BATCH", "32"))
TIERS = {
    "all": lambda r: True,
    "simple (<=6 faces)": lambda r: (r.get("n_faces") or 0) <= 6,
    "medium (7-12)": lambda r: 7 <= (r.get("n_faces") or 0) <= 12,
    "complex (>=13)": lambda r: (r.get("n_faces") or 0) >= 13,
    "multi-part": lambda r: r["n_parts"] > 1,
}


def extract_code(text):
    import re

    blocks = [b for b in re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S) if "cq" in b or "cadquery" in b]
    if blocks:
        return blocks[-1].strip() + "\n"
    return text[text.index("import cadquery"):] if "import cadquery" in text else None


def run(model, processor, out_path):
    path = os.path.join(DATA_DIR, "bench.jsonl")
    if not os.path.exists(path):
        print(f"[bench] no {path}: skipped")
        return None
    rows = [json.loads(line) for line in open(path) if line.strip()][:EVAL_N]
    t0 = time.time()
    model.eval()
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    texts = []
    for start in range(0, len(rows), EVAL_BATCH):
        batch = rows[start : start + EVAL_BATCH]
        prompts = [processor.apply_chat_template([as_parts(m) for m in r["prompt"]], tokenize=False, add_generation_prompt=True) for r in batch]
        images = [PILImage.open(image_path(r["images"][0])).convert("RGB") for r in batch]
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=int(os.getenv("EVAL_MAX_NEW", "1536")), do_sample=False)
        texts += processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"[bench] generated {len(texts)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    gen_s = time.time() - t0

    from cadcheck.pool import CadPool

    codes = [extract_code(t) for t in texts]
    with CadPool(workers=int(os.getenv("REWARD_WORKERS", max(2, (os.cpu_count() or 4) - 2))), timeout=30) as pool:
        graded = pool.map([{"code": c or "", "gold_code": r["gold_code"], "want_chamfer": False} for c, r in zip(codes, rows)])
    results = []
    for r, code, g in zip(rows, codes, graded):
        iou = g.get("iou_aligned") if g.get("runs") and code else None
        results.append({"id": r["id"], "n_faces": r.get("n_faces"), "n_parts": r["n_parts"], "runs": bool(code and g.get("runs")),
                        "iou_aligned": iou, "success": bool(iou is not None and iou >= 0.9), "error": None if code else "no code block",
                        "code": code})
    summary = {"n": len(results), "generation_s": round(gen_s, 1), "image_pixels": IMAGE_PIXELS, "decoding": "greedy, 1 sample"}
    for name, f in TIERS.items():
        part = [x for x in results if f(x)]
        summary[name] = {"n": len(part), "success": round(sum(x["success"] for x in part) / len(part), 4) if part else None}
    summary["run_rate"] = round(sum(x["runs"] for x in results) / max(len(results), 1), 4)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": results}, f)
    print("[bench] " + json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    from transformers import AutoModelForImageTextToText, AutoProcessor

    src = sys.argv[1]
    proc = AutoProcessor.from_pretrained(src, min_pixels=IMAGE_PIXELS, max_pixels=IMAGE_PIXELS)
    mdl = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                                                      device_map="auto" if torch.cuda.is_available() else "cpu")
    run(mdl, proc, os.path.join(os.getenv("BT_CHECKPOINT_DIR", "."), "bench_eval", "results.json"))
