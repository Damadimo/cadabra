"""Build Baseten training data for the drawing-sheet (vision) variant.

  uv run python -m understudy.cad.render --split train      # sheets for the training parts (cached)
  uv run python -m understudy.cad.build_vlm

Writes training/vlm/data/{train,val}.jsonl and copies the sheets into training/vlm/data/images/ (the job uploads
that folder). Rows: {"images": ["images/<id>.png"], "prompt": [system, user(image + bounding box)], "completion":
[assistant code]}: exactly the messages the benchmark sends a zero-shot lane, with the image as a content part.
Also data/grpo.jsonl (sheets with their reference code, mostly medium/complex, for the optional RL stage) and a copy
of the geometry checker in training/vlm/cadcheck/ for its reward.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil

from ..config import ROOT
from ..data import read_jsonl, split_by_source, write_jsonl
from . import prompts
from .render import OUT as IMAGES
from .render import image_path

OUT = ROOT / "training" / "vlm" / "data"


def tidy(code: str) -> str:
    """Drop what the dataset left behind where its display call was: trailing comment-only lines ("# Display the
    final model"), print(...) and bare-name lines. Geometry is untouched; the target just ends at the last real line."""
    lines = code.rstrip().split("\n")
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith(("#", "print(")) or lines[-1].split("#")[0].strip().isidentifier()):
        lines.pop()
    return "\n".join(lines) + "\n"


def row(rec: dict, bbox: list[float]) -> dict:
    name = image_path(rec["id"], IMAGES).name
    return {
        "images": [f"images/{name}"],
        "prompt": [
            {"role": "system", "content": prompts.SYSTEM_IMAGE},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompts.image_request(bbox)}]},
        ],
        "completion": [{"role": "assistant", "content": prompts.completion(tidy(rec["gold_code"]))}],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--grpo-size", type=int, default=2000)
    ap.add_argument("--no-extra", action="store_true", help="only train_high parts (skip the train_middle extras)")
    ap.add_argument("--extras", default="train_vlm_extra,train_vlm_extra2", help="extra splits to add (unrendered rows are skipped)")
    args = ap.parse_args()

    index = json.loads((IMAGES / "index.json").read_text())
    bench_sources = {r["source_id"] for r in read_jsonl(ROOT / "data" / "cad" / "bench.jsonl")}
    shot_ids = {r["id"] for r in read_jsonl(ROOT / "data" / "cad" / "shots.jsonl")}
    pool = read_jsonl(ROOT / "data" / "cad" / "train.jsonl")
    if not args.no_extra:
        for name in filter(None, args.extras.split(",")):
            path = ROOT / "data" / "cad" / f"{name}.jsonl"
            pool += read_jsonl(path) if path.exists() else []
    recs = [
        r
        for r in pool[: args.limit]
        if r["id"] in index and r["source_id"] not in bench_sources and r["id"] not in shot_ids and image_path(r["id"], IMAGES).exists()
    ]
    missing = len(pool[: args.limit]) - len(recs)
    fit, val = split_by_source(recs, args.val_frac, seed=args.seed)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for r in fit + val:
        dst = OUT / "images" / image_path(r["id"], IMAGES).name
        if not dst.exists():
            shutil.copy(image_path(r["id"], IMAGES), dst)
    write_jsonl(OUT / "train.jsonl", [row(r, index[r["id"]]["bbox"]) for r in fit])
    rng = random.Random(args.seed)
    hard = [r for r in fit if r.get("n_faces", 0) >= 7 or r["n_parts"] > 1]
    easy = [r for r in fit if not (r.get("n_faces", 0) >= 7 or r["n_parts"] > 1)]
    k_hard = min(len(hard), int(args.grpo_size * 0.8))
    picked = rng.sample(hard, k_hard) + rng.sample(easy, min(len(easy), args.grpo_size - k_hard))
    write_jsonl(OUT / "grpo.jsonl", [{**{k: v for k, v in row(r, index[r["id"]]["bbox"]).items() if k != "completion"}, "gold_code": r["gold_code"]} for r in picked])
    # Held-out benchmark sheets for the optional in-job eval (bench_eval.py). Never read by the training loaders.
    (OUT / "bench_images").mkdir(exist_ok=True)
    bench = [r for r in read_jsonl(ROOT / "data" / "cad" / "bench.jsonl") if r["id"] in index]
    for r in bench:
        dst = OUT / "bench_images" / image_path(r["id"], IMAGES).name
        if not dst.exists():
            shutil.copy(image_path(r["id"], IMAGES), dst)
    write_jsonl(OUT / "bench.jsonl", [{**{k: v for k, v in row(r, index[r["id"]]["bbox"]).items() if k != "completion"},
                                       "images": [f"bench_images/{image_path(r['id'], IMAGES).name}"],
                                       "id": r["id"], "gold_code": r["gold_code"], "n_faces": r.get("n_faces"), "n_parts": r["n_parts"]}
                                      for r in bench])
    cadcheck = OUT.parent / "cadcheck"
    cadcheck.mkdir(exist_ok=True)
    for name in ("geometry.py", "pool.py"):
        shutil.copy(ROOT / "understudy" / "cad" / name, cadcheck / name)
    (cadcheck / "__init__.py").write_text('"""Copy of understudy/cad geometry checker for the RL reward (generated by build_vlm)."""\n')
    write_jsonl(OUT / "val.jsonl", [row(r, index[r["id"]]["bbox"]) for r in val])
    size_mb = sum(p.stat().st_size for p in (OUT / "images").iterdir()) / 1e6
    stats = {"train": len(fit), "val": len(val), "images_mb": round(size_mb, 1), "skipped_unrendered_or_excluded": missing,
             "multi_part": sum(r["n_parts"] > 1 for r in fit), "complex_7plus_faces": sum(r.get("n_faces", 0) >= 7 for r in fit),
             "grpo_prompts": len(picked), "grpo_hard_share": round(k_hard / max(len(picked), 1), 2), "bench_eval_parts": len(bench)}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
