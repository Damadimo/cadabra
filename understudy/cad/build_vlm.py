"""Build Baseten training data for the drawing-sheet (vision) variant.

  uv run python -m understudy.cad.render --split train      # sheets for the training parts (cached)
  uv run python -m understudy.cad.build_vlm

Writes training/vlm/data/{train,val}.jsonl and copies the sheets into training/vlm/data/images/ (the job uploads
that folder). Rows: {"images": ["images/<id>.png"], "prompt": [system, user(image + bounding box)], "completion":
[assistant code]}: exactly the messages the benchmark sends a zero-shot lane, with the image as a content part.
"""

from __future__ import annotations

import argparse
import json
import shutil

from ..config import ROOT
from ..data import read_jsonl, split_by_source, write_jsonl
from . import prompts
from .render import OUT as IMAGES
from .render import image_path

OUT = ROOT / "training" / "vlm" / "data"


def row(rec: dict, bbox: list[float]) -> dict:
    name = image_path(rec["id"], IMAGES).name
    return {
        "images": [f"images/{name}"],
        "prompt": [
            {"role": "system", "content": prompts.SYSTEM_IMAGE},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompts.image_request(bbox)}]},
        ],
        "completion": [{"role": "assistant", "content": prompts.completion(rec["gold_code"])}],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-extra", action="store_true", help="only train_high parts (skip the train_middle extras)")
    args = ap.parse_args()

    index = json.loads((IMAGES / "index.json").read_text())
    bench_sources = {r["source_id"] for r in read_jsonl(ROOT / "data" / "cad" / "bench.jsonl")}
    shot_ids = {r["id"] for r in read_jsonl(ROOT / "data" / "cad" / "shots.jsonl")}
    pool = read_jsonl(ROOT / "data" / "cad" / "train.jsonl")
    if not args.no_extra:
        pool += read_jsonl(ROOT / "data" / "cad" / "train_vlm_extra.jsonl")
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
    write_jsonl(OUT / "val.jsonl", [row(r, index[r["id"]]["bbox"]) for r in val])
    size_mb = sum(p.stat().st_size for p in (OUT / "images").iterdir()) / 1e6
    stats = {"train": len(fit), "val": len(val), "images_mb": round(size_mb, 1), "skipped_unrendered_or_excluded": missing,
             "multi_part": sum(r["n_parts"] > 1 for r in fit), "complex_7plus_faces": sum(r.get("n_faces", 0) >= 7 for r in fit)}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
