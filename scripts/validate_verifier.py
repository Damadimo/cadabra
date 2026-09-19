"""Does render-and-compare (which only sees the input drawing and bounding box) agree with the answer-key IoU?

  uv run python scripts/validate_verifier.py runs/<image-run> [runs/<image-run> ...]

For every answer in the runs that built a solid, compute its render score against the input sheet and compare with
its aligned IoU against the reference: AUC of each signal (silhouette, edges, size, combined), how the reference
programs themselves score (should be ~1.0), and a best-of selection test: for parts with several answers (different
lanes/runs), how often the top-scored answer is correct vs the best available (oracle) and a random pick.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from understudy.cad.pool import CadPool
from understudy.cad.render import OUT as IMAGES
from understudy.cad.render import image_path
from understudy.cad.verify import combine
from understudy.config import ROOT
from understudy.data import read_jsonl

SIGNALS = {
    "silhouette only (v1)": lambda c: c["silhouette"],
    "edges only": lambda c: c["edges"],
    "size only": lambda c: c["size"],
    "silhouette x size (scored)": combine,
    "(silhouette + edges) / 2 x size": lambda c: (c["silhouette"] + c["edges"]) / 2 * c["size"],
}


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    bench = {r["id"]: r for r in read_jsonl(ROOT / "data" / "cad" / "bench.jsonl")}
    index = json.loads((IMAGES / "index.json").read_text())
    rows = [r for d in args.runs for r in read_jsonl((d if d.is_absolute() else ROOT / d) / "results.jsonl") if r["runs"] and r.get("code")]
    ids = sorted({r["id"] for r in rows})

    def task(code: str, rec_id: str) -> dict:
        return {"_fn": "render_score", "code": code, "sheet_path": str(image_path(rec_id, IMAGES)), "bbox": index[rec_id]["bbox"]}

    with CadPool(workers=args.workers, timeout=60) as pool:
        res = pool.map([task(r["code"], r["id"]) for r in rows] + [task(bench[i]["gold_code"], i) for i in ids])
    cand, gold = res[: len(rows)], res[len(rows):]
    keep = [(r, c) for r, c in zip(rows, cand) if c.get("runs")]
    ok = [(r["iou_aligned"] or 0) >= args.threshold for r, _ in keep]
    print(f"answers scored: {len(keep)} ({sum(ok)} correct, {len(ok) - sum(ok)} wrong by the answer key) on {len(ids)} parts")
    gs = sorted(combine(g) for g in gold if g.get("runs"))
    print(f"reference programs, combined score: min {gs[0]:.3f}, p10 {gs[len(gs) // 10]:.3f}, median {gs[len(gs) // 2]:.3f}")
    for name, f in SIGNALS.items():
        good = [f(c) for (r, c), y in zip(keep, ok) if y]
        bad = [f(c) for (r, c), y in zip(keep, ok) if not y]
        print(f"  AUC {auc(good, bad):.3f}  {name}")

    by_part = defaultdict(list)
    for (r, c), y in zip(keep, ok):
        by_part[r["id"]].append((c, y))
    multi = {k: v for k, v in by_part.items() if len(v) >= 2 and any(y for _, y in v)}
    print(f"\nselection test on {len(multi)} parts with >= 2 answers and at least one correct:")
    print(f"  random pick correct: {mean(mean(y for _, y in v) for v in multi.values()):.3f}  (oracle 1.000)")
    for name, f in SIGNALS.items():
        picked = [max(v, key=lambda cy: f(cy[0]))[1] for v in multi.values()]
        print(f"  {name:40s} picks a correct answer {mean(picked):.3f}")

    out = [{"id": r["id"], "lane": r["lane"], "iou_aligned": r["iou_aligned"], **{k: c.get(k) for k in ("render_score", "silhouette", "edges", "size")}}
           for r, c in keep]
    (ROOT / "runs" / "verifier_validation.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
