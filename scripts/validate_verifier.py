"""Does render-and-compare (which only sees the input drawing) agree with the answer-key IoU?

  uv run python scripts/validate_verifier.py runs/<image-run> [runs/<image-run> ...]

For every answer in the runs that built a solid, compute its render score against the input sheet and compare with
its aligned IoU against the reference. Also scores each reference program itself (should be ~1.0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from understudy.cad.pool import CadPool
from understudy.cad.render import OUT as IMAGES
from understudy.cad.render import image_path
from understudy.config import ROOT
from understudy.data import read_jsonl


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()
    bench = {r["id"]: r for r in read_jsonl(ROOT / "data" / "cad" / "bench.jsonl")}
    rows = [r for d in args.runs for r in read_jsonl((d if d.is_absolute() else ROOT / d) / "results.jsonl") if r["runs"] and r.get("code")]
    ids = sorted({r["id"] for r in rows})
    tasks = [{"_fn": "render_score", "code": r["code"], "sheet_path": str(image_path(r["id"], IMAGES))} for r in rows]
    tasks += [{"_fn": "render_score", "code": bench[i]["gold_code"], "sheet_path": str(image_path(i, IMAGES))} for i in ids]
    with CadPool(timeout=60) as pool:
        res = pool.map(tasks)
    cand, gold = res[: len(rows)], res[len(rows):]
    good = [c["render_score"] for r, c in zip(rows, cand) if (r["iou_aligned"] or 0) >= args.threshold]
    bad = [c["render_score"] for r, c in zip(rows, cand) if (r["iou_aligned"] or 0) < args.threshold]
    gold_scores = sorted(g["render_score"] for g in gold)
    print(f"answers scored: {len(rows)} ({len(good)} correct, {len(bad)} wrong by the answer key)")
    print(f"reference programs: render score min {gold_scores[0]:.3f}, p10 {gold_scores[len(gold_scores) // 10]:.3f}, median {gold_scores[len(gold_scores) // 2]:.3f}")
    print(f"render score, correct answers: median {sorted(good)[len(good) // 2]:.3f}" if good else "no correct answers")
    print(f"render score, wrong answers:   median {sorted(bad)[len(bad) // 2]:.3f}" if bad else "no wrong answers")
    print(f"AUC (render score separates correct from wrong): {auc(good, bad):.3f}")
    for t in (0.9, 0.95, 0.97, 0.99):
        tp = sum(s >= t for s in good)
        fp = sum(s >= t for s in bad)
        print(f"  accept if render score >= {t}: keeps {tp}/{len(good)} correct, {fp}/{len(bad)} wrong")
    out = [{"id": r["id"], "lane": r["lane"], "iou_aligned": r["iou_aligned"], "render_score": c["render_score"]} for r, c in zip(rows, cand)]
    (ROOT / "runs" / "verifier_validation.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
