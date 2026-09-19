"""Merge benchmark runs into one table, split by part complexity (for README/Devpost).

  uv run python scripts/leaderboard.py runs/<run1> runs/<run2> ...     # or: --latest pilot,ours
  uv run python scripts/leaderboard.py --modality image runs/...

Tiers by the reference solid's face count: simple (<= 6 faces: boxes, cylinders), medium (7-12), complex (>= 13),
plus multi-part. Success = code runs and aligned IoU >= 0.9. CIs are bootstrap 95%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from understudy.config import ROOT
from understudy.data import read_jsonl
from understudy.stats import bootstrap_ci

TIERS = {
    "all": lambda r: True,
    "simple (<=6 faces)": lambda r: (r.get("n_faces") or 0) <= 6,
    "medium (7-12)": lambda r: 7 <= (r.get("n_faces") or 0) <= 12,
    "complex (>=13)": lambda r: (r.get("n_faces") or 0) >= 13,
    "multi-part": lambda r: r["n_parts"] > 1,
}


def cell(rows: list[dict]) -> str:
    if not rows:
        return "–"
    s = [1.0 if r["success"] else 0.0 for r in rows]
    lo, hi = bootstrap_ci(s)
    return f"{100 * mean(s):.1f}% [{100 * lo:.0f}–{100 * hi:.0f}] (n={len(rows)})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--latest", default="", help="comma-separated tags: use the newest run for each")
    args = ap.parse_args()
    dirs = list(args.runs)
    for tag in filter(None, args.latest.split(",")):
        found = sorted((ROOT / "runs").glob(f"*_{tag}"), key=lambda p: p.stat().st_mtime)
        if found:
            dirs.append(found[-1])
    rows, metas = [], {}
    for d in dirs:
        d = d if d.is_absolute() else ROOT / d
        meta = json.loads((d / "summary.json").read_text())
        for r in read_jsonl(d / "results.jsonl"):
            r["_run"] = d.name
            r["_modality"] = meta.get("modality", "text")
            rows.append(r)
        metas[d.name] = meta
    lanes = sorted({(r["_modality"], r["lane"], r["model"]) for r in rows})
    head = "| Input | Lane | " + " | ".join(TIERS) + " | Latency p50 (s) | $ / 1K parts |"
    print(head)
    print("|" + "---|" * (len(TIERS) + 4))
    for modality, lane, model in lanes:
        mine = [r for r in rows if r["lane"] == lane and r["_modality"] == modality]
        lat = sorted(r["e2e_s"] for r in mine)
        costs = [r["cost_usd"] for r in mine if r.get("cost_usd") is not None]
        cost = f"${1000 * mean(costs):.2f}" if costs else "–"
        print(f"| {modality} | {lane} | " + " | ".join(cell([r for r in mine if f(r)]) for f in TIERS.values())
              + f" | {lat[len(lat) // 2]:.1f} | {cost} |")
    print("\nRuns: " + ", ".join(f"{k} ({v['n']} parts, shots {v['shots']})" for k, v in metas.items()))


if __name__ == "__main__":
    main()
