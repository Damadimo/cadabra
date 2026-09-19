"""Merge benchmark runs into one table, split by part complexity (for README/Devpost).

  uv run python scripts/leaderboard.py runs/<run1> runs/<run2> ...     # or: --latest sota-img,ours-img
  uv run python scripts/leaderboard.py --latest sota-img,sota-img-rest-flash --out-json data/demo/scoreboard.json

Rows of the same lane, input modality and best-of setting are pooled across runs (a part graded twice counts once,
newest run wins), so a 200-part run plus a 300-part run of the other parts reads as one 500-part result. Rows where
the API returned no answer at all (402/5xx/disconnect) are left out: they say nothing about the model.
Tiers by the reference solid's face count: simple (<= 6 faces: boxes, cylinders), medium (7-12), complex (>= 13),
plus multi-part. Success = code runs and aligned IoU >= 0.9. CIs are bootstrap 95%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from understudy.cad.bench import infra_failed
from understudy.config import PRETTY, ROOT
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


def pctl(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))] if xs else None


LEFT_OUT: dict[str, int] = {}


def load(dirs: list[Path]) -> dict:
    """(modality, lane, best_of) -> {"lane": lane summary of the newest run, "runs": [...], "rows": {id: row}}."""
    groups: dict[tuple, dict] = {}
    for d in sorted(dirs, key=lambda d: json.loads((d / "summary.json").read_text())["created"]):
        meta = json.loads((d / "summary.json").read_text())
        modality, best_of = meta.get("modality", "text"), meta.get("best_of", 1)
        by_lane = {s["lane"]: s for s in meta["lanes"]}
        for r in read_jsonl(d / "results.jsonl"):
            if infra_failed(r):
                LEFT_OUT[r["lane"]] = LEFT_OUT.get(r["lane"], 0) + 1
                continue
            g = groups.setdefault((modality, r["lane"], best_of), {"lane": by_lane[r["lane"]], "runs": {}, "rows": {}, "shots": meta["shots"]})
            g["lane"] = by_lane[r["lane"]]
            g["runs"][d.name] = by_lane[r["lane"]]
            g["rows"][r["id"]] = {**r, "_run": d.name}
    return groups


def entry(key: tuple, g: dict) -> dict:
    modality, lane_key, best_of = key
    rows = list(g["rows"].values())
    lane = g["lane"]
    succ = [1.0 if r["success"] else 0.0 for r in rows]
    # $ per 1K parts: n-weighted over runs (dedicated lanes are priced per run from GPU time, API lanes per token)
    per_run = [(s["cost_per_1k_usd"], sum(r["_run"] == run for r in rows)) for run, s in g["runs"].items() if s.get("cost_per_1k_usd") is not None]
    cost = sum(c * n for c, n in per_run) / sum(n for _, n in per_run) if per_run and sum(n for _, n in per_run) else None
    tiers = {}
    for name, f in TIERS.items():
        s_ = [1.0 if r["success"] else 0.0 for r in rows if f(r)]
        tiers[name] = {"n": len(s_), "success": mean(s_) if s_ else None, "ci95": list(bootstrap_ci(s_)) if s_ else [None, None]}
    effort = lane.get("reasoning_effort")
    label = lane.get("label") or PRETTY.get(lane["model"], lane_key) + (f" ({effort})" if effort else "")
    return {
        "lane": lane_key, "label": label, "model": lane["model"], "reasoning_effort": effort,
        "modality": modality, "best_of": best_of, "shots": g["shots"], "runs": sorted(g["runs"]), "n": len(rows),
        "success": mean(succ), "success_ci95": list(bootstrap_ci(succ)),
        "success_iou95": mean(1.0 if r["runs"] and (r["iou_aligned"] or 0) >= 0.95 else 0.0 for r in rows),
        "run_rate": mean(1.0 if r["runs"] else 0.0 for r in rows),
        "mean_iou_aligned": mean((r["iou_aligned"] or 0.0) if r["runs"] else 0.0 for r in rows),
        "oracle_success": mean(1.0 if (r.get("oracle_iou") or 0) >= 0.9 else 0.0 for r in rows) if "oracle_iou" in rows[0] else None,
        "latency_p50": pctl([r["e2e_s"] for r in rows], 50), "latency_p95": pctl([r["e2e_s"] for r in rows], 95),
        "cost_per_1k_usd": cost, "tiers": tiers,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--latest", default="", help="comma-separated tags: use the newest run for each")
    ap.add_argument("--out-json", type=Path, help="also write the demo scoreboard (e.g. data/demo/scoreboard.json)")
    args = ap.parse_args()
    dirs = [d if d.is_absolute() else ROOT / d for d in args.runs]
    for tag in filter(None, args.latest.split(",")):
        found = sorted((ROOT / "runs").glob(f"*_{tag}"), key=lambda p: p.name)
        if found:
            dirs.append(found[-1])
    groups = load(dirs)
    board = sorted((entry(k, g) for k, g in groups.items()), key=lambda e: (e["modality"], -e["success"]))
    print("| Input | Lane | Best of | " + " | ".join(TIERS) + " | Latency p50 (s) | $ / 1K parts |")
    print("|" + "---|" * (len(TIERS) + 5))
    for e in board:
        rows = list(groups[(e["modality"], e["lane"], e["best_of"])]["rows"].values())
        cost = "–" if e["cost_per_1k_usd"] is None else f"${e['cost_per_1k_usd']:.2f}"
        print(f"| {e['modality']} | {e['label']} | {e['best_of']} | " + " | ".join(cell([r for r in rows if f(r)]) for f in TIERS.values())
              + f" | {e['latency_p50']:.1f} | {cost} |")
    print("\nRuns: " + ", ".join(sorted({run for e in board for run in e["runs"]})))
    if LEFT_OUT:
        print("Left out, API returned no answer: " + ", ".join(f"{k} {v}" for k, v in sorted(LEFT_OUT.items())))
    if args.out_json:
        created = max(json.loads((d / "summary.json").read_text())["created"] for d in dirs)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({"n": max(e["n"] for e in board), "created": created, "lanes": board}, indent=1))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
