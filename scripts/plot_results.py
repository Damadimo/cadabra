"""Charts for the README, Devpost and pitch from the pooled scoreboard (scripts/leaderboard.py --out-json).

  uv run python scripts/plot_results.py                       # data/demo/scoreboard.json -> docs/*.png
  uv run python scripts/plot_results.py --board path.json --modality image

docs/results_by_tier.png   correct (IoU >= 0.9) per complexity tier, one bar per lane, 95% CI whiskers
docs/cost_vs_accuracy.png  $ per 1K parts (log) vs overall correct, one point per lane
Lanes that are not real vision baselines (GLM-5.3: text-only model card) are left out unless --all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cadabra.config import ROOT  # noqa: E402

TIERS = [("all", "All parts"), ("simple (<=6 faces)", "Simple\n(≤ 6 faces)"), ("medium (7-12)", "Medium\n(7–12 faces)"),
         ("complex (>=13)", "Complex\n(≥ 13 faces)"), ("multi-part", "Multi-part")]
TEXT_ONLY = {"zai-org/GLM-5.3"}
OURS, FRONTIER = "#0f9d58", ["#5f6b7a", "#9aa5b1", "#c3cad3", "#38414d"]


def name(lane: dict) -> str:
    extra = f" · best of {lane['best_of']}" if lane.get("best_of", 1) > 1 else ""
    return lane.get("label", lane["lane"]) + extra


def is_ours(lane: dict) -> bool:
    return lane["lane"] == "specialist" or "(ours)" in lane.get("label", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", type=Path, default=ROOT / "data" / "demo" / "scoreboard.json")
    ap.add_argument("--modality", default="image")
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    ap.add_argument("--all", action="store_true", help="also plot text-only models and partial (tier-only) runs")
    args = ap.parse_args()
    board = json.loads(args.board.read_text())
    lanes = [l for l in board["lanes"] if l.get("modality", "text") == args.modality]
    if not args.all:
        lanes = [l for l in lanes if l["model"] not in TEXT_ONLY and l["n"] >= 100]
    lanes.sort(key=lambda l: (not is_ours(l), -(l["success"] or 0)))
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    width = 0.8 / max(len(lanes), 1)
    frontier = iter(FRONTIER * 3)
    for i, lane in enumerate(lanes):
        color = OURS if is_ours(lane) else next(frontier)
        xs, ys, lo, hi = [], [], [], []
        for t, (key, _) in enumerate(TIERS):
            tier = lane["tiers"].get(key) or {}
            if tier.get("success") is None:
                continue
            xs.append(t - 0.4 + width * (i + 0.5))
            ys.append(100 * tier["success"])
            lo.append(100 * (tier["success"] - tier["ci95"][0]))
            hi.append(100 * (tier["ci95"][1] - tier["success"]))
        bars = ax.bar(xs, ys, width * 0.92, color=color, label=name(lane), yerr=[lo, hi], capsize=2,
                      error_kw={"elinewidth": 0.8, "ecolor": "#333"})
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, 1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=7,
                    color="white" if y > 8 else "#333", fontweight="bold" if is_ours(lane) else None)
    n = max((l["n"] for l in lanes), default=0)
    ax.set_xticks(range(len(TIERS)), [label for _, label in TIERS])
    ax.set_ylabel("Correct parts (%)  ·  code runs and IoU ≥ 0.9")
    ax.set_ylim(0, 100)
    ax.set_title(f"Drawing sheet → CAD code on {n} held-out parts (95% CI)", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out / "results_by_tier.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    for lane in lanes:
        if lane.get("cost_per_1k_usd") is None:
            continue
        ours = is_ours(lane)
        ax.scatter(lane["cost_per_1k_usd"], 100 * lane["success"], s=90 if ours else 60, color=OURS if ours else "#5f6b7a", zorder=3)
        ax.annotate(name(lane), (lane["cost_per_1k_usd"], 100 * lane["success"]), textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 500)
    ax.set_ylim(0, 100)
    ax.set_xlabel("$ per 1,000 parts (log scale; API list price, or GPU-hours for ours)")
    ax.set_ylabel("Correct parts (%)")
    ax.set_title("Accuracy vs cost", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(args.out / "cost_vs_accuracy.png")
    plt.close(fig)
    print(f"wrote {args.out / 'results_by_tier.png'} and {args.out / 'cost_vs_accuracy.png'} ({len(lanes)} lanes)")


if __name__ == "__main__":
    main()
