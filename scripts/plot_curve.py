"""Learning curve for the README/pitch: correct parts vs training step, frontier models as reference lines.

  uv run python scripts/plot_curve.py                      # -> docs/learning_curve.png

Points: the untuned base (step 0, base-4b lane), every run tagged ckpt<N>-img (lane specialist), and the final model
(tag ours-img, drawn at FINAL_STEP). All scored on the parts of the 200-part frontier run, so every point and both
reference lines use the same parts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from understudy.cad.bench import infra_failed  # noqa: E402
from understudy.config import ROOT  # noqa: E402
from understudy.data import read_jsonl  # noqa: E402

BENCH = {r["id"]: r for r in read_jsonl(ROOT / "data" / "cad" / "bench.jsonl")}
HARD = lambda i: (BENCH[i].get("n_faces") or 0) >= 7  # noqa: E731


def latest(tag: str) -> Path | None:
    found = sorted((ROOT / "runs").glob(f"*_{tag}"), key=lambda p: p.name)
    return found[-1] if found else None


def scores(run: Path, lane: str, parts: set[str]) -> tuple[float, float] | None:
    rows = {r["id"]: r for r in read_jsonl(run / "results.jsonl") if r["lane"] == lane and not infra_failed(r) and r["id"] in parts}
    if not rows:
        return None
    hard = [r["success"] for i, r in rows.items() if HARD(i)]
    return 100 * mean(r["success"] for r in rows.values()), 100 * mean(hard) if hard else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frontier", default="sota-img")
    ap.add_argument("--final-tag", default="ours-img")
    ap.add_argument("--final-step", type=int, default=1892)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "learning_curve.png")
    args = ap.parse_args()
    frontier = latest(args.frontier)
    parts = {r["id"] for r in read_jsonl(frontier / "results.jsonl") if not infra_failed(r)}
    points = []  # (step, overall, hard)
    base_run = latest("ckpt200-img")
    if base_run and (s := scores(base_run, "base-4b", parts)):
        points.append((0, *s))
    for run in sorted((ROOT / "runs").glob("*_ckpt*-img")):
        m = re.search(r"_ckpt(\d+)-img$", run.name)
        if m and (s := scores(run, "specialist", parts)):
            points.append((int(m.group(1)), *s))
    final = latest(args.final_tag)
    if final and (s := scores(final, "specialist", parts)):
        points.append((args.final_step, *s))
    points.sort()
    refs = {}
    for lane, label in (("glm-5.3-flash@high", "GLM-5.3 Flash (high)"), ("kimi-k3@high", "Kimi K3 (high)")):
        if s := scores(frontier, lane, parts):
            refs[label] = s

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    xs = [p[0] for p in points]
    ax.plot(xs, [p[1] for p in points], "-o", color="#0f9d58", lw=2, label="Ours: all parts")
    ax.plot(xs, [p[2] for p in points], "--o", color="#0f9d58", lw=1.4, alpha=0.7, label="Ours: medium + complex (≥ 7 faces)")
    for (label, (overall, hard)), color in zip(refs.items(), ("#5f6b7a", "#9aa5b1")):
        ax.axhline(overall, color=color, lw=1.2, ls="-", alpha=0.9, label=f"{label}: all {overall:.0f}%")
        ax.axhline(hard, color=color, lw=1.0, ls=":", alpha=0.9, label=f"{label}: ≥ 7 faces {hard:.0f}%")
    for step, overall, _ in points:
        ax.annotate(f"{overall:.0f}", (step, overall), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color="#0f9d58")
    ax.set_xlabel("Training step (batch 16; one epoch = 1,892 steps)")
    ax.set_ylabel("Correct parts (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Learning curve on {len(parts)} held-out drawing sheets (greedy, one sample)", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncols=2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(json.dumps({"points": points, "frontier": refs}, indent=0))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
