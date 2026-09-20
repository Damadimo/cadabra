"""What GRPO did to the SFT model, checkpoint by checkpoint.

  uv run python scripts/plot_rl.py --job wx22m73        # -> docs/rl_curve.png

Reads the [sweep] lines the eval-only job printed (scripts/../training/vlm/eval_sweep.py), which grade every saved
GRPO checkpoint and the SFT checkpoint they started from on the same 500 held-out sheets with the same harness. The
SFT point is the control: it has to land on the number that run reported for itself, or the sweep is measuring
something else.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cadabra.config import ROOT  # noqa: E402

TIERS = [("all", "All 500", "#0f9d58", 2.2, "-"),
         ("medium (7-12)", "Medium (7–12 faces)", "#c2410c", 1.6, "-"),
         ("complex (>=13)", "Complex (≥ 13)", "#5f6b7a", 1.4, "--"),
         ("simple (<=6 faces)", "Simple (≤ 6)", "#9aa5b1", 1.2, ":")]


SAVED = ROOT / "runs" / "rl_checkpoint_sweep.json"


def sweep(job: str) -> dict[str, dict]:
    if SAVED.exists():  # the summaries as they were printed, so the figure does not depend on the job still existing
        saved = json.loads(SAVED.read_text())
        if saved.get("job") == job:
            return saved["summaries"]
    out = subprocess.run(["baseten", "train", "job", "logs", "--job-id", job], capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in out.replace("\r", "").splitlines():
        m = re.search(r"\[sweep\] (\S+) (\{.*\})", line)
        if m:
            rows[m.group(1)] = json.loads(m.group(2))
    if not rows:
        raise SystemExit(f"no [sweep] lines in job {job}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", default="wx22m73", help="the eval-only sweep job")
    ap.add_argument("--sft-checkpoint", default="946", help="which checkpoint in the sweep is the SFT control")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "rl_curve.png")
    args = ap.parse_args()
    rows = sweep(args.job)
    sft = next(v for k, v in rows.items() if args.sft_checkpoint in k)
    steps = sorted((int(k.rsplit("-", 1)[1]), v) for k, v in rows.items() if args.sft_checkpoint not in k)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    xs = [0] + [s for s, _ in steps]
    for key, label, color, lw, ls in TIERS:
        ys = [100 * (sft[key]["success"] or 0)] + [100 * (v[key]["success"] or 0) for _, v in steps]
        ax.plot(xs, ys, ls, marker="o", ms=3.5, color=color, lw=lw, label=f"{label}: {ys[0]:.1f}% → {ys[-1]:.1f}%")
    ax.axvline(0, color="#111", lw=0.8, alpha=0.5)
    ax.annotate("SFT model\n(step 0)", (0, 100 * sft["all"]["success"]), textcoords="offset points", xytext=(8, 10), fontsize=7.5)
    ax.set_xlabel("GRPO steps on top of the SFT model (64 rollouts per step)")
    ax.set_ylabel("Correct parts (%)")
    ax.set_ylim(0, 100)
    ax.set_title("GRPO on the geometry reward, scored on the 500 held-out sheets (greedy, one sample)", loc="left", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncols=2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(json.dumps({"sft": sft["all"], "steps": [(s, v["all"]) for s, v in steps]}, indent=0))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
