"""How close is the training set to the held-out benchmark, and does closeness inflate our score?

  uv run python scripts/leakage_check.py --ours ours-img --frontier sota-img,sota-img-rest

By construction no benchmark source part is in training and no training part has a benchmark part's exact geometry
signature (rounded extents + face count). This checks the softer case: parts that are *nearly* the same size and shape.
For every benchmark part it finds the closest training part with the same face count and part count, by relative
difference of the (sorted, so rotation-independent) bounding-box extents, then reports our accuracy and the frontier's
on the near-duplicate parts vs the rest. If only our score rises on near-duplicates, the model is leaning on
memorization; if both rise, those parts are simply easier.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from understudy.cad.bench import infra_failed
from understudy.config import ROOT
from understudy.data import read_jsonl

CACHE = ROOT / "data" / "cad"


def audit_index() -> dict:
    out = {}
    for split in ("train_high", "train_middle"):
        path = ROOT / "data" / "cache" / f"audit_{split}.jsonl"
        for r in read_jsonl(path):
            out[r["id"]] = r
    return out


def latest(tag: str) -> Path | None:
    found = sorted((ROOT / "runs").glob(f"*_{tag}"), key=lambda p: p.name)
    return found[-1] if found else None


def closest(bench: list[dict], train: list[dict]) -> dict[str, float]:
    """id -> smallest relative extent difference to a training part with the same face count and part count."""
    buckets: dict[tuple, list] = {}
    for r in train:
        buckets.setdefault((r["n_faces"], r["n_parts"]), []).append(r["extents"])
    out = {}
    for b in bench:
        best = 1.0
        for e in buckets.get((b["n_faces"], b["n_parts"]), ()):
            d = max(abs(x - y) / max(x, y, 1e-9) for x, y in zip(b["extents"], e))
            best = min(best, d)
            if best == 0.0:
                break
        out[b["id"]] = best
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="ours-img")
    ap.add_argument("--frontier", default="sota-img,sota-img-rest")
    ap.add_argument("--threshold", type=float, default=0.01, help="relative extent difference counted as a near-duplicate")
    args = ap.parse_args()
    audit = audit_index()
    def rows(name):
        return [{**r, "extents": audit[r["id"]]["extents"], "n_faces": audit[r["id"]]["n_faces"]}
                for r in read_jsonl(CACHE / name) if r["id"] in audit]

    bench = rows("bench.jsonl")
    train = [r for name in ("train.jsonl", "train_vlm_extra.jsonl", "train_vlm_extra2.jsonl") for r in rows(name)]
    print(f"benchmark parts {len(bench)} · training parts {len(train)}")
    print(f"shared source parts: {len({r['source_id'] for r in bench} & {r['source_id'] for r in train})}")
    sig = lambda r: tuple(round(x, 3) for x in r["extents"]) + (r["n_faces"],)  # noqa: E731
    print(f"shared exact geometry signatures: {len({sig(r) for r in bench} & {sig(r) for r in train})}")

    dist = closest(bench, train)
    near = {i for i, d in dist.items() if d <= args.threshold}
    print(f"benchmark parts with a training part within {100 * args.threshold:.0f}% on every axis "
          f"(same face and part count): {len(near)} of {len(bench)}")
    for t in (0.0, 0.001, 0.01, 0.05):
        print(f"  within {100 * t:5.1f}%: {sum(1 for d in dist.values() if d <= t)}")

    # one row per model, pooling the frontier's runs (a part graded twice counts once)
    pools: dict[str, dict[str, bool]] = {}
    for tag in [args.ours] + args.frontier.split(","):
        run = latest(tag)
        if not run:
            continue
        for r in read_jsonl(run / "results.jsonl"):
            if infra_failed(r):
                continue
            name = "ours" if r["lane"] == "specialist" else r["lane"]
            pools.setdefault(name, {})[r["id"]] = bool(r["success"])
    print(f"\naccuracy on the {len(near)} near-duplicate parts vs the other {len(bench) - len(near)}:")
    for name, got in pools.items():
        a = [v for i, v in got.items() if i in near]
        b = [v for i, v in got.items() if i not in near]
        if not a or not b:
            continue
        print(f"  {name:22s} near-dup {100 * mean(a):5.1f}% (n={len(a):3d})   no near-dup {100 * mean(b):5.1f}% (n={len(b):3d})   gap {100 * (mean(a) - mean(b)):+5.1f}")


if __name__ == "__main__":
    main()
