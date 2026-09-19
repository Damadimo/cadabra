"""LOOK AT YOUR DATA: run every reference program in CAD-Coder and check it against its own spec.

  uv run python scripts/audit_cadcoder.py --splits test train_high --workers 12

For each row: does the reference code run, how many parts does the spec describe, are its global transforms the
identity, and do the reference solid's bbox extents match the dimensions the spec states? Writes
data/cache/audit_<split>.jsonl (used to build clean benchmark and training splits) and prints a summary.
"""

from __future__ import annotations

import argparse
import collections
import json
import time

from understudy.cad.data import load_split, normalized_code
from understudy.cad.pool import CadPool
from understudy.config import ROOT
from understudy.data import write_jsonl

CACHE = ROOT / "data" / "cache"


def dims_consistent(extents: list[float], stated: list[tuple], tol: float = 0.02) -> bool | None:
    """Single-part check: sorted bbox extents match the sorted stated (length, width, height)."""
    if len(stated) != 1:
        return None
    want = sorted(stated[0])
    return all(abs(a - b) <= tol * max(abs(b), 1e-6) + 1e-4 for a, b in zip(sorted(extents), want))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "train_high"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    records = {s: load_split(s, args.limit) for s in args.splits}
    with CadPool(workers=args.workers, timeout=30) as pool:
        print(f"{pool.workers} workers ready")
        for split, recs in records.items():
            t0 = time.time()
            results = pool.map([{"code": r["gold_code"], "want_chamfer": False} for r in recs])
            rows, errors = [], collections.Counter()
            for rec, res in zip(recs, results):
                row = {k: rec[k] for k in ("id", "source_id", "n_parts", "stated_dims", "identity_transforms")}
                row["runs"] = res["runs"]
                row["error"] = res.get("error")
                if res["runs"]:
                    row["extents"] = res["stats"]["extents"]
                    row["n_solids"] = res["stats"]["n_solids"]
                    row["n_faces"] = res["stats"]["n_faces"]
                    row["dims_ok"] = dims_consistent(res["stats"]["extents"], rec["stated_dims"])
                else:
                    errors[(res.get("error") or "")[:60]] += 1
                rows.append(row)
            write_jsonl(CACHE / f"audit_{split}.jsonl", rows)
            n = len(rows)
            ran = [r for r in rows if r["runs"]]
            single = [r for r in ran if r["n_parts"] == 1]
            checked = [r for r in single if r["dims_ok"] is not None]
            ok = [r for r in checked if r["dims_ok"]]
            ident = [r for r in single if r["identity_transforms"]]
            print(f"\n== {split}: {n} usable rows, audited in {time.time() - t0:.0f}s")
            print(f"  reference runs:        {len(ran)}/{n} ({len(ran) / n:.1%})")
            print(f"  single-part:           {len(single)}/{len(ran)} ({len(single) / max(len(ran), 1):.1%})")
            print(f"  single, identity xform:{len(ident)}/{len(single)}")
            print(f"  single, dims match:    {len(ok)}/{len(checked)} checked ({len(ok) / max(len(checked), 1):.1%})")
            parts = collections.Counter(min(r["n_parts"], 5) for r in ran)
            print(f"  parts histogram (5=5+): {dict(sorted(parts.items()))}")
            faces = sorted(r["n_faces"] for r in ran)
            print(f"  faces p50/p90: {faces[len(faces) // 2]}/{faces[int(len(faces) * 0.9)]}")
            print("  top errors:", errors.most_common(5))

    if {"test", "train_high"} <= records.keys():
        test, train = records["test"], records["train_high"]
        src = {r["source_id"] for r in train} & {r["source_id"] for r in test}
        code = {normalized_code(r["gold_code"]) for r in train} & {normalized_code(r["gold_code"]) for r in test}
        spec = {r["spec"] for r in train} & {r["spec"] for r in test}
        print(f"\n== overlap test vs train_high: source_id {len(src)}, identical code {len(code)}, identical spec {len(spec)}")


if __name__ == "__main__":
    main()
