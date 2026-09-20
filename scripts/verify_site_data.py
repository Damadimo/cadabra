"""Check that what the published site shows is what the benchmark actually produced.

  uv run python scripts/verify_site_data.py [--sample 40] [--all]

Three independent checks, because "the page renders" says nothing about whether the right numbers are on it:
  1. every lane in every exported compare document against its own row in runs/<run>/results.jsonl (IoU, verdict, code)
  2. a sample of those answers re-executed from their code, in the grader's own isolated processes, so the IoU on
     screen is one this machine can reproduce rather than one copied forward
  3. the served STL for each answer against the solid that code actually builds, and the reference STL against the
     gold program, so a render can never belong to a different model or a different part
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import struct

from cadabra.cad.pool import CadPool
from cadabra.config import ROOT
from cadabra.data import read_jsonl

COMPARE = ROOT / "site" / "public" / "data" / "compare"
MESH = ROOT / "site" / "public" / "data" / "mesh"


def signature(stl: bytes) -> tuple:
    """Triangle count and bounding box: enough to tell two solids apart, indifferent to float noise."""
    n = struct.unpack("<I", stl[80:84])[0]
    lo, hi = [1e9] * 3, [-1e9] * 3
    for i in range(n):
        base = 84 + i * 50 + 12
        for v in range(3):
            for j, c in enumerate(struct.unpack("<3f", stl[base + v * 12: base + v * 12 + 12])):
                lo[j] = min(lo[j], c)
                hi[j] = max(hi[j], c)
    return n, tuple(round(v, 3) for v in lo), tuple(round(v, 3) for v in hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=40, help="parts to re-execute (checks 2 and 3)")
    ap.add_argument("--all", action="store_true", help="re-execute every part, not a sample")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    bench = {r["id"]: r for r in read_jsonl(ROOT / "data" / "cad" / "bench.jsonl")}
    docs = sorted(COMPARE.glob("*.json"))
    print(f"{len(docs)} exported parts\n")

    # ---- 1. every lane against the run it came from ----
    runs: dict[str, dict] = {}

    def rows(run: str) -> dict:
        if run not in runs:
            runs[run] = {(r["id"], r["lane"]): r for r in read_jsonl(ROOT / "runs" / run / "results.jsonl")}
        return runs[run]

    checked = bad = 0
    for path in docs:
        doc = json.loads(path.read_text())
        for lane in doc["lanes"]:
            src = rows(lane["run"]).get((doc["id"], lane["lane"]))
            checked += 1
            if src is None:
                bad += 1
                print(f"  MISSING {doc['id']} {lane['key']}: no row in {lane['run']}")
                continue
            a, b = lane.get("iou_aligned"), src.get("iou_aligned")
            same = ((a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-9)) \
                and bool(lane["success"]) == bool(src["success"]) \
                and (lane.get("code") or "").strip() == (src.get("code") or "").strip()
            if not same:
                bad += 1
                print(f"  MISMATCH {doc['id']} {lane['key']}: shown {a}/{lane['success']} vs run {b}/{src['success']}")
    print(f"1. {checked} lane rows against their benchmark run: {bad} mismatches")

    # ---- 2 and 3. rebuild the solids and compare ----
    random.seed(args.seed)
    chosen = docs if args.all else random.sample(docs, min(args.sample, len(docs)))
    jobs, meta = [], []
    for path in chosen:
        doc = json.loads(path.read_text())
        gold = bench[doc["id"]]["gold_code"]
        if doc["reference"].get("mesh"):
            jobs.append({"code": gold, "gold_code": None, "want_mesh": True, "want_chamfer": False})
            meta.append((doc["id"], "reference", doc["reference"]["mesh"], None))
        for lane in doc["lanes"]:
            if lane.get("code") and lane.get("mesh"):
                jobs.append({"code": lane["code"], "gold_code": gold, "want_mesh": True, "want_chamfer": False})
                meta.append((doc["id"], lane["key"], lane["mesh"], lane.get("iou_aligned")))

    with CadPool(workers=6, timeout=90) as pool:
        results = pool.map(jobs)

    iou_bad = mesh_bad = 0
    for (pid, key, mesh_key, shown), res in zip(meta, results):
        served = (MESH / f"{mesh_key}.stl").read_bytes()
        rebuilt = res.get("aligned_mesh_stl") or res.get("mesh_stl")
        if rebuilt is None or signature(rebuilt) != signature(served):
            mesh_bad += 1
            print(f"  MESH {pid} {key}: the STL served is not the solid that code builds")
        if shown is not None:
            got = res.get("iou_aligned")
            # OpenCascade's boolean volumes are not bit-reproducible; differences this small never move a
            # verdict, which turns at 0.9.
            if got is None or abs(got - shown) > 5e-3:
                iou_bad += 1
                print(f"  IOU {pid} {key}: recomputed {got} vs shown {shown}")
    print(f"2. {sum(1 for m in meta if m[3] is not None)} answers re-executed from their code: {iou_bad} IoU mismatches")
    print(f"3. {len(meta)} solids compared with the STL served: {mesh_bad} mismatches")
    raise SystemExit(1 if (bad or iou_bad or mesh_bad) else 0)


if __name__ == "__main__":  # the grader's workers re-import this file when they spawn
    main()
