"""Build the leakage-safe splits used everywhere (deterministic; run once after scripts/audit_cadcoder.py).

  uv run python -m cadabra.cad.splits

- bench.jsonl        500 specs held out of CAD-Coder train_high (the curated split), grouped by source part,
                     references verified to run (and to match their stated size where the spec states one).
- train.jsonl        the rest of train_high after the same checks. Never overlaps bench by source part.
- test_verified.jsonl official CAD-Coder test rows whose single-part reference runs and matches its stated size.
- shots.jsonl        2 worked examples from train, shown to frontier baselines as prior turns.
- train_vlm_extra.jsonl  extra parts from train_middle for the drawing-sheet model (images are rendered from the
                     reference code, so image and code always agree even where the text spec is noisy).
- train_vlm_extra2.jsonl every remaining medium/complex/multi-part train_middle part, one per geometry signature not
                     already covered (`python -m cadabra.cad.splits --extra2`; leaves every other file untouched).

Leakage guards: no source part is shared with bench, and no training part has the same geometry signature
(bbox extents + face count) as a bench part: the underlying CAD library repeats many identical boxes.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from ..config import ROOT
from ..data import read_jsonl, write_jsonl
from .data import load_split

OUT = ROOT / "data" / "cad"
CACHE = ROOT / "data" / "cache"
KEEP = ("id", "source_id", "spec", "gold_code", "n_parts", "identity_transforms")
SHOT_IDS = ("train_high:03751", "train_high:04183")


def signature(rec: dict) -> tuple:
    return tuple(round(x, 3) for x in rec["extents"]) + (rec["n_faces"],)


def audited(split: str) -> list[dict]:
    audit = {row["id"]: row for row in read_jsonl(CACHE / f"audit_{split}.jsonl")}
    if not audit:
        raise SystemExit(f"run scripts/audit_cadcoder.py --splits {split} first")
    out = []
    for rec in load_split(split):
        a = audit.get(rec["id"])
        if not a or not a["runs"] or a.get("dims_ok") is False:
            continue
        out.append({**{k: rec[k] for k in KEEP}, "n_faces": a["n_faces"], "extents": a["extents"], "dims_ok": a.get("dims_ok")})
    return out


def main(bench_size: int = 500, seed: int = 7, extra_size: int = 12000) -> None:
    rng = random.Random(seed)
    pool = audited("train_high")
    by_source = defaultdict(list)
    for rec in pool:
        by_source[rec["source_id"]].append(rec)
    sources = sorted(by_source)
    rng.shuffle(sources)
    bench, taken = [], set()
    for src in sources:
        if len(bench) >= bench_size:
            break
        bench.extend(by_source[src])
        taken.add(src)
    bench_sigs = {signature(r) for r in bench}
    train = [r for r in pool if r["source_id"] not in taken]
    same_geometry = sum(signature(r) in bench_sigs for r in train)
    train = [r for r in train if signature(r) not in bench_sigs]

    test = [r for r in audited("test") if r["n_parts"] == 1 and r["dims_ok"] is True]
    rng.shuffle(test)

    # Pinned worked examples (chosen once: a single-part profile with arcs, and a two-part cut). Every frontier
    # baseline in runs/ used exactly these two, so they must not change when the training pool does.
    by_id = {r["id"]: r for r in pool}
    shots = [by_id[i] for i in SHOT_IDS]
    train = [r for r in train if r["id"] not in SHOT_IDS]

    # Extra drawing-sheet training parts from train_middle: complex-weighted, deduped against bench and train.
    known = {r["source_id"] for r in bench} | {r["source_id"] for r in train}
    middle = [r for r in audited("train_middle") if r["source_id"] not in known and signature(r) not in bench_sigs]
    seen, uniq = set(), []
    for r in middle:  # one row per source part (train_middle repeats parts with different descriptions)
        if r["source_id"] not in seen:
            seen.add(r["source_id"])
            uniq.append(r)
    hard = [r for r in uniq if r["n_faces"] >= 7 or r["n_parts"] > 1]
    easy = [r for r in uniq if not (r["n_faces"] >= 7 or r["n_parts"] > 1)]
    k_hard = min(len(hard), int(extra_size * 0.7))
    extra = rng.sample(hard, k_hard) + rng.sample(easy, min(len(easy), extra_size - k_hard))

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "train_vlm_extra.jsonl", extra)
    write_jsonl(OUT / "bench.jsonl", bench)
    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "test_verified.jsonl", test)
    write_jsonl(OUT / "shots.jsonl", shots)
    stats = {
        "train": len(train),
        "bench": len(bench),
        "bench_multi_part": sum(r["n_parts"] > 1 for r in bench),
        "test_verified": len(test),
        "shots": [s["id"] for s in shots],
        "source_overlap_train_bench": len({r["source_id"] for r in train} & {r["source_id"] for r in bench}),
        "dropped_same_geometry_as_bench": same_geometry,
        "train_vlm_extra": len(extra),
        "train_vlm_extra_complex_share": round(k_hard / max(len(extra), 1), 2),
    }
    (OUT / "splits.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def extra2() -> None:
    """Append-only second batch for the sheet model: all unused hard train_middle parts, one per new geometry."""
    bench = read_jsonl(OUT / "bench.jsonl")
    used = bench + read_jsonl(OUT / "train.jsonl") + read_jsonl(OUT / "train_vlm_extra.jsonl")
    audit = {r["id"]: r for split in ("train_high", "train_middle") for r in read_jsonl(CACHE / f"audit_{split}.jsonl")}
    bench_sigs = {signature({**r, **audit[r["id"]]}) for r in bench}
    known_sources = {r["source_id"] for r in used}
    covered = {signature({**r, **audit[r["id"]]}) for r in used if r["id"] in audit}
    out = []
    for r in audited("train_middle"):
        sig = signature(r)
        if r["source_id"] in known_sources or sig in bench_sigs or sig in covered:
            continue
        if r["n_faces"] >= 7 or r["n_parts"] > 1:
            known_sources.add(r["source_id"])
            covered.add(sig)
            out.append(r)
    write_jsonl(OUT / "train_vlm_extra2.jsonl", out)
    stats = json.loads((OUT / "splits.json").read_text())
    stats["train_vlm_extra2"] = len(out)
    stats["train_vlm_extra2_complex_13plus"] = sum(r["n_faces"] >= 13 for r in out)
    (OUT / "splits.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k.startswith("train_vlm")}, indent=2))


if __name__ == "__main__":
    import sys

    extra2() if "--extra2" in sys.argv else main()
