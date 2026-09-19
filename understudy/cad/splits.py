"""Build the leakage-safe splits used everywhere (deterministic; run once after scripts/audit_cadcoder.py).

  uv run python -m understudy.cad.splits

- bench.jsonl        500 specs held out of CAD-Coder train_high (the curated split), grouped by source part,
                     references verified to run (and to match their stated size where the spec states one).
- train.jsonl        the rest of train_high after the same checks. Never overlaps bench by source part.
- test_verified.jsonl official CAD-Coder test rows whose single-part reference runs and matches its stated size.
- shots.jsonl        2 worked examples from train, shown to frontier baselines as prior turns.
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


def main(bench_size: int = 500, seed: int = 7) -> None:
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
    train = [r for r in pool if r["source_id"] not in taken]

    test = [r for r in audited("test") if r["n_parts"] == 1 and r["dims_ok"] is True]
    rng.shuffle(test)

    def pick(cond):
        cands = sorted((r for r in train if cond(r)), key=lambda r: r["id"])
        return cands[len(cands) // 2]

    shots = [
        pick(lambda r: r["n_parts"] == 1 and "arc" in r["spec"].lower() and 8 <= r["n_faces"] <= 12),
        pick(lambda r: r["n_parts"] == 2 and ("remove" in r["spec"].lower() or "cut" in r["spec"].lower())),
    ]
    shot_ids = {s["id"] for s in shots}
    train = [r for r in train if r["id"] not in shot_ids]

    OUT.mkdir(parents=True, exist_ok=True)
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
    }
    (OUT / "splits.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
