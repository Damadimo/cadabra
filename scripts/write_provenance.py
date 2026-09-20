"""Write data/cad/provenance.json: the dataset counts the site quotes, taken from the files that produced them.

  uv run python scripts/write_provenance.py

The How it works page renders this rather than repeating numbers in prose, so a re-audit or a re-split cannot leave
the page quoting counts nothing else agrees with.
"""

from __future__ import annotations

import json

from cadabra.config import ROOT

CACHE = ROOT / "data" / "cache"


def main() -> None:
    audited = {}
    for split in ("train_high", "train_middle", "test"):
        path = CACHE / f"audit_{split}.jsonl"
        audited[split] = sum(1 for line in path.open() if line.strip())
    splits = json.loads((ROOT / "data" / "cad" / "splits.json").read_text())
    stats = json.loads((ROOT / "training" / "vlm" / "data" / "stats.json").read_text())

    out = {
        "audited_total": sum(audited.values()),
        "audited_by_split": audited,
        "bench": splits["bench"],
        "bench_multi_part": splits["bench_multi_part"],
        "train_high": splits["train"],
        "train_middle": splits["train_vlm_extra"] + splits["train_vlm_extra2"],
        "sheets_trained_on": stats["train"],
        "sheets_validation": stats["val"],
        "share_medium_or_complex": round(stats["complex_7plus_faces"] / stats["train"], 4),
        "share_multi_part": round(stats["multi_part"] / stats["train"], 4),
        "source_overlap_train_bench": splits["source_overlap_train_bench"],
        "dropped_same_geometry_as_bench": splits["dropped_same_geometry_as_bench"],
    }
    path = ROOT / "data" / "cad" / "provenance.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
