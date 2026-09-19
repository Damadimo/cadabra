"""Pick the parts for the live demo dropdown from finished benchmark runs (writes data/cad/demo.json).

  uv run python scripts/pick_demo.py --ours ours-img --frontier sota-img,sota-img-rest

Order: parts where ours is right and every frontier lane is wrong (medium/complex first, around 14 faces),
then one simple part everyone gets (a fair start), then one part ours misses (be honest about the limits).
Uses the newest run for each tag; prints the table so a human can veto before the demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cadabra.cad.bench import infra_failed
from cadabra.config import ROOT
from cadabra.data import read_jsonl


def latest(tag: str) -> Path:
    found = sorted((ROOT / "runs").glob(f"*_{tag}"), key=lambda p: p.name)
    if not found:
        raise SystemExit(f"no run tagged {tag}")
    return found[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="ours-img", help="tag of our run (lane 'specialist')")
    ap.add_argument("--frontier", default="sota-img", help="comma-separated tags of frontier runs")
    ap.add_argument("--wins", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cad" / "demo.json")
    args = ap.parse_args()
    ours = {r["id"]: r for r in read_jsonl(latest(args.ours) / "results.jsonl") if r["lane"] == "specialist" and not infra_failed(r)}
    frontier: dict[str, dict[str, dict]] = {}
    for tag in args.frontier.split(","):
        for r in read_jsonl(latest(tag) / "results.jsonl"):
            if not infra_failed(r) and r["lane"] != "glm-5.3@high":
                frontier.setdefault(r["id"], {})[r["lane"]] = r
    both = [i for i in ours if i in frontier and len(frontier[i]) >= 2]
    wins = [i for i in both if ours[i]["success"] and not any(f["success"] for f in frontier[i].values())]
    faces = lambda i: ours[i].get("n_faces") or 0  # noqa: E731
    wins.sort(key=lambda i: (faces(i) <= 6, abs(faces(i) - 14), i))  # medium/complex first, ~14 faces reads best on stage
    easy = [i for i in both if ours[i]["success"] and all(f["success"] for f in frontier[i].values()) and (ours[i].get("n_faces") or 0) <= 6]
    miss = [i for i in both if not ours[i]["success"] and any(f["success"] for f in frontier[i].values())]
    picks = wins[: args.wins] + easy[:1] + miss[:1]
    print(f"{len(both)} parts graded for every lane: ours right & frontier wrong {len(wins)}, all right {len(easy)}, ours wrong & a frontier lane right {len(miss)}")
    print("| part | faces | parts | ours IoU | " + " | ".join(sorted({k for i in picks for k in frontier[i]})) + " |")
    for i in picks:
        lanes = sorted({k for j in picks for k in frontier[j]})
        cells = [f"{(frontier[i].get(k) or {}).get('iou_aligned') or 0:.2f}" for k in lanes]
        print(f"| {i} | {ours[i].get('n_faces')} | {ours[i]['n_parts']} | {ours[i].get('iou_aligned') or 0:.2f} | " + " | ".join(cells) + " |")
    args.out.write_text(json.dumps(picks, indent=1))
    print(f"wrote {args.out} ({len(picks)} parts)")


if __name__ == "__main__":
    main()
