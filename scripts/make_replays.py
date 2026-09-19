"""Build demo replays from benchmark results, so the race can be shown with no API calls, GPU or network.

  uv run python scripts/make_replays.py --runs sota-img,ours-img                 # parts from data/cad/demo.json
  uv run python scripts/make_replays.py --runs sota-img --ids train_high:01505

Each lane's saved answer is replayed at its measured speed (first token, then the code at its end-to-end latency),
then built and shown over the target, exactly as the live race grades it. Written to data/demo/replays/<id>-image.json
(+ meshes/); play with /?replay=<name>. The UI labels these as answers from the benchmark run, not live calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from understudy.cad.bench import infra_failed
from understudy.cad.pool import CadPool
from understudy.cad.render import OUT as IMAGES
from understudy.config import PRETTY, ROOT
from understudy.data import read_jsonl

REPLAYS = ROOT / "data" / "demo" / "replays"
TEXT_ONLY = {"glm-5.3@high"}


def latest(tag: str) -> Path:
    found = sorted((ROOT / "runs").glob(f"*_{tag}"), key=lambda p: p.name)
    if not found:
        raise SystemExit(f"no run tagged {tag}")
    return found[-1]


def keep_mesh(stl: bytes) -> str:
    key = hashlib.sha1(stl).hexdigest()[:16]  # same key scheme as app/server.py store()
    (REPLAYS / "meshes").mkdir(parents=True, exist_ok=True)
    (REPLAYS / "meshes" / f"{key}.stl").write_bytes(stl)
    return key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="sota-img", help="comma-separated run tags (newest run of each)")
    ap.add_argument("--ids", default="", help="comma-separated part ids (default: data/cad/demo.json)")
    args = ap.parse_args()
    bench = {r["id"]: r for r in read_jsonl(ROOT / "data" / "cad" / "bench.jsonl")}
    sheets = json.loads((IMAGES / "index.json").read_text())
    ids = [i for i in args.ids.split(",") if i] or json.loads((ROOT / "data" / "cad" / "demo.json").read_text())
    rows: dict[str, dict[str, dict]] = {}
    lanes: dict[str, dict] = {}
    sources = []
    for tag in args.runs.split(","):
        run = latest(tag)
        sources.append(run.name)
        meta = json.loads((run / "summary.json").read_text())
        for s in meta["lanes"]:
            if s["lane"] not in TEXT_ONLY:
                effort = s.get("reasoning_effort")
                label = s["label"] if s.get("label") not in (None, s["lane"]) else PRETTY.get(s["model"], s["lane"]) + (f" ({effort})" if effort else "")
                lanes[s["lane"]] = {"key": s["lane"], "label": label + (f" · best of {meta['best_of']}" if meta.get("best_of", 1) > 1 else ""),
                                    "model": s["model"], "effort": s.get("reasoning_effort"), "dedicated": s["lane"] in ("specialist", "base-4b")}
        for r in read_jsonl(run / "results.jsonl"):
            if r["lane"] in lanes and not infra_failed(r):
                rows.setdefault(r["id"], {})[r["lane"]] = r
    order = sorted(lanes, key=lambda k: (not lanes[k]["dedicated"], k))
    with CadPool(workers=4, timeout=60) as pool:
        for part in ids:
            got = rows.get(part, {})
            if part not in bench or not got:
                print(f"skip {part}: no answers in {sources}")
                continue
            gold = bench[part]["gold_code"]
            target = pool.run(code=gold, want_mesh=True, want_chamfer=False)
            events = [{"t": 0.0, "type": "target", "mesh": keep_mesh(target["mesh_stl"]), "stats": target["stats"]}]
            for k in order:
                r = got.get(k)
                if r is None:
                    continue
                ttft, e2e = r.get("ttft_s") or 0.5, max(r.get("e2e_s") or 1.0, 0.6)
                events.append({"t": round(min(ttft, e2e - 0.1), 3), "lane": k, "type": "reasoning", "text": "…"})
                events.append({"t": round(min(ttft, e2e - 0.1), 3), "lane": k, "type": "status", "text": "answer from the benchmark run, replayed at its measured speed"})
                if r.get("code"):
                    events.append({"t": round(e2e, 3), "lane": k, "type": "content", "text": r["code"]})
                metrics = {"ttft_s": r.get("ttft_s"), "e2e_s": e2e, "output_tokens": r.get("output_tokens"), "cost_usd": r.get("cost_usd")}
                events.append({"t": round(e2e, 3), "lane": k, "type": "generated", "metrics": metrics, "error": None})
                graded = {"t": round(e2e + 0.4, 3), "lane": k, "type": "graded", "code": r.get("code"), "runs": False, "error": r.get("error")}
                if r.get("runs") and r.get("code"):
                    res = pool.run(code=r["code"], gold_code=gold, want_mesh=True, want_chamfer=True)
                    graded.update(runs=bool(res.get("runs")), error=res.get("error"), iou=res.get("iou"), iou_aligned=res.get("iou_aligned"),
                                  chamfer=res.get("chamfer"), stats=res.get("stats"))
                    if res.get("runs"):
                        graded["mesh"] = keep_mesh(res.get("aligned_mesh_stl") or res["mesh_stl"])
                    if r.get("candidates"):
                        graded.update(candidates=r["candidates"], render_score=r.get("render_score"))
                events.append(graded)
            events.sort(key=lambda e: e["t"])
            payload = {"example_id": part, "modality": "image", "spec": bench[part]["spec"], "source": "benchmark run " + ", ".join(sources),
                       "sheet": f"/api/sheet/{part}", "bbox": sheets.get(part, {}).get("bbox"),
                       "lanes": [lanes[k] for k in order if k in got], "events": events}
            name = f"{part.replace(':', '_')}-image"
            (REPLAYS / f"{name}.json").write_text(json.dumps(payload))
            summary = ", ".join(f"{k} {'✓' if (got[k].get('success')) else '✗'} {got[k].get('e2e_s') or 0:.0f}s" for k in order if k in got)
            print(f"wrote replay {name}: {summary}")


if __name__ == "__main__":
    main()
