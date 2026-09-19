"""Benchmark lanes on held-out specs: every lane gets the same prompt; the geometry checker grades every answer.

  uv run python -m understudy.cad.bench --lanes kimi-k3,glm-5.3 --n 100 --shots 2 --tag pilot
  uv run python -m understudy.cad.bench --lanes specialist,base-4b --n 500 --shots 0 --tag ours
  uv run python -m understudy.cad.bench --lanes kimi-k3 --n 100 --retries 2 --tag k3-retries   # repair loop

Lanes come from understudy.config.lanes() or are given inline as slug:effort (e.g. zai-org/GLM-5.3-Flash:low).
Success = the code runs and the aligned IoU with the reference is at least 0.9. Writes runs/<time>_<tag>/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime
from statistics import mean, median

from ..config import PRETTY, ROOT, Lane, lanes as known_lanes
from ..data import read_jsonl, write_jsonl
from ..llm import complete
from ..stats import bootstrap_ci, pct
from . import prompts
from .pool import CadPool

DATA = ROOT / "data" / "cad"
SUCCESS_IOU = 0.9


def resolve_lanes(spec: str) -> list[Lane]:
    known = known_lanes()
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        if item in known:
            out.append(known[item])
            continue
        if "/" not in item.partition(":")[0]:
            print(f"skipping lane {item!r}: not configured (set UNDERSTUDY_BASE_URL for specialist/base-4b)")
            continue
        slug, _, effort = item.partition(":")
        key = slug.split("/")[-1].lower() + (f"@{effort}" if effort else "")
        label = PRETTY.get(slug, slug.split("/")[-1]) + (f" ({effort})" if effort else "")
        out.append(Lane(key, label, slug, reasoning_effort=effort or None))
    return out


def sample(records: list[dict], n: int | None, seed: int) -> list[dict]:
    """Deterministic sample that keeps the single/multi-part mix of the file."""
    if not n or n >= len(records):
        return records
    rng = random.Random(seed)
    multi = [r for r in records if r["n_parts"] > 1]
    single = [r for r in records if r["n_parts"] == 1]
    k_multi = round(n * len(multi) / len(records))
    return sorted(rng.sample(multi, min(k_multi, len(multi))) + rng.sample(single, n - min(k_multi, len(multi))), key=lambda r: r["id"])


IMAGES = DATA / "img"


def image_inputs(rec: dict, index: dict) -> tuple[str, list[float]]:
    from .render import image_path

    return prompts.data_url(image_path(rec["id"], IMAGES)), index[rec["id"]]["bbox"]


async def best_of_n(lane: Lane, rec: dict, pool: CadPool, msgs: list[dict], n: int, max_tokens: int, temperature: float, slot: int) -> dict:
    """Sample n programs, keep the one whose render best matches the INPUT sheet (render-and-compare), grade that one.
    Also grades every candidate against the answer key, only to report how often the best was available (oracle@n)."""
    from .render import image_path

    calls = await asyncio.gather(
        *(complete(lane, msgs, max_tokens=max_tokens, temperature=temperature, session_id=f"cad-{lane.key}-{slot % 4}", restart_on_disconnect=True) for _ in range(n)),
        return_exceptions=True,
    )
    calls = [c for c in calls if not isinstance(c, BaseException)]
    codes = [prompts.extract_code(c.content) if not c.error else None for c in calls]
    sheet = str(image_path(rec["id"], IMAGES))
    live = [(i, code) for i, code in enumerate(codes) if code]
    checks = await asyncio.gather(*(asyncio.to_thread(pool.run, _fn="render_score", code=code, sheet_path=sheet) for _, code in live))
    grades = await asyncio.gather(*(asyncio.to_thread(pool.run, code=code, gold_code=rec["gold_code"], want_chamfer=True) for _, code in live))
    if not live:
        return {"graded": {"runs": False, "error": "no candidate produced code"}, "code": None, "calls": calls, "extra": {"candidates": len(calls)}}
    ranked = sorted(range(len(live)), key=lambda k: (-(checks[k].get("render_score") or 0.0), k))
    pick = ranked[0]
    ious = [(g.get("iou_aligned") or 0.0) if g.get("runs") else 0.0 for g in grades]
    extra = {
        "candidates": len(calls),
        "render_score": checks[pick].get("render_score"),
        "oracle_iou": max(ious),
        "candidates_correct": sum(i >= SUCCESS_IOU for i in ious),
    }
    return {"graded": grades[pick], "code": live[pick][1], "calls": calls, "extra": extra}


async def solve(lane: Lane, rec: dict, pool: CadPool, shots: list[dict], retries: int, max_tokens: int, slot: int, index: dict | None = None,
                best_of: int = 1, temperature: float = 0.7) -> dict:
    if best_of > 1 and index is not None:
        image, bbox = image_inputs(rec, index)
        out = await best_of_n(lane, rec, pool, prompts.image_messages(image, bbox, shots), best_of, max_tokens, temperature, slot)
        return row_for(lane, rec, out["graded"], out["code"], out["calls"], 1, out["extra"])
    if index is not None:
        image, bbox = image_inputs(rec, index)
        msgs = prompts.image_messages(image, bbox, shots)
    else:
        msgs = prompts.messages(rec["spec"], shots)
    attempts, calls = 0, []
    graded: dict = {"runs": False, "error": "no attempt"}
    code = None
    while attempts <= retries:
        attempts += 1
        try:
            r = await complete(
                lane,
                msgs,
                max_tokens=max_tokens,
                temperature=0.0 if lane.dedicated else None,
                session_id=f"cad-{lane.key}-{slot % 4}",
                restart_on_disconnect=True,
            )
        except Exception as e:  # noqa: BLE001 - record and move on
            graded = {"runs": False, "error": f"api: {type(e).__name__}: {str(e)[:200]}"}
            break
        calls.append(r)
        if r.error:
            graded = {"runs": False, "error": f"api: {r.error}"}
            break
        code = prompts.extract_code(r.content)
        if code is None:
            graded = {"runs": False, "error": "no code block in the reply"}
        else:
            graded = await asyncio.to_thread(pool.run, code=code, gold_code=rec["gold_code"], want_chamfer=True)
        if graded.get("runs"):
            break
        msgs = msgs + [
            {"role": "assistant", "content": r.content},
            {"role": "user", "content": prompts.REPAIR.format(error=graded.get("error"))},
        ]
    return row_for(lane, rec, graded, code, calls, attempts)


def row_for(lane: Lane, rec: dict, graded: dict, code: str | None, calls: list, attempts: int, extra: dict | None = None) -> dict:
    costs = [c.cost_usd for c in calls]
    iou_a = graded.get("iou_aligned") or 0.0
    return {
        **(extra or {}),
        "id": rec["id"],
        "lane": lane.key,
        "model": lane.model,
        "n_parts": rec["n_parts"],
        "n_faces": rec.get("n_faces"),
        "runs": bool(graded.get("runs")),
        "success": bool(graded.get("runs")) and iou_a >= SUCCESS_IOU,
        "iou": graded.get("iou"),
        "iou_aligned": graded.get("iou_aligned"),
        "chamfer": graded.get("chamfer"),
        "error": graded.get("error"),
        "attempts": attempts,
        "ttft_s": calls[0].ttft_s if calls else None,
        # sequential attempts add up; best-of-n samples run in parallel, so the slowest one is the latency
        "e2e_s": (max(c.e2e_s for c in calls) if extra else sum(c.e2e_s for c in calls)) if calls else 0.0,
        "output_tokens": sum(c.output_tokens for c in calls),
        "reasoning_chars": sum(len(c.reasoning) for c in calls),
        "cost_usd": None if any(c is None for c in costs) else sum(costs),
        "code": code,
    }


def summarize(rows: list[dict], lane: Lane, wall_s: float, concurrency: int) -> dict:
    n = len(rows)
    succ = [1.0 if r["success"] else 0.0 for r in rows]
    lo, hi = bootstrap_ci(succ)
    ran = [r for r in rows if r["runs"]]
    if lane.dedicated:
        cost_1k = lane.gpu_hourly_usd / 3600 * wall_s / n * 1000
        cost_note = f"GPU ${lane.gpu_hourly_usd:.2f}/h amortized at concurrency {concurrency}"
    else:
        costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
        cost_1k = mean(costs) * 1000 if costs else None
        cost_note = "list price per token"

    def slice_rate(pred):
        part = [r for r in rows if pred(r)]
        return {"n": len(part), "success": mean(1.0 if r["success"] else 0.0 for r in part) if part else None}

    return {
        "lane": lane.key,
        "label": lane.label,
        "model": lane.model,
        "reasoning_effort": lane.reasoning_effort,
        "n": n,
        "success": mean(succ),
        "success_ci95": [lo, hi],
        "success_iou95": mean(1.0 if r["runs"] and (r["iou_aligned"] or 0) >= 0.95 else 0.0 for r in rows),
        "run_rate": len(ran) / n,
        "mean_iou_aligned": mean((r["iou_aligned"] or 0.0) if r["runs"] else 0.0 for r in rows),
        "median_chamfer": median([r["chamfer"] for r in ran if r["chamfer"] is not None]) if ran else None,
        "single_part": slice_rate(lambda r: r["n_parts"] == 1),
        "multi_part": slice_rate(lambda r: r["n_parts"] > 1),
        "latency_p50": pct((r["e2e_s"] for r in rows), 50),
        "latency_p95": pct((r["e2e_s"] for r in rows), 95),
        "ttft_p50": pct((r["ttft_s"] for r in rows), 50),
        "output_tokens_mean": mean(r["output_tokens"] for r in rows),
        "attempts_mean": mean(r["attempts"] for r in rows),
        "oracle_success": mean(1.0 if (r.get("oracle_iou") or 0) >= SUCCESS_IOU else 0.0 for r in rows) if rows and "oracle_iou" in rows[0] else None,
        "cost_per_1k_usd": cost_1k,
        "cost_note": cost_note,
        "wall_s": wall_s,
    }


def _p(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def _f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def markdown(summaries: list[dict], meta: dict) -> str:
    lines = [
        f"`{meta['data']}` · {meta['n']} held-out {'drawing sheets' if meta.get('modality') == 'image' else 'specs'} · shots {meta['shots']} · retries {meta['retries']} · {meta['created']}",
        "",
        "| Lane | Model (effort) | Success (IoU ≥ 0.9) [95% CI] | IoU ≥ 0.95 | Runs | Mean IoU | Median Chamfer | Single / multi-part | Latency p50 / p95 (s) | Out tokens | $ / 1K parts |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        ci = s["success_ci95"]
        lines.append(
            f"| {s['lane']} | `{s['model']}` ({s['reasoning_effort'] or 'default'}) | **{_p(s['success'])}** [{_p(ci[0])}, {_p(ci[1])}] "
            f"| {_p(s['success_iou95'])} | {_p(s['run_rate'])} | {_f(s['mean_iou_aligned'], 3)} | {_f(s['median_chamfer'], 4)} "
            f"| {_p(s['single_part']['success'])} / {_p(s['multi_part']['success'])} | {_f(s['latency_p50'], 1)} / {_f(s['latency_p95'], 1)} "
            f"| {_f(s['output_tokens_mean'], 0)} | {'–' if s['cost_per_1k_usd'] is None else '$' + format(s['cost_per_1k_usd'], '.2f')} |"
        )
    lines += ["", "Costs: " + "; ".join(f"{s['lane']}: {s['cost_note']}" for s in summaries)]
    return "\n".join(lines) + "\n"


async def run(args) -> None:
    os.environ["LLM_TIMEOUT"] = str(args.timeout)
    records = read_jsonl(args.data)
    index = None
    if args.modality == "image":
        index = json.loads((IMAGES / "index.json").read_text())
        records = [r for r in records if r["id"] in index]
    if args.multi_only:
        records = [r for r in records if r["n_parts"] > 1]
    if args.min_faces:
        records = [r for r in records if (r.get("n_faces") or 0) >= args.min_faces]
    records = sample(records, args.n, args.seed)
    shots = read_jsonl(DATA / "shots.jsonl")[: args.shots] if args.shots else []
    if args.modality == "image" and shots:
        shots = [{"image": image_inputs(s, index)[0], "bbox": index[s["id"]]["bbox"], "gold_code": s["gold_code"]} for s in shots]
    lanes = resolve_lanes(args.lanes)
    created = datetime.now()
    out_dir = ROOT / "runs" / f"{created:%Y%m%d-%H%M%S}_{args.tag}"
    print(f"{len(records)} specs ({sum(r['n_parts'] > 1 for r in records)} multi-part) x {[l.key for l in lanes]}", flush=True)
    with CadPool(workers=args.workers) as pool:

        async def lane_run(lane: Lane):
            sem = asyncio.Semaphore(args.concurrency)
            done = 0

            async def one(i, rec):
                nonlocal done
                async with sem:
                    lane_shots = [] if (lane.dedicated and not args.shots_for_ours) else shots
                    row = await solve(lane, rec, pool, lane_shots, args.retries, args.max_tokens if not lane.dedicated else 2048, i, index,
                                      best_of=args.best_of, temperature=args.temperature)
                done += 1
                if done % 10 == 0 or done == len(records):
                    print(f"  {lane.key}: {done}/{len(records)}", flush=True)
                return row

            t0 = time.perf_counter()
            rows = await asyncio.gather(*(one(i, r) for i, r in enumerate(records)))
            return rows, summarize(rows, lane, time.perf_counter() - t0, args.concurrency)

        results = await asyncio.gather(*(lane_run(l) for l in lanes))
    rows = [row for lane_rows, _ in results for row in lane_rows]
    summaries = [s for _, s in results]
    meta = {
        "data": str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data),
        "n": len(records),
        "shots": args.shots,
        "retries": args.retries,
        "modality": args.modality,
        "best_of": args.best_of,
        "created": created.isoformat(timespec="seconds"),
    }
    write_jsonl(out_dir / "results.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps({**meta, "lanes": summaries}, indent=2))
    (out_dir / "summary.md").write_text(markdown(summaries, meta))
    print("\n" + markdown(summaries, meta))
    print(f"saved {out_dir.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=lambda p: (ROOT / p) if not p.startswith("/") else __import__("pathlib").Path(p), default=DATA / "bench.jsonl")
    ap.add_argument("--lanes", default="kimi-k3,glm-5.3")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=2, help="worked examples shown to API lanes")
    ap.add_argument("--shots-for-ours", action="store_true", help="also show the examples to dedicated lanes")
    ap.add_argument("--retries", type=int, default=0, help="repair attempts after code that fails to run")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--modality", choices=["text", "image"], default="text", help="text specs, or rendered drawing sheets")
    ap.add_argument("--best-of", type=int, default=1, help="image mode: sample N programs, keep the best render-and-compare match")
    ap.add_argument("--temperature", type=float, default=0.7, help="sampling temperature for --best-of")
    ap.add_argument("--multi-only", action="store_true", help="only multi-part specs")
    ap.add_argument("--min-faces", type=int, default=0, help="only parts with at least this many faces")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-request timeout (long reasoning runs)")
    ap.add_argument("--tag", default="bench")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
