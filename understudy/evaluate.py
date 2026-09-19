"""Score lanes on the frozen gold set: accuracy with bootstrap CIs, check pass rate, latency, cost.

  uv run python -m understudy.evaluate --gold data/gold/gold.jsonl --lanes specialist,kimi-k3,glm-5.3
  uv run python -m understudy.evaluate --gold data/gold/gold.jsonl --lanes specialist,glm-5.3-flash --concurrency 16 --tag c16
  uv run python -m understudy.evaluate ... --lanes cascade          # specialist, escalating to Kimi K3 on failed checks
  uv run python -m understudy.evaluate ... --effort kimi-k3=high     # override reasoning effort per lane

Gold rows: {"id", "source_id", "text", "gold": {<score fields and decision>}}. Writes
runs/<timestamp>_<tag>/{results.jsonl, summary.json, summary.md}; the race UI shows the newest summary.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time
from datetime import datetime
from statistics import mean

from .cascade import run_cascade
from .checks import same
from .config import ROOT, Lane, lanes as all_lanes
from .data import read_jsonl, write_jsonl
from .llm import complete
from .stats import bootstrap_ci, pct, safe_mean
from .tasks import TaskSpec, load_task


async def run_item(task: TaskSpec, key: str, lanes: dict[str, Lane], row: dict, slot: int) -> dict:
    if key == "cascade":
        return await run_cascade(task, row, lanes["specialist"], lanes["kimi-k3"])
    lane = lanes[key]
    r = await complete(lane, task.messages(row), schema=task.json_schema(), session_id=f"eval-{key}-{slot % 4}")
    output, error = (None, r.error) if r.error else task.parse(r.content)
    return {"output": output, "error": error, "calls": [r], "answered_by": key}


def score(task: TaskSpec, row: dict, res: dict) -> dict:
    out, gold, calls = res["output"], row["gold"], res["calls"]
    checks = task.run_checks(row, out) if out else {}
    fields = task.field_matches(out, gold) if out else {f: False for f in task.score_fields}
    decision_ok = bool(out and task.decision_field and same(getattr(out, task.decision_field), gold.get(task.decision_field)))
    costs = [c.cost_usd for c in calls]
    return {
        "id": row["id"],
        "answered_by": res["answered_by"],
        "error": res["error"],
        "decision_ok": decision_ok,
        "fields": fields,
        "checks": checks,
        "all_checks": bool(checks) and all(checks.values()),
        "exact": bool(out) and all(fields.values()) and decision_ok,
        "ttft_s": calls[0].ttft_s,
        "e2e_s": sum(c.e2e_s for c in calls),
        "tok_per_s": calls[-1].tok_per_s,
        "output_tokens": sum(c.output_tokens for c in calls),
        "cost_usd": None if any(c is None for c in costs) else sum(costs),
        "output": out.model_dump() if out else None,
    }


def summarize(key: str, lane: Lane | None, scored: list[dict], wall_s: float, concurrency: int) -> dict:
    n = len(scored)
    decisions = [1.0 if s["decision_ok"] else 0.0 for s in scored]
    lo, hi = bootstrap_ci(decisions)
    if lane is not None and lane.dedicated:
        cost_1k = lane.gpu_hourly_usd / 3600 * wall_s / n * 1000
        cost_note = f"GPU ${lane.gpu_hourly_usd:.2f}/h amortized over this run at concurrency {concurrency}"
    else:
        costs = [s["cost_usd"] for s in scored if s["cost_usd"] is not None]
        cost_1k = mean(costs) * 1000 if costs else None
        cost_note = "list price per token" if key != "cascade" else "specialist GPU-seconds (c=1) + Kimi K3 tokens"
    return {
        "lane": key,
        "model": lane.model if lane else "specialist -> moonshotai/Kimi-K3",
        "n": n,
        "decision_acc": mean(decisions),
        "decision_ci95": [lo, hi],
        "field_acc": safe_mean([mean(s["fields"].values()) for s in scored]),
        "exact_match": mean(1.0 if s["exact"] else 0.0 for s in scored),
        "checks_pass": mean(1.0 if s["all_checks"] else 0.0 for s in scored),
        "errors": sum(1 for s in scored if s["error"]),
        "ttft_p50": pct((s["ttft_s"] for s in scored), 50),
        "e2e_p50": pct((s["e2e_s"] for s in scored if not s["error"]), 50),
        "e2e_p95": pct((s["e2e_s"] for s in scored if not s["error"]), 95),
        "tok_per_s_p50": pct((s["tok_per_s"] for s in scored), 50),
        "output_tokens_mean": safe_mean(s["output_tokens"] for s in scored),
        "cost_per_1k_usd": cost_1k,
        "cost_note": cost_note,
        "escalation_rate": mean(1.0 if s["answered_by"] != "specialist" else 0.0 for s in scored) if key == "cascade" else None,
        "wall_s": wall_s,
        "reasoning_effort": lane.reasoning_effort if lane else None,
    }


def _f(x: float | None, digits: int = 2) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def _p(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def markdown(summaries: list[dict], meta: dict) -> str:
    lines = [
        f"Task `{meta['task']}` · {meta['n_items']} held-out items · concurrency {meta['concurrency']} · {meta['created']}",
        "",
        "| Lane | Model | Decision acc [95% CI] | Field acc | Checks pass | Latency p50 / p95 (s) | TTFT p50 (s) | Out tok/s p50 | $ / 1K tasks |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        ci = s["decision_ci95"]
        lines.append(
            f"| {s['lane']} | `{s['model']}` | {_p(s['decision_acc'])} [{_p(ci[0])}, {_p(ci[1])}] | {_p(s['field_acc'])} "
            f"| {_p(s['checks_pass'])} | {_f(s['e2e_p50'])} / {_f(s['e2e_p95'])} | {_f(s['ttft_p50'])} "
            f"| {_f(s['tok_per_s_p50'], 0)} | {'–' if s['cost_per_1k_usd'] is None else '$' + format(s['cost_per_1k_usd'], '.2f')} |"
        )
    lines += ["", "Costs: " + "; ".join(f"{s['lane']}: {s['cost_note']}" for s in summaries)]
    return "\n".join(lines) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/gold.jsonl")
    ap.add_argument("--lanes", default="specialist,kimi-k3,glm-5.3")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1, help="run each item N times (latency variance)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--effort", default="", help="per-lane reasoning effort, e.g. kimi-k3=high,glm-5.3=low")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--task")
    args = ap.parse_args()

    task = load_task(args.task)
    gold = read_jsonl(args.gold)[: args.limit]
    if not gold:
        raise SystemExit(f"No gold rows in {args.gold}")
    lanes = all_lanes()
    for pair in filter(None, args.effort.split(",")):
        key, effort = pair.split("=")
        lanes[key] = dataclasses.replace(lanes[key], reasoning_effort=None if effort == "default" else effort)
    wanted = [k.strip() for k in args.lanes.split(",") if k.strip()]
    needed = {k for k in wanted if k != "cascade"} | ({"specialist", "kimi-k3"} if "cascade" in wanted else set())
    if missing := sorted(needed - lanes.keys()):
        raise SystemExit(f"Unknown lanes {missing}. Known: {sorted(lanes)} (set UNDERSTUDY_BASE_URL for specialist/base-4b)")

    created = datetime.now()
    run_dir = ROOT / "runs" / f"{created:%Y%m%d-%H%M%S}_{args.tag}"
    summaries, rows = [], []
    for key in wanted:
        sem = asyncio.Semaphore(args.concurrency)

        async def one(slot: int, row: dict) -> dict:
            async with sem:
                res = await run_item(task, key, lanes, row, slot)
            return score(task, row, res)

        t0 = time.perf_counter()
        scored = await asyncio.gather(*(one(i, row) for i, row in enumerate(gold * args.repeat)))
        summary = summarize(key, lanes.get(key), scored, time.perf_counter() - t0, args.concurrency)
        summaries.append(summary)
        rows += [{"lane": key, **s} for s in scored]
        print(f"{key:14} decision={_p(summary['decision_acc'])} fields={_p(summary['field_acc'])} "
              f"checks={_p(summary['checks_pass'])} p50={_f(summary['e2e_p50'])}s errors={summary['errors']}", flush=True)

    meta = {
        "task": task.name,
        "gold": args.gold,
        "n_items": len(gold),
        "repeat": args.repeat,
        "concurrency": args.concurrency,
        "created": created.isoformat(timespec="seconds"),
    }
    write_jsonl(run_dir / "results.jsonl", rows)
    (run_dir / "summary.json").write_text(json.dumps({**meta, "lanes": summaries}, indent=2))
    (run_dir / "summary.md").write_text(markdown(summaries, meta))
    print("\n" + markdown(summaries, meta))
    print(f"Saved to {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
