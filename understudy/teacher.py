"""Label inputs with a strong teacher and keep only outputs that pass every check.

  uv run python -m understudy.teacher --inputs data/raw/inputs.jsonl --perturb 2 --rewrites 1

Input rows: {"id", "source_id"?, "text"} (seclogs may use "lines"). Output rows are appended to --out as
they finish, so the run resumes where it stopped. Rejected rows are kept with their failed checks: read them.
Items the teacher fails go to the adjudicator (Kimi K3 by default; --adjudicator none to skip).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import time

from .checks import strip_think
from .config import ADJUDICATOR_MODEL, BULK_MODEL, TEACHER_MODEL, Lane
from .data import append_jsonl, perturb, read_jsonl, source_of
from .llm import complete
from .tasks import TaskSpec, load_task

REWRITE_PROMPT = (
    "Rewrite this document the way a different, hurried sender might: reorder sections, abbreviate, change the "
    "formatting, add a few typos. Keep EVERY fact, number, name and code exactly the same. "
    "Output only the rewritten document."
)


async def label_one(task: TaskSpec, record: dict, teacher: Lane, adjudicator: Lane | None) -> dict:
    attempts = []
    for lane in [teacher] + ([adjudicator] if adjudicator else []):
        r = await complete(lane, task.messages(record), schema=task.json_schema(), max_tokens=8192, session_id=f"teach-{lane.key}")
        output, error = (None, r.error) if r.error else task.parse(r.content)
        checks = task.run_checks(record, output) if output else {}
        attempts.append({"model": lane.model, "error": error, "checks": checks, "cost_usd": r.cost_usd, "e2e_s": r.e2e_s})
        if output and all(checks.values()):
            return {"status": "accepted", "label": output.model_dump(), "labeled_by": lane.model, "attempts": attempts}
    return {"status": "rejected", "label": None, "labeled_by": None, "attempts": attempts}


async def rewrite(record: dict, k: int, lane: Lane) -> dict | None:
    source_text = record.get("text") or "\n".join(record.get("lines", []))
    messages = [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": source_text}]
    r = await complete(lane, messages, max_tokens=8192, temperature=0.9)
    text = strip_think(r.content).strip()
    if r.error or not text:
        return None
    base = {key: value for key, value in record.items() if key != "lines"}
    return {**base, "id": f"{record['id']}~r{k}", "source_id": source_of(record), "derived_from": record["id"], "text": text}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", default="data/labels/teacher.jsonl")
    ap.add_argument("--teacher", default=TEACHER_MODEL)
    ap.add_argument("--teacher-effort", default="high")
    ap.add_argument("--adjudicator", default=ADJUDICATOR_MODEL, help='"none" to disable')
    ap.add_argument("--adjudicator-effort", default="high")
    ap.add_argument("--perturb", type=int, default=0, help="surface-noise variants per input (free)")
    ap.add_argument("--rewrites", type=int, default=0, help="LLM rewrites per input via the bulk model (cheap)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    task = load_task()
    teacher = Lane("teacher", "teacher", args.teacher, reasoning_effort=args.teacher_effort or None)
    adjudicator = (
        None
        if args.adjudicator.lower() == "none"
        else Lane("adjudicator", "adjudicator", args.adjudicator, reasoning_effort=args.adjudicator_effort or None)
    )
    bulk = Lane("bulk", "bulk", BULK_MODEL, reasoning_effort="low")

    inputs = read_jsonl(args.inputs)[: args.limit]
    done = {row["id"] for row in read_jsonl(args.out)}
    records = list(inputs) + [perturb(rec, k, args.seed) for rec in inputs for k in range(args.perturb)]
    pending_rewrites = [(rec, k) for rec in inputs for k in range(args.rewrites) if f"{rec['id']}~r{k}" not in done]
    if pending_rewrites:
        print(f"Rewriting {len(pending_rewrites)} variants with {bulk.model}...", flush=True)
        rewritten = await asyncio.gather(*(rewrite(rec, k, bulk) for rec, k in pending_rewrites))
        records += [r for r in rewritten if r]
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(todo)} to label ({len(done)} already in {args.out}); teacher={teacher.model}", flush=True)
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)
    status = collections.Counter()
    failures = collections.Counter()
    spend = 0.0
    t0 = time.time()

    async def work(record: dict) -> None:
        nonlocal spend
        async with sem:
            result = await label_one(task, record, teacher, adjudicator)
        append_jsonl(args.out, {"id": record["id"], "source_id": source_of(record), "record": record, **result})
        status[result["status"]] += 1
        for attempt in result["attempts"]:
            spend += attempt["cost_usd"] or 0.0
            if attempt["error"]:
                failures["api or parse error"] += 1
            failures.update(name for name, ok in attempt["checks"].items() if not ok)
        n = sum(status.values())
        if n % 10 == 0 or n == len(todo):
            print(f"[{n}/{len(todo)}] accepted={status['accepted']} rejected={status['rejected']} spend=${spend:.2f}", flush=True)

    await asyncio.gather(*(work(r) for r in todo))
    total = sum(status.values()) or 1
    print(f"\nDone in {time.time() - t0:.0f}s. Accepted {status['accepted']}/{total} ({status['accepted'] / total:.0%}), spend ${spend:.2f}")
    if failures:
        print("Most common failures (read these rows):")
        for name, count in failures.most_common(8):
            print(f"  {count:5d}  {name}")


if __name__ == "__main__":
    asyncio.run(main())
