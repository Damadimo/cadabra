"""Specialist first; escalate to the frontier model only when a deterministic check fails."""

from __future__ import annotations

from .config import Lane
from .llm import complete
from .tasks import TaskSpec


async def run_cascade(task: TaskSpec, record: dict, specialist: Lane, fallback: Lane) -> dict:
    calls = []
    output, error = None, None
    for lane in (specialist, fallback):
        r = await complete(lane, task.messages(record), schema=task.json_schema())
        calls.append(r)
        output, error = (None, r.error) if r.error else task.parse(r.content)
        if output is not None and (lane is fallback or all(task.run_checks(record, output).values())):
            return {"output": output, "error": None, "calls": calls, "answered_by": lane.key}
    return {"output": output, "error": error, "calls": calls, "answered_by": fallback.key}
