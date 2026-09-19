"""Race UI backend: streams every lane side by side, shows the held-out scoreboard, collects corrections.

  uv run uvicorn app.server:app --port 8000        # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

from understudy.config import ROOT, Lane, race_lanes
from understudy.data import append_jsonl, read_jsonl
from understudy.llm import stream_chat
from understudy.tasks import load_task

TASK = load_task()
STATIC = Path(__file__).parent / "static"
CORRECTIONS = ROOT / "data" / "corrections.jsonl"
app = FastAPI(title="Understudy")


class RaceRequest(BaseModel):
    text: str
    lanes: list[str] | None = None


class Correction(BaseModel):
    text: str
    output: dict
    field: str
    value: Any = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict:
    return {
        "task": TASK.name,
        "lanes": [{"key": l.key, "label": l.label, "model": l.model, "dedicated": l.dedicated} for l in race_lanes()],
        "checks": [c.name for c in TASK.checks],
        "fields": TASK.correctable_fields,
    }


@app.get("/api/examples")
def examples() -> list[dict]:
    demo = ROOT / "data" / "demo" / f"{TASK.name}.jsonl"
    rows = read_jsonl(demo) or read_jsonl(ROOT / "data" / "samples" / f"{TASK.name}_inputs.jsonl")
    return [
        {"id": r["id"], "title": r.get("title") or r["id"], "text": r.get("text") or "\n".join(r.get("lines", []))}
        for r in rows
    ]


@app.post("/api/race")
async def race(req: RaceRequest) -> StreamingResponse:
    record = {"id": "live", "source_id": "live", "text": req.text}
    lanes = [l for l in race_lanes() if not req.lanes or l.key in req.lanes]
    queue: asyncio.Queue = asyncio.Queue()
    session = uuid.uuid4().hex[:12]

    async def run(lane: Lane) -> None:
        try:
            async for event in stream_chat(
                lane, TASK.messages(record), schema=TASK.json_schema(), session_id=f"race-{session}-{lane.key}"
            ):
                if event["type"] != "done":
                    await queue.put({"lane": lane.key, **event})
                    continue
                result = event["result"]
                output, parse_error = (None, None) if result.error else TASK.parse(result.content)
                await queue.put(
                    {
                        "lane": lane.key,
                        "type": "done",
                        "metrics": result.metrics(),
                        "error": result.error,
                        "parse_error": parse_error,
                        "output": output.model_dump() if output else None,
                        "checks": TASK.run_checks(record, output) if output else {},
                    }
                )
        except Exception as e:  # keep the other lanes racing
            await queue.put({"lane": lane.key, "type": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            await queue.put({"lane": lane.key, "type": "_end"})

    async def events():
        tasks = [asyncio.create_task(run(lane)) for lane in lanes]
        remaining = len(tasks)
        while remaining:
            event = await queue.get()
            if event["type"] == "_end":
                remaining -= 1
                continue
            yield f"data: {json.dumps(event)}\n\n"
        yield 'data: {"type": "all_done"}\n\n'

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/scoreboard")
def scoreboard() -> dict:
    pinned = os.getenv("SCOREBOARD")
    if pinned:
        return json.loads((ROOT / pinned).read_text())
    runs = sorted((ROOT / "runs").glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(runs[-1].read_text()) if runs else {}


@app.post("/api/correct")
def correct(c: Correction) -> dict:
    if c.field not in TASK.correctable_fields:
        raise HTTPException(422, f"unknown field {c.field}")
    label = {**c.output, c.field: c.value}
    try:
        TASK.output_model.model_validate(label)
    except ValidationError as e:
        raise HTTPException(422, e.errors()[0]["msg"])
    source = "live-" + hashlib.sha1(c.text.encode()).hexdigest()[:10]
    append_jsonl(
        CORRECTIONS,
        {
            "record": {"id": source, "source_id": source, "text": c.text},
            "label": label,
            "corrected_field": c.field,
            "ts": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return {"count": len(read_jsonl(CORRECTIONS))}
