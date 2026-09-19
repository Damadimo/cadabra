"""Spec-to-CAD race: every lane writes CadQuery for the same spec; the server builds each part and scores it.

  uv run uvicorn app.server:app --port 8000        # then open http://127.0.0.1:8000

Lanes: RACE_LANES (default "specialist,moonshotai/Kimi-K3:high,zai-org/GLM-5.3:high"). Frontier lanes get the
same worked examples as in the benchmark; our model gets the zero-shot prompt it was trained on.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from understudy.cad import prompts
from understudy.cad.bench import resolve_lanes
from understudy.cad.pool import CadPool
from understudy.config import ROOT, Lane
from understudy.data import read_jsonl
from understudy.llm import stream_chat

STATIC = Path(__file__).parent / "static"
DATA = ROOT / "data" / "cad"
BENCH = {r["id"]: r for r in read_jsonl(DATA / "bench.jsonl")}
SHOTS = read_jsonl(DATA / "shots.jsonl")
MESHES: OrderedDict[str, bytes] = OrderedDict()
POOL: CadPool | None = None


def race_lanes() -> list[Lane]:
    return resolve_lanes(os.getenv("RACE_LANES", "specialist,moonshotai/Kimi-K3:high,zai-org/GLM-5.3:high"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global POOL
    POOL = await asyncio.to_thread(CadPool, int(os.getenv("CAD_WORKERS", "4")), 30.0)
    yield
    POOL.close()


app = FastAPI(title="Understudy-CAD", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def store(stl: bytes) -> str:
    key = hashlib.sha1(stl).hexdigest()[:16]
    MESHES[key] = stl
    MESHES.move_to_end(key)
    while len(MESHES) > 300:
        MESHES.popitem(last=False)
    return key


_NOUNS = (
    "block|bar|plate|prism|box|cube|cylinder|rod|pipe|tube|ring|disk|disc|washer|bracket|beam|panel|frame|flange|"
    "shaft|cap|slab|wedge|cone|column|post|bolt|nut|gear|housing|bushing|spacer|sleeve|hinge|clip|mount|rail|channel|"
    "star|hexagon|triangle|arch|bowl|cup|knob|handle|leg|table|chair|shelf|stand|holder|support|cover|lid|tray|container"
)
_TITLE = re.compile(rf"\b((?:(?!new |first |second |final |the |a |an )[a-z\-]+ ){{0,2}}(?:{_NOUNS}))s?\b", re.I)


_NAMED = re.compile(r"(?:dimensions of (?:the|this) |resulting |[Tt]he final )([a-z][a-z\- ]{2,30}?) (?:will|are|is|has|have|measures)")


def title_of(rec: dict) -> str:
    m = _NAMED.findall(rec["spec"])
    name = m[-1] if m and m[-1] not in ("part", "dimensions", "shape", "model") else None
    if name is None:
        n = _TITLE.search(rec["spec"])
        name = n.group(1) if n else "part"
    name = re.sub(r"^(?:final |dimensions of (?:the |this )?)+", "", name.strip()).capitalize() or "Part"
    return name if rec["n_parts"] == 1 else f"{name} ({rec['n_parts']} parts)"


class RaceRequest(BaseModel):
    spec: str
    example_id: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config() -> dict:
    return {"lanes": [{"key": l.key, "label": l.label, "model": l.model, "effort": l.reasoning_effort, "dedicated": l.dedicated} for l in race_lanes()]}


@app.get("/api/examples")
def examples() -> list[dict]:
    """Held-out benchmark specs for the dropdown: data/cad/demo.json if present, else a spread by complexity."""
    demo = DATA / "demo.json"
    ids = json.loads(demo.read_text()) if demo.exists() else None
    if not ids:
        recs = sorted(BENCH.values(), key=lambda r: (r["n_parts"], r.get("n_faces", 0)))
        ids = [recs[int(i * (len(recs) - 1) / 11)]["id"] for i in range(12)]
    out = []
    for i in ids:
        rec = BENCH.get(i)
        if rec:
            out.append({"id": i, "title": title_of(rec), "n_parts": rec["n_parts"], "n_faces": rec.get("n_faces"), "spec": rec["spec"]})
    return out


@app.get("/api/mesh/{key}")
def mesh(key: str) -> Response:
    if key not in MESHES:
        raise HTTPException(404, "unknown mesh")
    return Response(MESHES[key], media_type="model/stl")


@app.get("/api/scoreboard")
def scoreboard() -> dict:
    pinned = os.getenv("SCOREBOARD")
    if pinned:
        return json.loads((ROOT / pinned).read_text())
    runs = sorted((ROOT / "runs").glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(runs[-1].read_text()) if runs else {}


@app.post("/api/race")
async def race(req: RaceRequest) -> StreamingResponse:
    if not req.spec.strip():
        raise HTTPException(422, "empty spec")
    gold = BENCH[req.example_id]["gold_code"] if req.example_id in BENCH else None
    lanes = race_lanes()
    queue: asyncio.Queue = asyncio.Queue()
    session = uuid.uuid4().hex[:10]

    async def target() -> None:
        if gold:
            res = await asyncio.to_thread(POOL.run, code=gold, want_mesh=True, want_chamfer=False)
            if res.get("runs"):
                await queue.put({"type": "target", "mesh": store(res["mesh_stl"]), "stats": res["stats"]})

    async def run(lane: Lane) -> None:
        try:
            msgs = prompts.messages(req.spec, [] if lane.dedicated else SHOTS)
            async for event in stream_chat(
                lane,
                msgs,
                max_tokens=2048 if lane.dedicated else 16384,
                temperature=0.0 if lane.dedicated else None,
                session_id=f"race-{session}-{lane.key}",
            ):
                if event["type"] != "done":
                    await queue.put({"lane": lane.key, **event})
                    continue
                result = event["result"]
                await queue.put({"lane": lane.key, "type": "generated", "metrics": result.metrics(), "error": result.error})
                code = None if result.error else prompts.extract_code(result.content)
                if code is None:
                    await queue.put({"lane": lane.key, "type": "graded", "runs": False, "error": result.error or "no code block in the reply"})
                    return
                res = await asyncio.to_thread(POOL.run, code=code, gold_code=gold, want_mesh=True, want_chamfer=bool(gold))
                payload = {
                    "lane": lane.key,
                    "type": "graded",
                    "code": code,
                    "runs": bool(res.get("runs")),
                    "error": res.get("error"),
                    "iou": res.get("iou"),
                    "iou_aligned": res.get("iou_aligned"),
                    "chamfer": res.get("chamfer"),
                    "stats": res.get("stats"),
                }
                if res.get("runs"):
                    payload["mesh"] = store(res.get("aligned_mesh_stl") or res["mesh_stl"])
                await queue.put(payload)
        except Exception as e:  # noqa: BLE001 - keep the other lanes racing
            await queue.put({"lane": lane.key, "type": "error", "error": f"{type(e).__name__}: {e}"})
        finally:
            await queue.put({"lane": lane.key, "type": "_end"})

    async def events():
        tasks = [asyncio.create_task(target())] + [asyncio.create_task(run(l)) for l in lanes]
        remaining = len(lanes)
        while remaining:
            event = await queue.get()
            if event["type"] == "_end":
                remaining -= 1
                continue
            yield f"data: {json.dumps(event)}\n\n"
        await asyncio.gather(*tasks, return_exceptions=True)
        while not queue.empty():
            event = queue.get_nowait()
            if event["type"] != "_end":
                yield f"data: {json.dumps(event)}\n\n"
        yield 'data: {"type": "all_done"}\n\n'

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
