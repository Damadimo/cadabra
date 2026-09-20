"""Spec-to-CAD race: every lane writes CadQuery for the same spec; the server builds each part and scores it.

  uv run uvicorn app.server:app --port 8000        # then open http://127.0.0.1:8000

Lanes: RACE_LANES (default "specialist,moonshotai/Kimi-K3:high,zai-org/GLM-5.3-Flash:high"). Frontier lanes get the
same worked examples as in the benchmark; our model gets the zero-shot prompt it was trained on.
If a Model API call fails for account reasons (402/401/403) on a held-out part, that lane shows the same model's answer
from the benchmark run instead (FALLBACK_RUNS, default "sota-img,sota-img-rest"), labeled as such.
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

from cadabra.cad import prompts
from cadabra.cad.bench import FATAL, infra_failed, resolve_lanes
from cadabra.cad.render import image_path
from cadabra.cad.pool import CadPool
from cadabra.config import PRETTY, ROOT, Lane
from cadabra.data import read_jsonl
from cadabra.llm import complete, stream_chat

STATIC = Path(__file__).parent / "static"
DATA = ROOT / "data" / "cad"
BENCH = {r["id"]: r for r in read_jsonl(DATA / "bench.jsonl")}
SHOTS = read_jsonl(DATA / "shots.jsonl")
IMAGES = DATA / "img"
SHEETS = json.loads((IMAGES / "index.json").read_text()) if (IMAGES / "index.json").exists() else {}


def image_shots() -> list[dict]:
    return [
        {"image": prompts.data_url(image_path(s["id"], IMAGES)), "bbox": SHEETS[s["id"]]["bbox"], "gold_code": s["gold_code"]}
        for s in SHOTS
        if s["id"] in SHEETS
    ]


def saved_answers() -> dict[tuple[str, str], dict]:
    """(lane, part) -> that lane's scored answer in the newest benchmark run of each FALLBACK_RUNS tag."""
    out = {}
    for tag in filter(None, os.getenv("FALLBACK_RUNS", "sota-img,sota-img-rest").split(",")):
        found = sorted((ROOT / "runs").glob(f"*_{tag}"), key=lambda p: p.name)
        if found:
            for r in read_jsonl(found[-1] / "results.jsonl"):
                if not infra_failed(r):
                    out[(r["lane"], r["id"])] = {**r, "_run": found[-1].name}
    return out


SAVED = saved_answers()


def compare_runs() -> dict[str, list[dict]]:
    """part id -> one entry per model, from saved benchmark runs: no API call, so the demo works offline.
    COMPARE_RUNS lists run tags, newest of each; a lane's newest run wins."""
    out: dict[str, dict[tuple, dict]] = {}
    for tag in filter(None, os.getenv("COMPARE_RUNS", "sota-img,sota-img-rest,ours2-img,ours2-img-bo8").split(",")):
        found = sorted((ROOT / "runs").glob(f"*_{tag}"), key=lambda p: p.name)
        if not found:
            continue
        meta = json.loads((found[-1] / "summary.json").read_text())
        best_of = meta.get("best_of", 1)
        labels = {l["lane"]: l for l in meta["lanes"]}
        for r in read_jsonl(found[-1] / "results.jsonl"):
            if infra_failed(r) or r["lane"] in SKIP_LANES:
                continue
            lane = labels.get(r["lane"], {})
            effort = lane.get("reasoning_effort")
            label = lane.get("label") if lane.get("label") not in (None, r["lane"]) else PRETTY.get(r["model"], r["lane"]) + (f" ({effort})" if effort else "")
            out.setdefault(r["id"], {})[(r["lane"], best_of)] = {
                "key": f"{r['lane']}#{best_of}", "lane": r["lane"], "best_of": best_of, "label": label + (f" · best of {best_of}" if best_of > 1 else " · 1 sample" if r["lane"] == "specialist" else ""),
                "model": r["model"], "ours": r["lane"] == "specialist", "run": found[-1].name,
                "success": bool(r["success"]), "runs": bool(r["runs"]), "iou_aligned": r.get("iou_aligned"),
                "e2e_s": r.get("e2e_s"), "cost_usd": r.get("cost_usd"), "output_tokens": r.get("output_tokens"),
                "error": r.get("error"), "code": r.get("code"),
            }
    return {i: sorted(v.values(), key=lambda e: (0 if e["ours"] else 1, -e["best_of"], e["label"])) for i, v in out.items()}


SKIP_LANES = {"glm-5.3@high", "base-4b"}  # GLM-5.3's model card is text-only; the untuned base is in the scoreboard
COMPARE = compare_runs()
_COMPARE_CACHE: OrderedDict[str, dict] = OrderedDict()
MESHES: OrderedDict[str, bytes] = OrderedDict()
POOL: CadPool | None = None
REPLAYS = ROOT / "data" / "demo" / "replays"  # recorded races: offline fallback for the live demo


def race_lanes() -> list[Lane]:
    return resolve_lanes(os.getenv("RACE_LANES", "specialist,moonshotai/Kimi-K3:high,zai-org/GLM-5.3-Flash:high"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global POOL
    POOL = await asyncio.to_thread(CadPool, int(os.getenv("CAD_WORKERS", "4")), 30.0)
    yield
    POOL.close()


app = FastAPI(title="Cadabra", lifespan=lifespan)
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
    spec: str = ""
    example_id: str | None = None
    modality: str = "text"  # "text": the written spec; "image": the rendered drawing sheet
    record: bool = False  # save the event stream + meshes under data/demo/replays/ for offline replay


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/how-it-works")
def how_it_works() -> FileResponse:
    return FileResponse(STATIC / "how-it-works.html")


@app.get("/api/config")
def config() -> dict:
    return {
        "lanes": [{"key": l.key, "label": l.label, "model": l.model, "effort": l.reasoning_effort, "dedicated": l.dedicated} for l in race_lanes()],
        "default_modality": os.getenv("DEFAULT_MODALITY", "image"),
    }


@app.get("/api/examples")
def examples() -> list[dict]:
    """Held-out benchmark specs for the dropdown: data/cad/demo.json if present, else a spread by complexity."""
    demo = DATA / "demo.json"
    ids = json.loads(demo.read_text()) if demo.exists() else None
    if not ids:  # 4 simple, 4 medium, 4 complex or multi-part, spread across each tier
        tiers = (
            [r for r in BENCH.values() if r.get("n_faces", 0) <= 6 and r["n_parts"] == 1],
            [r for r in BENCH.values() if 7 <= r.get("n_faces", 0) <= 12 and r["n_parts"] == 1],
            [r for r in BENCH.values() if r.get("n_faces", 0) >= 13 or r["n_parts"] > 1],
        )
        ids = []
        for tier in tiers:
            tier = sorted(tier, key=lambda r: (r.get("n_faces", 0), r["id"]))
            ids += [tier[int(i * (len(tier) - 1) / 3)]["id"] for i in range(4)] if tier else []
    out = []
    for i in ids:
        rec = BENCH.get(i)
        if rec:
            out.append({"id": i, "title": title_of(rec), "n_parts": rec["n_parts"], "n_faces": rec.get("n_faces"), "spec": rec["spec"],
                        "sheet": f"/api/sheet/{i}" if i in SHEETS else None, "bbox": SHEETS.get(i, {}).get("bbox")})
    return out


@app.get("/api/parts")
def parts() -> list[dict]:
    """Held-out parts every compared model answered, grouped by complexity."""
    def tier(rec):
        if rec["n_parts"] > 1:
            return "multi-part"
        n = rec.get("n_faces") or 0
        return "simple" if n <= 6 else "medium" if n <= 12 else "complex"

    out = []
    for i, lanes in COMPARE.items():
        rec = BENCH.get(i)
        if not rec or i not in SHEETS or len(lanes) < 2:
            continue
        ours = next((l for l in lanes if l["ours"]), None)
        out.append({"id": i, "title": title_of(rec), "tier": tier(rec), "n_faces": rec.get("n_faces"), "n_parts": rec["n_parts"],
                    "extents": rec.get("extents"),  # the demo opens on a part that is not a sliver
                    "models": len(lanes), "ours_correct": bool(ours and ours["success"]),
                    "frontier_correct": sum(1 for l in lanes if not l["ours"] and l["success"]),
                    "results": [{"key": l["key"], "label": l["label"], "ours": l["ours"], "success": l["success"],
                                 "iou": l.get("iou_aligned")} for l in lanes]})
    return sorted(out, key=lambda r: (["simple", "medium", "complex", "multi-part"].index(r["tier"]), -(r["n_faces"] or 0), r["id"]))


@app.get("/api/compare/{rec_id}")
async def compare(rec_id: str) -> dict:
    """The input sheet, the reference solid, and each model's saved answer built into a solid."""
    if rec_id not in COMPARE or rec_id not in BENCH:
        raise HTTPException(404, "no saved answers for this part")
    if rec_id in _COMPARE_CACHE and all(k in MESHES for k in _COMPARE_CACHE[rec_id]["meshes"]):
        return _COMPARE_CACHE[rec_id]["payload"]
    rec = BENCH[rec_id]
    gold = rec["gold_code"]
    target = await asyncio.to_thread(POOL.run, code=gold, want_mesh=True, want_chamfer=False)
    lanes, keys = [], []
    for entry in COMPARE[rec_id]:
        out = {**entry}
        if entry.get("code") and entry["runs"]:
            res = await asyncio.to_thread(POOL.run, code=entry["code"], gold_code=gold, want_mesh=True, want_chamfer=False)
            if res.get("runs"):
                out["mesh"] = store(res.get("aligned_mesh_stl") or res["mesh_stl"])
                out["stats"] = res.get("stats")
                keys.append(out["mesh"])
            else:
                out["error"] = res.get("error")
        lanes.append(out)
    payload = {
        "id": rec_id, "title": title_of(rec), "tier_faces": rec.get("n_faces"), "n_parts": rec["n_parts"],
        "sheet": f"/api/sheet/{rec_id}", "bbox": SHEETS.get(rec_id, {}).get("bbox"), "spec": rec["spec"],
        "reference": {"label": "Reference (dataset)", "mesh": store(target["mesh_stl"]) if target.get("runs") else None,
                      "stats": target.get("stats"), "code": gold},
        "lanes": lanes,
    }
    keys.append(payload["reference"]["mesh"])
    _COMPARE_CACHE[rec_id] = {"payload": payload, "meshes": [k for k in keys if k]}
    while len(_COMPARE_CACHE) > 40:
        _COMPARE_CACHE.popitem(last=False)
    return payload


@app.get("/api/sheet/{rec_id}")
def sheet(rec_id: str) -> FileResponse:
    path = image_path(rec_id, IMAGES)
    if rec_id not in SHEETS or not path.exists():
        raise HTTPException(404, "no drawing sheet for this part")
    return FileResponse(path, media_type="image/png")


@app.get("/api/mesh/{key}")
def mesh(key: str) -> Response:
    if key not in MESHES:
        saved = REPLAYS / "meshes" / f"{key}.stl"
        if not re.fullmatch(r"[0-9a-f]{16}", key) or not saved.exists():
            raise HTTPException(404, "unknown mesh")
        return Response(saved.read_bytes(), media_type="model/stl")
    return Response(MESHES[key], media_type="model/stl")


@app.get("/api/replays")
def replays() -> list[str]:
    return sorted(p.stem for p in REPLAYS.glob("*.json"))


@app.get("/api/replay/{name}")
def replay(name: str) -> dict:
    path = REPLAYS / f"{name}.json"
    if not re.fullmatch(r"[\w.\-]+", name) or not path.exists():
        raise HTTPException(404, "unknown replay")
    return json.loads(path.read_text())


@app.get("/api/scoreboard")
def scoreboard() -> dict:
    pinned = os.getenv("SCOREBOARD") or ("data/demo/scoreboard.json" if (ROOT / "data/demo/scoreboard.json").exists() else None)
    if pinned:
        return json.loads((ROOT / pinned).read_text())
    runs = sorted((ROOT / "runs").glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(runs[-1].read_text()) if runs else {}


def save_replay(req: RaceRequest, lanes: list[Lane], log: list[dict]) -> None:
    (REPLAYS / "meshes").mkdir(parents=True, exist_ok=True)
    for event in log:
        key = event.get("mesh")
        if key and key in MESHES:
            (REPLAYS / "meshes" / f"{key}.stl").write_bytes(MESHES[key])
    name = f"{(req.example_id or 'custom').replace(':', '_')}-{req.modality}"
    payload = {
        "example_id": req.example_id,
        "modality": req.modality,
        "spec": req.spec,
        "lanes": [{"key": l.key, "label": l.label, "model": l.model, "effort": l.reasoning_effort, "dedicated": l.dedicated} for l in lanes],
        "events": log,
    }
    (REPLAYS / f"{name}.json").write_text(json.dumps(payload))


@app.post("/api/race")
async def race(req: RaceRequest) -> StreamingResponse:
    image_mode = req.modality == "image"
    if image_mode and req.example_id not in SHEETS:
        raise HTTPException(422, "drawing-sheet mode needs a held-out example with a rendered sheet")
    if not image_mode and not req.spec.strip():
        raise HTTPException(422, "empty spec")
    gold = BENCH[req.example_id]["gold_code"] if req.example_id in BENCH else None
    sheet_url = prompts.data_url(image_path(req.example_id, IMAGES)) if image_mode else None
    lanes = race_lanes()
    queue: asyncio.Queue = asyncio.Queue()
    session = uuid.uuid4().hex[:10]

    async def target() -> None:
        if gold:
            res = await asyncio.to_thread(POOL.run, code=gold, want_mesh=True, want_chamfer=False)
            if res.get("runs"):
                await queue.put({"type": "target", "mesh": store(res["mesh_stl"]), "stats": res["stats"]})

    best_of = int(os.getenv("OURS_BEST_OF", "8"))

    async def run_best_of(lane: Lane, msgs: list[dict]) -> None:
        """Our lane on a drawing sheet: sample N programs, keep the one whose render matches the sheet best."""
        await queue.put({"lane": lane.key, "type": "status", "text": f"sampling {best_of} programs…"})
        calls = await asyncio.gather(*(complete(lane, msgs, max_tokens=2048, temperature=0.7, session_id=f"race-{session}-{lane.key}") for _ in range(best_of)))
        ok = [c for c in calls if not c.error]
        first = min(ok, key=lambda c: c.ttft_s or 1e9) if ok else calls[0]
        metrics = {**first.metrics(), "e2e_s": max(c.e2e_s for c in calls), "output_tokens": sum(c.output_tokens for c in calls),
                   "cost_usd": sum(c.cost_usd or 0.0 for c in calls)}
        await queue.put({"lane": lane.key, "type": "generated", "metrics": metrics, "error": None if ok else calls[0].error})
        codes = [prompts.extract_code(c.content) for c in ok]
        codes = [c for c in codes if c]
        if not codes:
            await queue.put({"lane": lane.key, "type": "graded", "runs": False, "error": "no candidate produced code"})
            return
        await queue.put({"lane": lane.key, "type": "status", "text": f"checking {len(codes)} candidates against the drawing…"})
        sheet_file = str(image_path(req.example_id, IMAGES))
        bbox = SHEETS[req.example_id]["bbox"]
        checks = await asyncio.gather(*(asyncio.to_thread(POOL.run, _fn="render_score", code=c, sheet_path=sheet_file, bbox=bbox) for c in codes))
        pick = max(range(len(codes)), key=lambda k: checks[k].get("render_score") or 0.0)
        res = await asyncio.to_thread(POOL.run, code=codes[pick], gold_code=gold, want_mesh=True, want_chamfer=bool(gold))
        payload = {"lane": lane.key, "type": "graded", "code": codes[pick], "runs": bool(res.get("runs")), "error": res.get("error"),
                   "iou": res.get("iou"), "iou_aligned": res.get("iou_aligned"), "chamfer": res.get("chamfer"), "stats": res.get("stats"),
                   "render_score": checks[pick].get("render_score"), "candidates": len(codes)}
        if res.get("runs"):
            payload["mesh"] = store(res.get("aligned_mesh_stl") or res["mesh_stl"])
        await queue.put(payload)

    async def replay_saved(lane: Lane, row: dict) -> None:
        """The API refused the call (credits/auth), not the model: show this model's answer from the benchmark run."""
        await queue.put({"lane": lane.key, "type": "status", "text": f"API unavailable: showing this model's answer from benchmark run {row['_run']}"})
        if row.get("code"):
            await queue.put({"lane": lane.key, "type": "content", "text": row["code"]})
        metrics = {"ttft_s": row.get("ttft_s"), "e2e_s": row.get("e2e_s"), "output_tokens": row.get("output_tokens"), "cost_usd": row.get("cost_usd")}
        await queue.put({"lane": lane.key, "type": "generated", "metrics": metrics, "error": None})
        if not (row.get("runs") and row.get("code")):
            await queue.put({"lane": lane.key, "type": "graded", "runs": False, "code": row.get("code"), "error": row.get("error") or "no solid"})
            return
        res = await asyncio.to_thread(POOL.run, code=row["code"], gold_code=gold, want_mesh=True, want_chamfer=bool(gold))
        payload = {"lane": lane.key, "type": "graded", "code": row["code"], "runs": bool(res.get("runs")), "error": res.get("error"),
                   "iou": res.get("iou"), "iou_aligned": res.get("iou_aligned"), "chamfer": res.get("chamfer"), "stats": res.get("stats")}
        if res.get("runs"):
            payload["mesh"] = store(res.get("aligned_mesh_stl") or res["mesh_stl"])
        await queue.put(payload)

    async def run(lane: Lane) -> None:
        try:
            if image_mode and lane.dedicated and best_of > 1:
                await run_best_of(lane, prompts.image_messages(sheet_url, SHEETS[req.example_id]["bbox"], []))
                return
            if image_mode:
                msgs = prompts.image_messages(sheet_url, SHEETS[req.example_id]["bbox"], [] if lane.dedicated else image_shots())
            else:
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
                saved = SAVED.get((lane.key, req.example_id))
                if result.error and any(code in result.error for code in FATAL) and saved:
                    await replay_saved(lane, saved)
                    return
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

    started = asyncio.get_running_loop().time()
    log: list[dict] = []

    def emit(event: dict) -> str:
        if req.record:
            log.append({"t": round(asyncio.get_running_loop().time() - started, 3), **event})
        return f"data: {json.dumps(event)}\n\n"

    async def events():
        tasks = [asyncio.create_task(target())] + [asyncio.create_task(run(l)) for l in lanes]
        remaining = len(lanes)
        while remaining:
            event = await queue.get()
            if event["type"] == "_end":
                remaining -= 1
                continue
            yield emit(event)
        await asyncio.gather(*tasks, return_exceptions=True)
        while not queue.empty():
            event = queue.get_nowait()
            if event["type"] != "_end":
                yield emit(event)
        yield emit({"type": "all_done"})
        if req.record:
            save_replay(req, lanes, log)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
