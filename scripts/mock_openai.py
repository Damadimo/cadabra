"""Local stand-in for Baseten's OpenAI-compatible API, to build and rehearse the demo without spending credits.

  uv run python scripts/mock_openai.py          # serves http://127.0.0.1:8001/v1
  BASETEN_BASE_URL=http://127.0.0.1:8001/v1 UNDERSTUDY_BASE_URL=http://127.0.0.1:8001/v1 \
    uv run uvicorn app.server:app --port 8000

For held-out specs it answers with the reference CadQuery code; "frontier" model names sometimes get a copy with
the sketch scaled wrongly, so every UI state (correct, wrong part, crash) shows up. Speeds and answers are fake:
never quote numbers from the mock.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from understudy.config import ROOT
from understudy.data import read_jsonl

app = FastAPI()
BENCH = read_jsonl(ROOT / "data" / "cad" / "bench.jsonl")
REFERENCE = {r["spec"].strip(): r["gold_code"] for r in BENCH}
_INDEX = ROOT / "data" / "cad" / "img" / "index.json"
BY_BBOX = {}
if _INDEX.exists():
    boxes = json.loads(_INDEX.read_text())
    BY_BBOX = {tuple(f"{v:.4f}" for v in boxes[r["id"]]["bbox"]): r["gold_code"] for r in BENCH if r["id"] in boxes}
FALLBACK = "import cadquery as cq\n\nr = cq.Workplane('XY').box(0.6, 0.4, 0.1)\n"
# model-name fragment -> (seconds to first token, output tokens/sec, reasoning tokens, chance of a wrong answer)
PROFILES = {"Kimi-K3": (0.9, 110, 400, 0.15), "GLM-5.3": (0.8, 120, 300, 0.35)}
DEDICATED = (0.12, 350, 0, 0.0)


def profile(model: str) -> tuple:
    return next((v for k, v in PROFILES.items() if k in model), DEDICATED)


def _text(content) -> str:
    if isinstance(content, list):  # image requests: take the text part (it carries the bounding box)
        return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content or ""


def answer(messages: list[dict], wrong_rate: float) -> str:
    spec = next((_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "").strip()
    code = REFERENCE.get(spec)
    if code is None:
        found = re.search(r"X (\d+\.\d+), Y (\d+\.\d+), Z (\d+\.\d+)", spec)
        code = BY_BBOX.get(found.groups(), FALLBACK) if found else FALLBACK
    roll = random.random()
    if roll < wrong_rate / 2:
        code = code.replace("extrude(", "extrude(2 * ")  # builds, wrong part
    elif roll < wrong_rate:
        code = code.replace("cq.Workplane", "cq.Workplan")  # crashes
    return f"```python\n{code.strip()}\n```"


@app.get("/v1/models")
def models() -> dict:
    names = ["moonshotai/Kimi-K3", "zai-org/GLM-5.3", "checkpoint-final", "Qwen/Qwen3-4B-Instruct-2507"]
    return {"object": "list", "data": [{"id": n, "object": "model"} for n in names]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model", "")
    ttft, tps, reasoning_tokens, wrong_rate = profile(model)
    content = answer(body.get("messages", []), wrong_rate)
    prompt_tokens = sum(len(_text(m.get("content"))) for m in body.get("messages", [])) // 4
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": reasoning_tokens + len(content) // 4,
        "total_tokens": prompt_tokens + reasoning_tokens + len(content) // 4,
        "prompt_tokens_details": {"cached_tokens": prompt_tokens // 2},
    }
    if not body.get("stream"):
        await asyncio.sleep(ttft + usage["completion_tokens"] / tps)
        choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        return JSONResponse({"id": "mock", "object": "chat.completion", "created": int(time.time()), "model": model, "choices": [choice], "usage": usage})

    async def events():
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]

        def chunk(delta: dict | None, finish: str | None = None, with_usage: bool = False) -> str:
            payload = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                       "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if with_usage:
                payload["usage"] = usage
            return f"data: {json.dumps(payload)}\n\n"

        await asyncio.sleep(ttft)
        for _ in range(reasoning_tokens // 4):
            yield chunk({"reasoning_content": "checking the sketch… "})
            await asyncio.sleep(4 / tps)
        for i in range(0, len(content), 16):
            yield chunk({"content": content[i : i + 16]})
            await asyncio.sleep(4 / tps)
        yield chunk({}, finish="stop")
        yield chunk(None, with_usage=True)
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("MOCK_PORT", "8001")), log_level="warning")
