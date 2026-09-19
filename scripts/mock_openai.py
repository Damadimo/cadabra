"""Local stand-in for Baseten's OpenAI-compatible API, to build and rehearse the UI without spending credits.

  uv run python scripts/mock_openai.py          # serves http://127.0.0.1:8001/v1
  # then in .env: BASETEN_BASE_URL=http://127.0.0.1:8001/v1 and UNDERSTUDY_BASE_URL=http://127.0.0.1:8001/v1

Answers with the sample reference label when the input matches a sample, otherwise the task's example.
The speeds are invented. Never quote numbers from the mock.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from understudy.config import ROOT
from understudy.data import read_jsonl
from understudy.tasks import load_task

TASK = load_task()
app = FastAPI()

# model-name fragment -> (seconds to first token, output tokens/sec, reasoning tokens before the answer)
PROFILES = {"Kimi-K3": (0.9, 110, 240), "GLM-5.3-Flash": (0.4, 130, 80), "GLM-5.3": (0.7, 125, 160)}
DEDICATED = (0.12, 380, 0)

REFERENCE = {}
for path in (ROOT / "data" / "samples").glob(f"{TASK.name}_gold*.jsonl"):
    for row in read_jsonl(path):
        text = row.get("text") or "\n".join(row.get("lines", []))
        REFERENCE[text.strip()] = row["label"]


def profile(model: str) -> tuple:
    return next((v for k, v in PROFILES.items() if k in model), DEDICATED)


def answer(messages: list[dict]) -> str:
    user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    label = next((lab for text, lab in REFERENCE.items() if text and text[:200] in user), TASK.example)
    return json.dumps(label)


@app.get("/v1/models")
def models() -> dict:
    names = ["moonshotai/Kimi-K3", "zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash", "checkpoint-final", "Qwen/Qwen3-4B"]
    return {"object": "list", "data": [{"id": n, "object": "model"} for n in names]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model", "")
    ttft, tps, reasoning_tokens = profile(model)
    content = answer(body.get("messages", []))
    prompt_tokens = sum(len(m.get("content") or "") for m in body.get("messages", [])) // 4
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": reasoning_tokens + len(content) // 4,
        "total_tokens": prompt_tokens + reasoning_tokens + len(content) // 4,
        "prompt_tokens_details": {"cached_tokens": prompt_tokens // 2},
    }
    if not body.get("stream"):
        await asyncio.sleep(ttft + usage["completion_tokens"] / tps)
        return JSONResponse(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": usage,
            }
        )

    async def events():
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]

        def chunk(delta: dict | None, finish: str | None = None, with_usage: bool = False) -> str:
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if with_usage:
                payload["usage"] = usage
            return f"data: {json.dumps(payload)}\n\n"

        await asyncio.sleep(ttft)
        for _ in range(reasoning_tokens // 4):
            yield chunk({"reasoning_content": "weighing… "})
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
