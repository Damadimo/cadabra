"""Async client for Baseten Model APIs and dedicated deployments.

Every call streams, so one request gives time-to-first-token (TTFT), output tokens/sec and
end-to-end latency, and is priced from the usage the server returns. 429 / 5xx / 529 are
retried with exponential backoff, but only before the first token arrives.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterator

import openai
from openai import AsyncOpenAI

from .config import PRICES, RPM, Lane, api_key

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

_clients: dict[str, AsyncOpenAI] = {}
_limiters: dict[str, RateLimiter] = {}


def client_for(base_url: str) -> AsyncOpenAI:
    if base_url not in _clients:
        _clients[base_url] = AsyncOpenAI(base_url=base_url, api_key=api_key(), max_retries=0, timeout=600)
    return _clients[base_url]


class RateLimiter:
    """Spaces requests evenly to stay under a requests-per-minute budget."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            await asyncio.sleep(start - now)


def _limiter(lane: Lane) -> RateLimiter | None:
    if lane.dedicated:
        return None  # dedicated deployments have no per-model RPM cap
    if lane.model not in _limiters:
        _limiters[lane.model] = RateLimiter(RPM)
    return _limiters[lane.model]


@dataclass
class CallResult:
    lane: str
    model: str
    content: str = ""
    reasoning: str = ""
    ttft_s: float | None = None
    e2e_s: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    usage_estimated: bool = False
    cost_usd: float | None = None
    error: str | None = None

    @property
    def tok_per_s(self) -> float | None:
        if self.ttft_s is None or not self.output_tokens:
            return None
        generating = self.e2e_s - self.ttft_s
        return self.output_tokens / generating if generating > 0 else None

    def metrics(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("content")
        d.pop("reasoning")
        d["tok_per_s"] = self.tok_per_s
        return d


def price(lane: Lane, r: CallResult) -> float | None:
    """List price per token for Model APIs. For a dedicated GPU: the GPU-seconds this request held at
    concurrency 1, an upper bound (evaluate.py reports the amortized figure at real concurrency)."""
    if lane.dedicated:
        return lane.gpu_hourly_usd / 3600 * r.e2e_s
    p = PRICES.get(lane.model)
    if p is None:
        return None
    uncached = max(r.prompt_tokens - r.cached_tokens, 0)
    return (uncached * p.input + r.cached_tokens * p.cached + r.output_tokens * p.output) / 1e6


def _reasoning_delta(delta: Any) -> str | None:
    for name in ("reasoning_content", "reasoning"):
        value = getattr(delta, name, None)
        if isinstance(value, str) and value:
            return value
    return None


async def stream_chat(
    lane: Lane,
    messages: list[dict],
    *,
    schema: dict | None = None,
    max_tokens: int = 4096,
    temperature: float | None = 0.2,
    session_id: str | None = None,
    max_attempts: int = 6,
) -> AsyncIterator[dict]:
    """Yield {"type": "reasoning"|"content", "text": ...} deltas, then {"type": "done", "result": CallResult}."""
    kwargs: dict[str, Any] = {
        "model": lane.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if schema is not None and lane.json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": schema, "strict": True},
        }
    extra_body = {"reasoning_effort": lane.reasoning_effort} if lane.reasoning_effort else None
    extra_headers = {"x-session-affinity": session_id} if session_id else None

    result = CallResult(lane=lane.key, model=lane.model)
    client = client_for(lane.base_url)
    limiter = _limiter(lane)

    for attempt in range(max_attempts):
        if limiter:
            await limiter.wait()
        t0 = time.perf_counter()
        started = False
        try:
            stream = await client.chat.completions.create(**kwargs, extra_body=extra_body, extra_headers=extra_headers)
            async for chunk in stream:
                if chunk.usage:
                    result.prompt_tokens = chunk.usage.prompt_tokens or 0
                    result.output_tokens = chunk.usage.completion_tokens or 0
                    details = getattr(chunk.usage, "prompt_tokens_details", None)
                    result.cached_tokens = getattr(details, "cached_tokens", 0) or 0
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                for kind, text in (("reasoning", _reasoning_delta(delta)), ("content", delta.content)):
                    if not text:
                        continue
                    if result.ttft_s is None:
                        result.ttft_s = time.perf_counter() - t0
                    started = True
                    if kind == "reasoning":
                        result.reasoning += text
                    else:
                        result.content += text
                    yield {"type": kind, "text": text}
            result.e2e_s = time.perf_counter() - t0
            result.error = None
            break
        except (openai.APIStatusError, openai.APIConnectionError) as e:
            status = getattr(e, "status_code", None)
            result.e2e_s = time.perf_counter() - t0
            result.error = f"{status or type(e).__name__}: {str(e)[:300]}"
            if status == 400 and "stream_options" in str(e) and "stream_options" in kwargs:
                kwargs.pop("stream_options")  # server doesn't accept it; usage falls back to an estimate
                continue
            retryable = status is None or status in RETRYABLE_STATUS
            if started or not retryable or attempt == max_attempts - 1:
                break
            retry_after = None
            if isinstance(e, openai.APIStatusError):
                try:
                    retry_after = float(e.response.headers.get("retry-after", ""))
                except ValueError:
                    pass
            await asyncio.sleep(retry_after or min(30.0, 2**attempt + random.random()))

    if not result.output_tokens and (result.content or result.reasoning):
        result.output_tokens = max(1, len(result.content + result.reasoning) // 4)
        result.usage_estimated = True
    result.cost_usd = None if result.error else price(lane, result)
    yield {"type": "done", "result": result}


async def complete(lane: Lane, messages: list[dict], **kwargs: Any) -> CallResult:
    """Collect a streamed call into one CallResult (TTFT is still measured)."""
    result: CallResult | None = None
    async for event in stream_chat(lane, messages, **kwargs):
        if event["type"] == "done":
            result = event["result"]
    assert result is not None
    return result
