"""Endpoints, pinned model slugs, list prices and race lanes. Everything tunable lives here or in .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL_API_BASE_URL = os.getenv("BASETEN_BASE_URL", "https://inference.baseten.co/v1")


def is_local(url: str) -> bool:
    return url.startswith(("http://127.0.0.1", "http://localhost"))


def api_key() -> str:
    key = os.getenv("BASETEN_API_KEY", "")
    if key:
        return key
    if is_local(MODEL_API_BASE_URL):
        return "mock"
    raise SystemExit("BASETEN_API_KEY is not set. Copy .env.example to .env and add your key.")


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    cached: float
    output: float


# Baseten list prices, snapshot 2026-09-19 (baseten.co/pricing). Re-check before quoting numbers.
PRICES: dict[str, Price] = {
    "zai-org/GLM-5.3": Price(1.40, 0.14, 4.40),
    "zai-org/GLM-5.3-Fast": Price(2.10, 0.21, 6.60),
    "zai-org/GLM-5.3-Flash": Price(0.15, 0.03, 0.50),
    "zai-org/GLM-5.2-Fast": Price(2.10, 0.21, 6.60),
    "moonshotai/Kimi-K3": Price(3.00, 0.30, 15.00),
    "deepseek-ai/DeepSeek-V4.1-Flash": Price(0.30, 0.03, 1.20),
    "deepseek-ai/DeepSeek-V4-Flash-0731": Price(0.13, 0.028, 0.26),
    "deepseek-ai/DeepSeek-V4-Pro-0813": Price(1.32, 0.132, 3.96),
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": Price(0.60, 0.12, 2.40),
    "openai/gpt-oss-120b": Price(0.10, 0.10, 0.50),  # no cache discount
}

# These slugs return errors after 2026-09-25 17:00 PT. Don't build the demo on them.
DEPRECATED = {
    "thinkingmachines/inkling",
    "thinkingmachines/inkling-small",
    "moonshotai/Kimi-K2.6",
    "moonshotai/Kimi-K2.7-Code",
    "zai-org/GLM-4.7",
    "deepseek-ai/DeepSeek-V4-Pro",
}

# Requests/min per model we allow ourselves. Unverified accounts get 15, verified 120.
RPM = int(os.getenv("BASETEN_RPM", "100"))


@dataclass(frozen=True)
class Lane:
    """One model endpoint in the race or the eval."""

    key: str
    label: str
    model: str  # Model API slug, or the served model name on a dedicated deployment
    base_url: str = MODEL_API_BASE_URL
    reasoning_effort: str | None = None
    json_schema: bool = False  # send response_format=json_schema when a caller passes a schema
    gpu_hourly_usd: float | None = None  # set for dedicated deployments (billed per GPU-hour)

    @property
    def dedicated(self) -> bool:
        return self.gpu_hourly_usd is not None


PRETTY = {
    "moonshotai/Kimi-K3": "Kimi K3",
    "zai-org/GLM-5.3": "GLM-5.3",
    "zai-org/GLM-5.3-Flash": "GLM-5.3 Flash",
    "deepseek-ai/DeepSeek-V4-Pro-0813": "DeepSeek V4 Pro",
    "deepseek-ai/DeepSeek-V4.1-Flash": "DeepSeek V4.1 Flash",
    "openai/gpt-oss-120b": "gpt-oss-120b",
}


def lanes() -> dict[str, Lane]:
    """Named lanes. Any Model API slug also works inline as slug:effort (see cad.bench.resolve_lanes)."""
    out = {
        "kimi-k3": Lane("kimi-k3", "Kimi K3", "moonshotai/Kimi-K3", reasoning_effort=os.getenv("K3_EFFORT", "high")),
        "glm-5.3": Lane("glm-5.3", "GLM-5.3", "zai-org/GLM-5.3", reasoning_effort=os.getenv("GLM_EFFORT", "high")),
    }
    url = os.getenv("UNDERSTUDY_BASE_URL")
    if url:
        gpu = float(os.getenv("UNDERSTUDY_GPU_HOURLY", "6.50"))
        out["specialist"] = Lane(
            "specialist", "Understudy-CAD 4B (ours)", os.getenv("UNDERSTUDY_MODEL", "checkpoint-final"), base_url=url, gpu_hourly_usd=gpu
        )
        # v0 baseline: the untuned base model on the same vLLM deployment (check the deployment's /v1/models)
        out["base-4b"] = Lane(
            "base-4b", "Qwen3-4B-Instruct (untuned)", os.getenv("UNDERSTUDY_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"), base_url=url, gpu_hourly_usd=gpu
        )
    return out
