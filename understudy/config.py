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

TEACHER_MODEL = os.getenv("TEACHER_MODEL", "zai-org/GLM-5.3")
ADJUDICATOR_MODEL = os.getenv("ADJUDICATOR_MODEL", "moonshotai/Kimi-K3")
BULK_MODEL = os.getenv("BULK_MODEL", "zai-org/GLM-5.3-Flash")

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
    json_schema: bool = True  # send response_format=json_schema
    gpu_hourly_usd: float | None = None  # set for dedicated deployments (billed per GPU-hour)

    @property
    def dedicated(self) -> bool:
        return self.gpu_hourly_usd is not None


def lanes() -> dict[str, Lane]:
    out = {
        "kimi-k3": Lane("kimi-k3", "Kimi K3", "moonshotai/Kimi-K3", reasoning_effort=os.getenv("K3_EFFORT", "low")),
        "glm-5.3": Lane("glm-5.3", "GLM-5.3", "zai-org/GLM-5.3", reasoning_effort=os.getenv("GLM_EFFORT", "low")),
        "glm-5.3-flash": Lane(
            "glm-5.3-flash", "GLM-5.3 Flash", "zai-org/GLM-5.3-Flash", reasoning_effort=os.getenv("FLASH_EFFORT", "low")
        ),
    }
    url = os.getenv("UNDERSTUDY_BASE_URL")
    if url:
        gpu = float(os.getenv("UNDERSTUDY_GPU_HOURLY", "6.50"))
        schema = os.getenv("UNDERSTUDY_JSON_SCHEMA", "1") == "1"
        out["specialist"] = Lane(
            "specialist",
            "Understudy 4B (ours)",
            os.getenv("UNDERSTUDY_MODEL", "checkpoint-final"),
            base_url=url,
            json_schema=schema,
            gpu_hourly_usd=gpu,
        )
        # v0 baseline: the untuned base model on the same vLLM deployment (check the deployment's /v1/models)
        out["base-4b"] = Lane(
            "base-4b",
            "Qwen3-4B (untuned)",
            os.getenv("UNDERSTUDY_BASE_MODEL", "Qwen/Qwen3-4B"),
            base_url=url,
            json_schema=schema,
            gpu_hourly_usd=gpu,
        )
    return out


def race_lanes() -> list[Lane]:
    wanted = os.getenv("RACE_LANES", "specialist,kimi-k3,glm-5.3").split(",")
    available = lanes()
    return [available[k.strip()] for k in wanted if k.strip() in available]
