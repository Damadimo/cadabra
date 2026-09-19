"""Smoke-test a Baseten vLLM VLM deployment: one base64 PNG + text, streamed; prints TTFT, total latency, tok/s.

  set -a; . ./.env; set +a
  CADABRA_BASE_URL=https://model-<model_id>.api.baseten.co/environments/production/sync/v1 \
  CADABRA_MODEL=Qwen/Qwen3-VL-4B-Instruct \
    uv run python deploy/test_vlm.py [image.png] [--dry-run]

Default image: /tmp/img2cad/montage_406.png if present, else a generated 1024x1024 PNG with a rectangle.
--dry-run builds the request and prints its size without calling anything.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

DEFAULT_IMAGE = Path("/tmp/img2cad/montage_406.png")
PROMPT = (
    "Describe the mechanical part shown in this image (overall shape, features, rough proportions), "
    "then write CadQuery Python code that builds it. Put the code in one ```python block."
)


def image_data_url(path: str | None) -> tuple[str, str]:
    src = Path(path) if path else DEFAULT_IMAGE
    if src.exists():
        return "data:image/png;base64," + base64.b64encode(src.read_bytes()).decode(), str(src)
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1024, 1024), "white")
    ImageDraw.Draw(img).rectangle((212, 312, 812, 712), outline="black", width=12)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "generated 1024x1024 rectangle"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    url, src = image_data_url(args[0] if args else None)
    model = os.getenv("CADABRA_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": PROMPT}]}]
    print(f"image: {src} ({len(url) // 1024} KiB as a data URL), model: {model}")
    if "--dry-run" in sys.argv:
        return

    client = OpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=os.environ["CADABRA_BASE_URL"], timeout=900)
    print("served models:", [m.id for m in client.models.list().data])

    t0 = time.perf_counter()
    ttft, parts, usage = None, [], None
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=1024,
        temperature=0,
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(chunk.choices[0].delta.content)
    total = time.perf_counter() - t0

    print("\n" + "".join(parts) + "\n")
    if ttft is None:
        sys.exit(f"no content streamed after {total:.2f}s")
    line = f"TTFT {ttft:.2f}s | total {total:.2f}s"
    if usage:
        line += f" | prompt {usage.prompt_tokens} tok (incl. image) | completion {usage.completion_tokens} tok"
        if total > ttft:
            line += f" | {usage.completion_tokens / (total - ttft):.1f} tok/s after first token"
    print(line)


if __name__ == "__main__":
    main()
