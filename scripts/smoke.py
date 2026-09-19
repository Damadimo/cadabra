"""Pre-flight check (costs well under a cent): the key works, pinned slugs are live, every lane answers.

  uv run python scripts/smoke.py
"""

from __future__ import annotations

import asyncio

import httpx

from understudy.config import ADJUDICATOR_MODEL, BULK_MODEL, DEPRECATED, MODEL_API_BASE_URL, TEACHER_MODEL, api_key, lanes
from understudy.llm import complete


async def main() -> None:
    headers = {"Authorization": f"Bearer {api_key()}"}
    all_lanes = lanes()
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.get(f"{MODEL_API_BASE_URL}/models", headers=headers)
        if r.status_code != 200:
            raise SystemExit(f"GET {MODEL_API_BASE_URL}/models -> {r.status_code}: {r.text[:300]}")
        live = {m["id"] for m in r.json().get("data", [])}
        pinned = {TEACHER_MODEL, ADJUDICATOR_MODEL, BULK_MODEL} | {l.model for l in all_lanes.values() if not l.dedicated}
        print(f"Live Model API catalog: {len(live)} models")
        for slug in sorted(pinned):
            print(f"  {'ok     ' if slug in live else 'MISSING'}  {slug}")
        for slug in sorted(pinned & DEPRECATED):
            print(f"  WARNING  {slug} stops working 2026-09-25 17:00 PT")
        dedicated = next((l for l in all_lanes.values() if l.dedicated), None)
        if dedicated:
            rr = await http.get(f"{dedicated.base_url}/models", headers=headers)
            served = [m["id"] for m in rr.json().get("data", [])] if rr.status_code == 200 else f"HTTP {rr.status_code}"
            print(f"Dedicated deployment serves: {served}  (set UNDERSTUDY_MODEL to the checkpoint name)")

    print("\nOne tiny call per lane:")
    for lane in all_lanes.values():
        res = await complete(lane, [{"role": "user", "content": "Reply with exactly one word: ready"}], max_tokens=512)
        if res.error:
            print(f"  {lane.key:14} ERROR {res.error}")
            continue
        ttft = f"{res.ttft_s:.2f}s" if res.ttft_s is not None else "–"
        cost = f"${res.cost_usd:.6f}" if res.cost_usd is not None else "–"
        print(f"  {lane.key:14} ttft={ttft} e2e={res.e2e_s:.2f}s out_tokens={res.output_tokens} cost={cost} -> {res.content.strip()[:40]!r}")


if __name__ == "__main__":
    asyncio.run(main())
