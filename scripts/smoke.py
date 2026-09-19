"""Pre-flight check (costs well under a cent): key works, benchmark slugs are live, each lane builds one part.

  uv run python scripts/smoke.py
  UNDERSTUDY_BASE_URL=... UNDERSTUDY_MODEL=checkpoint-... uv run python scripts/smoke.py   # also checks our deployment
"""

from __future__ import annotations

import asyncio

import httpx

from understudy.cad import prompts
from understudy.cad.bench import resolve_lanes
from understudy.cad.pool import CadPool
from understudy.config import DEPRECATED, MODEL_API_BASE_URL, ROOT, api_key, lanes
from understudy.data import read_jsonl
from understudy.llm import complete

BENCH_SLUGS = ["moonshotai/Kimi-K3", "zai-org/GLM-5.3"]


async def main() -> None:
    headers = {"Authorization": f"Bearer {api_key()}"}
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.get(f"{MODEL_API_BASE_URL}/models", headers=headers)
        if r.status_code != 200:
            raise SystemExit(f"GET {MODEL_API_BASE_URL}/models -> {r.status_code}: {r.text[:300]}")
        live = {m["id"] for m in r.json().get("data", [])}
        print(f"Live Model API catalog: {len(live)} models")
        for slug in BENCH_SLUGS:
            print(f"  {'ok     ' if slug in live else 'MISSING'}  {slug}{'  (deprecated Sep 25!)' if slug in DEPRECATED else ''}")
        limit = r.headers.get("x-ratelimit-limit-requests")
        if limit:
            print(f"  rate limit: {limit} requests/min per model (15 = unverified account; verify for 120)")
        ours = lanes().get("specialist")
        if ours:
            rr = await http.get(f"{ours.base_url}/models", headers=headers)
            served = [m["id"] for m in rr.json().get("data", [])] if rr.status_code == 200 else f"HTTP {rr.status_code}"
            print(f"Our deployment serves: {served}  (UNDERSTUDY_MODEL={ours.model})")

    spec = read_jsonl(ROOT / "data" / "cad" / "bench.jsonl")[0]
    todo = [l for l in (lanes().get("specialist"),) if l] + resolve_lanes(",".join(f"{s}:low" for s in BENCH_SLUGS))
    print("\nOne held-out spec per lane:")
    with CadPool(workers=2) as pool:
        for lane in todo:
            res = await complete(lane, prompts.messages(spec["spec"]), max_tokens=8192, temperature=0.0 if lane.dedicated else None)
            if res.error:
                print(f"  {lane.key:18} API ERROR {res.error}")
                continue
            code = prompts.extract_code(res.content)
            graded = pool.run(code=code, gold_code=spec["gold_code"], want_chamfer=False) if code else {"runs": False, "error": "no code"}
            verdict = f"IoU {graded['iou_aligned']:.3f}" if graded.get("runs") else f"failed: {graded.get('error')}"
            print(f"  {lane.key:18} {res.e2e_s:5.1f}s  {res.output_tokens:5d} tokens  {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
