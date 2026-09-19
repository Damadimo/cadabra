"""Import a training job's in-job benchmark (bench_eval/results.json) as a normal run under runs/.

  uv run python scripts/import_job_eval.py <training_job_id> --tag ours-injob

The training job grades itself on the held-out sheets at the end (training/vlm/bench_eval.py). This turns that file
into runs/<ts>_<tag>/{results.jsonl,summary.json} so the leaderboard, the charts and the demo picker treat it like any
other lane. Latency and cost are left out: the job generates in batches with transformers, so its per-part timing says
nothing about serving. Run the deployed model through cadabra/cad/bench.py for those.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from datetime import datetime
from statistics import mean

from cadabra.config import ROOT
from cadabra.data import write_jsonl
from cadabra.stats import bootstrap_ci


def fetch(job_id: str, name: str) -> dict:
    out = subprocess.run(["baseten", "train", "checkpoint", "files", "--job-id", job_id, "--output", "jsonl"],
                         capture_output=True, text=True, check=True).stdout
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    hits = [r for r in rows if r["relative_file_name"].endswith(name)]
    if not hits:
        raise SystemExit(f"{name} not in job {job_id}: {sorted({r['relative_file_name'] for r in rows})[:10]}")
    with urllib.request.urlopen(hits[0]["url"]) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_id")
    ap.add_argument("--tag", default="ours-injob")
    ap.add_argument("--label", default="Cadabra 4B (ours)")
    ap.add_argument("--model", default="cadabra-vl")
    args = ap.parse_args()
    payload = fetch(args.job_id, "bench_eval/results.json")
    summary, rows = payload["summary"], payload["rows"]
    out_rows = [{
        "id": r["id"], "lane": "specialist", "model": args.model, "n_parts": r["n_parts"], "n_faces": r.get("n_faces"),
        "runs": bool(r["runs"]), "success": bool(r["success"]), "iou": None, "iou_aligned": r.get("iou_aligned"),
        "chamfer": None, "error": r.get("error"), "attempts": 1, "ttft_s": None, "e2e_s": None, "output_tokens": None,
        "reasoning_chars": 0, "cost_usd": None, "code": r.get("code"),
    } for r in rows]
    succ = [1.0 if r["success"] else 0.0 for r in out_rows]
    lo, hi = bootstrap_ci(succ)
    created = datetime.now()
    lane = {
        "lane": "specialist", "label": args.label, "model": args.model, "reasoning_effort": None, "n": len(out_rows),
        "n_infra_failed": 0, "success": mean(succ), "success_ci95": [lo, hi],
        "success_iou95": mean(1.0 if (r["iou_aligned"] or 0) >= 0.95 else 0.0 for r in out_rows),
        "run_rate": mean(1.0 if r["runs"] else 0.0 for r in out_rows),
        "mean_iou_aligned": mean(r["iou_aligned"] or 0.0 for r in out_rows),
        "median_chamfer": None, "latency_p50": None, "latency_p95": None, "ttft_p50": None,
        "single_part": {"n": sum(r["n_parts"] == 1 for r in out_rows), "success": mean(1.0 if r["success"] else 0.0 for r in out_rows if r["n_parts"] == 1)},
        "multi_part": {"n": sum(r["n_parts"] > 1 for r in out_rows), "success": mean(1.0 if r["success"] else 0.0 for r in out_rows if r["n_parts"] > 1)},
        "output_tokens_mean": None, "attempts_mean": 1, "oracle_success": None, "cost_per_1k_usd": None,
        "cost_note": f"in-job eval on the training GPU ({summary.get('generation_s')} s for {summary['n']} parts, batched transformers): not a serving measurement",
        "wall_s": summary.get("generation_s"),
    }
    out_dir = ROOT / "runs" / f"{created:%Y%m%d-%H%M%S}_{args.tag}"
    write_jsonl(out_dir / "results.jsonl", out_rows)
    meta = {"data": "data/cad/bench.jsonl", "n": len(out_rows), "shots": 0, "retries": 0, "modality": "image",
            "best_of": 1, "seed": 0, "created": created.isoformat(timespec="seconds"),
            "source": f"training job {args.job_id} bench_eval/results.json ({summary.get('decoding')})", "lanes": [lane]}
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    print(f"{out_dir.relative_to(ROOT)}: {100 * mean(succ):.1f}% of {len(out_rows)} parts ({summary.get('decoding')})")


if __name__ == "__main__":
    main()
