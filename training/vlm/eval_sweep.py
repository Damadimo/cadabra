"""Score several saved LoRA adapters on the held-out sheets with the harness bench_eval.py uses.

  SHARDS=4 SHARD=0 python eval_sweep.py        # run_eval_sweep.sh starts one of these per GPU

Every adapter mirrored under $BT_LOAD_CHECKPOINT_DIR is evaluated: the base model is loaded once, each adapter is
attached, graded and unloaded again. Shard i takes every SHARDS-th adapter, so n checkpoints on 4 H100s cost
ceil(n/4) evaluations instead of n. Per adapter it writes $BT_CHECKPOINT_DIR/bench_eval/<name>.json (the shape
bench_eval.py writes) and prints one [sweep] line. Same 500 sheets, same greedy decoding and same geometry checker as
the in-job evals of the SFT and RL runs, so the numbers are directly comparable.
"""

import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from bench_eval import run
from vlm_common import IMAGE_PIXELS, MODEL_ID

SHARD = int(os.getenv("SHARD", "0"))
SHARDS = int(os.getenv("SHARDS", "1"))


def main() -> None:
    root = Path(os.getenv("ADAPTER_ROOT") or os.getenv("BT_LOAD_CHECKPOINT_DIR", "/nonexistent"))
    adapters = sorted({p.parent for p in root.rglob("adapter_config.json")} if root.exists() else [])
    if not adapters:
        raise SystemExit(f"no adapter under {root}")
    mine = adapters[SHARD::SHARDS]
    print(f"[sweep] shard {SHARD}/{SHARDS}: {len(mine)} of {len(adapters)} adapters: {[p.name for p in mine]}", flush=True)

    out_dir = Path(os.getenv("BT_CHECKPOINT_DIR", ".")) / "bench_eval"
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=IMAGE_PIXELS, max_pixels=IMAGE_PIXELS)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu")

    for adapter in mine:
        name = "-".join(adapter.relative_to(root).parts)  # e.g. wd669g3-rank-0-checkpoint-20
        model = PeftModel.from_pretrained(base, str(adapter))
        model.eval()
        summary = run(model, processor, str(out_dir / f"{name}.json"))
        print(f"[sweep] {name} {json.dumps(summary)}", flush=True)
        base = model.unload()  # drop the LoRA deltas, keep the base weights on the GPU for the next adapter


if __name__ == "__main__":
    main()
