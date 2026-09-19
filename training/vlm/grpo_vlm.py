"""Stage 2 (optional): GRPO for the drawing-sheet model with the geometry reward. Runs inside a Baseten Training job.

Continues the stage-1 (SFT) LoRA adapter, found under $SFT_ADAPTER or $BT_LOAD_CHECKPOINT_DIR. Each sheet is sampled
NUM_GENERATIONS times; every completion's code is executed by the sandboxed checker in cadcheck/ and rewarded with its
aligned IoU against the reference solid, plus a bonus for clearing the success bar (IoU >= 0.9). Code that crashes
scores -0.2, a reply without a code block -0.5. Rollouts use plain transformers generation (no vLLM in the image), so
time a few steps first. The result is the same kind of adapter as stage 1, plus merged weights for deploy_vlm.sh.

Reads data/grpo.jsonl: {"images": ["images/<id>.png"], "prompt": [system, user(image + bounding box)], "gold_code": ...}.
"""

import json
import os
import re
from pathlib import Path

import torch
from datasets import Dataset, Image, List
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from cadcheck.pool import CadPool
from vlm_common import CPU_DRY_RUN, DATA_DIR, IMAGE_PIXELS, MODEL_ID, as_parts, image_path, save_merged

OUTPUT_DIR = os.getenv("BT_CHECKPOINT_DIR", "./checkpoints")
MERGE_AT_END = os.getenv("MERGE_AT_END", "0" if CPU_DRY_RUN else "1") == "1"
SUCCESS_BONUS = float(os.getenv("SUCCESS_BONUS", "0.5"))
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
LM_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def find_adapter() -> str | None:
    """Explicit SFT_ADAPTER path, else the newest adapter under $BT_LOAD_CHECKPOINT_DIR (merged/ has none)."""
    explicit = os.getenv("SFT_ADAPTER")
    if explicit is not None:
        return explicit or None
    root = Path(os.getenv("BT_LOAD_CHECKPOINT_DIR", "/nonexistent"))
    configs = sorted(root.rglob("adapter_config.json"), key=lambda p: p.stat().st_mtime) if root.exists() else []
    if not configs:
        return None
    steps = [(int(m.group(1)) if (m := re.search(r"checkpoint-(\d+)", str(p))) else 10**9, p) for p in configs]
    return str(max(steps)[1].parent)


def extract_code(text: str) -> str | None:
    blocks = [b for b in FENCE.findall(text) if "cq" in b or "cadquery" in b]
    if blocks:
        return blocks[-1].strip() + "\n"
    return text[text.index("import cadquery"):] if "import cadquery" in text else None


_pool: CadPool | None = None
_cache: dict = {}


def score(pairs: list[tuple[str, str]]) -> list[dict]:
    global _pool
    if _pool is None:
        _pool = CadPool(workers=int(os.getenv("REWARD_WORKERS", max(2, (os.cpu_count() or 4) - 2))), timeout=20)
    todo = list(dict.fromkeys(p for p in pairs if p not in _cache))
    for key, res in zip(todo, _pool.map([{"code": c, "gold_code": g, "want_chamfer": False} for c, g in todo])):
        _cache[key] = res
    if len(_cache) > 20000:
        _cache.clear()
    return [_cache.get(k, {"runs": False}) for k in pairs]


def _texts(completions) -> list[str]:
    out = []
    for c in completions:
        content = c[-1].get("content") or "" if isinstance(c, list) else c
        out.append("".join(p.get("text") or "" for p in content) if isinstance(content, list) else content)
    return out


def geometry_reward(prompts, completions, gold_code, **kwargs) -> list[float]:
    codes = [extract_code(t) for t in _texts(completions)]
    results = iter(score([(c, g) for c, g in zip(codes, gold_code) if c is not None]))
    rewards = []
    for code in codes:
        if code is None:
            rewards.append(-0.5)
            continue
        res = next(results)
        iou = float(res.get("iou_aligned") or 0.0) if res.get("runs") else None
        rewards.append(-0.2 if iou is None else iou + (SUCCESS_BONUS if iou >= 0.9 else 0.0))
    return rewards


def success_metric(prompts, completions, gold_code, **kwargs) -> list[float]:
    """Logged only (weight 0): share of rollouts that are correct parts (code runs, aligned IoU >= 0.9)."""
    codes = [extract_code(t) for t in _texts(completions)]
    out = []
    for code, gold in zip(codes, gold_code):
        res = _cache.get((code, gold), {}) if code else {}
        out.append(1.0 if res.get("runs") and (res.get("iou_aligned") or 0) >= 0.9 else 0.0)
    return out


class SheetGRPOTrainer(GRPOTrainer):
    """Rollouts in eval mode. With gradient checkpointing on, transformers turns the KV cache off for any layer in
    training mode (GradientCheckpointingLayer), and TRL generates before its own checkpointing-off block, so plain
    generate() would recompute the whole sequence for every new token. Eval mode keeps the cache; the gradient pass
    after it still runs checkpointed."""

    def _generate(self, prompts):
        was_training = self.model.training
        self.model.eval()
        try:
            return super()._generate(prompts)
        finally:
            if was_training:
                self.model.train()


def load_dataset_rows() -> Dataset:
    with open(os.path.join(DATA_DIR, "grpo.jsonl")) as f:
        rows = [
            {"images": [image_path(p) for p in row["images"]], "prompt": [as_parts(m) for m in row["prompt"]], "gold_code": row["gold_code"]}
            for row in map(json.loads, filter(str.strip, f))
        ]
    print(f"grpo: {len(rows)} prompts")
    return Dataset.from_list(rows).cast_column("images", List(Image(mode="RGB")))


def main() -> None:
    dataset = load_dataset_rows()
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=IMAGE_PIXELS, max_pixels=IMAGE_PIXELS, padding_side="left")
    world, local_rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("LOCAL_RANK", "0"))  # torchrun: one process per GPU
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float32 if CPU_DRY_RUN else torch.bfloat16,
        device_map="cpu" if CPU_DRY_RUN else ({"": local_rank} if world > 1 else "auto"),
        attn_implementation=os.getenv("ATTN_IMPL", "sdpa"),
    )
    adapter = find_adapter()
    peft_config = None
    if adapter:
        print(f"continuing from SFT adapter {adapter}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    elif not CPU_DRY_RUN:
        raise SystemExit("no SFT adapter under $BT_LOAD_CHECKPOINT_DIR: check SFT_JOB_ID / SFT_CHECKPOINT in config_grpo_vlm.py")
    else:
        print("no SFT adapter found: starting a fresh LoRA (dry run)")
        rank = int(os.getenv("LORA_RANK", "64"))
        peft_config = LoraConfig(r=rank, lora_alpha=int(os.getenv("LORA_ALPHA", "32")), target_modules=LM_TARGETS, lora_dropout=0.0, task_type="CAUSAL_LM")

    args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=float(os.getenv("LR", "2e-5")),
        per_device_train_batch_size=int(os.getenv("BATCH", "8")),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "8")),  # 64 rollouts per step = 8 sheets x 8 samples
        num_generations=int(os.getenv("NUM_GENERATIONS", "8")),
        max_completion_length=int(os.getenv("MAX_COMPLETION", "1024")),
        temperature=float(os.getenv("TEMPERATURE", "1.0")),
        beta=float(os.getenv("BETA", "0.0")),
        max_steps=int(os.getenv("MAX_STEPS", "60")),
        save_steps=int(os.getenv("SAVE_STEPS", "10")),
        save_only_model=True,
        logging_steps=1,
        bf16=not CPU_DRY_RUN,
        use_cpu=CPU_DRY_RUN,
        gradient_checkpointing=os.getenv("GRADIENT_CHECKPOINTING", "0" if CPU_DRY_RUN else "1") == "1",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        reward_weights=[1.0, 0.0],
    )
    trainer = SheetGRPOTrainer(
        model=model,
        reward_funcs=[geometry_reward, success_metric],
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)  # every rank calls it; only the main process writes
    if not trainer.is_world_process_zero():
        return
    timing = {k: round(v, 2) for e in trainer.state.log_history[-3:] for k, v in e.items() if "generate" in k and isinstance(v, float)}
    print(f"rollout timing (last steps): {timing}")
    with open(os.path.join(OUTPUT_DIR, "grpo_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    print(f"GRPO complete. Adapter saved to {OUTPUT_DIR}")
    if MERGE_AT_END:
        save_merged(trainer.accelerator.unwrap_model(trainer.model), os.path.join(OUTPUT_DIR, "merged"))
    if _pool is not None:
        _pool.close()


if __name__ == "__main__":
    main()
