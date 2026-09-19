"""Stage 2 (optional): GRPO on spec→CadQuery with a geometry reward. Runs inside a Baseten Training job.

Continues training the stage-1 LoRA adapter (found under $SFT_ADAPTER or $BT_LOAD_CHECKPOINT_DIR). Each prompt is
sampled NUM_GENERATIONS times; every completion's code is executed by the sandboxed checker in cadcheck/ and
rewarded with its aligned IoU against the reference (crashes -0.2, no code block -0.5). The result is a LoRA
adapter with the same rank as stage 1, so it deploys exactly like the SFT checkpoint.
"""

import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from cadcheck.pool import CadPool

MODEL_ID = os.getenv("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
OUTPUT_DIR = os.getenv("BT_CHECKPOINT_DIR", "./checkpoints")
CPU_DRY_RUN = os.getenv("CPU_DRY_RUN") == "1"
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def find_adapter() -> str | None:
    """Explicit SFT_ADAPTER path, else the newest adapter under $BT_LOAD_CHECKPOINT_DIR."""
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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = [b for b in FENCE.findall(text) if "cq" in b or "cadquery" in b]
    if blocks:
        return blocks[-1].strip() + "\n"
    return text[text.index("import cadquery"):] if "import cadquery" in text else None


_pool: CadPool | None = None
_cache: dict = {}


def score(codes_and_golds: list[tuple[str, str]]) -> list[dict]:
    global _pool
    if _pool is None:
        _pool = CadPool(workers=int(os.getenv("REWARD_WORKERS", max(2, (os.cpu_count() or 4) - 2))), timeout=20)
    todo = [(c, g) for c, g in codes_and_golds if (c, g) not in _cache]
    for key, res in zip(todo, _pool.map([{"code": c, "gold_code": g, "want_chamfer": False} for c, g in todo])):
        _cache[key] = res
    if len(_cache) > 20000:
        _cache.clear()
    return [_cache.get(k, {"runs": False}) for k in codes_and_golds]


def _texts(completions) -> list[str]:
    return [c[0]["content"] if isinstance(c, list) else c for c in completions]


def geometry_reward(prompts, completions, gold_code, **kwargs) -> list[float]:
    texts = _texts(completions)
    codes = [extract_code(t) for t in texts]
    pairs = [(c, g) for c, g in zip(codes, gold_code) if c is not None]
    results = iter(score(pairs))
    rewards = []
    for code in codes:
        if code is None:
            rewards.append(-0.5)
            continue
        res = next(results)
        rewards.append(float(res.get("iou_aligned") or 0.0) if res.get("runs") else -0.2)
    return rewards


def success_metric(prompts, completions, gold_code, **kwargs) -> list[float]:
    """Logged only (weight 0): share of completions that clear IoU 0.9."""
    codes = [extract_code(t) for t in _texts(completions)]
    out = []
    for code, gold in zip(codes, gold_code):
        res = _cache.get((code, gold), {}) if code else {}
        out.append(1.0 if res.get("runs") and (res.get("iou_aligned") or 0) >= 0.9 else 0.0)
    return out


def main() -> None:
    dataset = load_dataset("json", data_files="data/grpo.jsonl", split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32 if CPU_DRY_RUN else torch.bfloat16,
        device_map="cpu" if CPU_DRY_RUN else "auto",
    )
    adapter = find_adapter()
    peft_config = None
    if adapter:
        print(f"continuing from SFT adapter {adapter}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        print("no SFT adapter found: starting a fresh LoRA")
        rank = int(os.getenv("LORA_RANK", "16"))
        peft_config = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules="all-linear", lora_dropout=0.0, task_type="CAUSAL_LM")

    use_vllm = os.getenv("USE_VLLM", "1") == "1" and not CPU_DRY_RUN
    args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=float(os.getenv("LR", "1e-5")),
        per_device_train_batch_size=int(os.getenv("BATCH", "8")),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "4")),
        num_generations=int(os.getenv("NUM_GENERATIONS", "8")),
        max_completion_length=int(os.getenv("MAX_COMPLETION", "1024")),
        temperature=float(os.getenv("TEMPERATURE", "1.0")),
        beta=float(os.getenv("BETA", "0.0")),
        max_steps=int(os.getenv("MAX_STEPS", "150")),
        save_steps=int(os.getenv("SAVE_STEPS", "25")),
        logging_steps=1,
        bf16=not CPU_DRY_RUN,
        use_cpu=CPU_DRY_RUN,
        gradient_checkpointing=not CPU_DRY_RUN,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=float(os.getenv("VLLM_MEM", "0.35")),
        report_to="none",
        reward_weights=[1.0, 0.0],
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[geometry_reward, success_metric],
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "grpo_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    print(f"GRPO complete. Adapter saved to {OUTPUT_DIR}")
    if _pool is not None:
        _pool.close()


if __name__ == "__main__":
    main()
