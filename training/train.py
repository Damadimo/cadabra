"""LoRA SFT with TRL. Runs inside the Baseten Training container (see config.py).

Reads data/train.jsonl (+ data/val.jsonl if present) written by `python -m understudy.build_sft`.
Prompt/completion rows train on the answer only; plain `messages` rows train on the whole conversation.
Checkpoints go to $BT_CHECKPOINT_DIR, which Baseten syncs and can deploy. Never delete them mid-run.
"""

import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = os.getenv("BASE_MODEL", "Qwen/Qwen3-4B")
RANK = int(os.getenv("LORA_RANK", "16"))
OUTPUT_DIR = os.getenv("BT_CHECKPOINT_DIR", "./checkpoints")

train = load_dataset("json", data_files="data/train.jsonl", split="train")
has_val = os.path.exists("data/val.jsonl") and os.path.getsize("data/val.jsonl") > 0
val = load_dataset("json", data_files="data/val.jsonl", split="train") if has_val else None
print(f"train={len(train)} val={len(val) if val else 0} columns={train.column_names}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

CPU_DRY_RUN = os.getenv("CPU_DRY_RUN") == "1"  # local smoke test on a laptop; never set on Baseten

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32 if CPU_DRY_RUN else torch.bfloat16,
    device_map="cpu" if CPU_DRY_RUN else "auto",
    use_cache=False,
)

peft_config = LoraConfig(
    r=RANK,
    lora_alpha=2 * RANK,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    learning_rate=float(os.getenv("LR", "2e-4")),
    num_train_epochs=float(os.getenv("EPOCHS", "2")),
    max_steps=int(os.getenv("MAX_STEPS", "-1")),
    per_device_train_batch_size=int(os.getenv("BATCH", "4")),
    gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "4")),
    gradient_checkpointing=True,
    max_length=int(os.getenv("MAX_LEN", "4096")),
    warmup_steps=int(os.getenv("WARMUP_STEPS", "10")),  # warmup_ratio is gone in current transformers
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=int(os.getenv("SAVE_STEPS", "100")),
    eval_strategy="steps" if val is not None else "no",
    eval_steps=int(os.getenv("SAVE_STEPS", "100")),
    bf16=not CPU_DRY_RUN,
    use_cpu=CPU_DRY_RUN,
    report_to="none",
    output_dir=OUTPUT_DIR,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=val,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=1)
print(f"Training complete. Adapter saved to {OUTPUT_DIR}")
