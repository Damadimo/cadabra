"""LoRA SFT of Qwen3-VL: rendered 2x2 part sheet + bounding box -> CadQuery. Runs inside the Baseten Training container
(see config_vlm.py).

Reads data/train.jsonl (+ data/val.jsonl if present) and the PNGs they name. One row:
  {"images": ["images/<id>.png"],
   "prompt": [{"role": "system", "content": "..."},
              {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Bounding box: ..."}]}],
   "completion": [{"role": "assistant", "content": "```python\n...\n```"}]}
Loss is on the completion only. LoRA touches the language model only; the vision tower and merger stay frozen.
LoRA checkpoints go to $BT_CHECKPOINT_DIR every SAVE_STEPS. With MERGE_AT_END=1 (default on GPU) a merged full-weight
model that vLLM serves like the base repo goes to $BT_CHECKPOINT_DIR/merged.

Rows are converted at load time; the files on disk stay as written:
- Every message content becomes a list of typed parts. Arrow holds one type per column, so a string system/assistant
  content next to a list user content cannot load as is. TRL and the Qwen chat template accept the list form.
- Image paths resolve against DATA_DIR (default: data/ next to this script), then this script's folder. The column is
  cast to datasets.Image, so PNGs decode to RGB PIL images lazily, one batch at a time, inside TRL's collator.
"""

import json
import os
import shutil
import time
from pathlib import Path

import torch
from datasets import Dataset, Image, List
from peft import LoraConfig
from PIL import Image as PILImage
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
from trl import SFTConfig, SFTTrainer

from vlm_common import CPU_DRY_RUN, DATA_DIR, IMAGE_PIXELS, MODEL_ID, as_parts, image_path, save_merged

OUTPUT_DIR = os.getenv("BT_CHECKPOINT_DIR", "./checkpoints")
RANK = int(os.getenv("LORA_RANK", "16"))
# alpha fixed at 32 (scale alpha/r): with this parametrization the best LR barely moves with rank ("LoRA Without Regret",
# Thinking Machines 2025), so LR=2e-4 carries over from r=16 to r=64.
ALPHA = int(os.getenv("LORA_ALPHA", "32"))
MAX_LEN = int(os.getenv("MAX_LEN", "6144"))  # ~1024 image + ~600 text tokens per row; far above any row
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "200"))
# Wall-clock cap on the training loop (hours, 0 = none). When it runs out, training stops at the next step and the
# final adapter + merged model are still written, so a fixed GPU window always ends with something deployable.
TIME_BUDGET_H = float(os.getenv("TIME_BUDGET_H", "0"))
MERGE_AT_END = os.getenv("MERGE_AT_END", "0" if CPU_DRY_RUN else "1") == "1"

# LoRA on the language model's projections only. PEFT full-matches this regex against module names such as
# model.language_model.layers.3.mlp.up_proj. The vision tower and merger (model.visual.*) get no adapters, so they stay
# frozen, and lm_head stays unwrapped, which TRL's default chunked loss requires.
LM_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_rows(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path) as f:
        return [
            {
                "images": [image_path(p) for p in row["images"]],
                "prompt": [as_parts(m) for m in row["prompt"]],
                "completion": [as_parts(m) for m in row["completion"]],
            }
            for row in map(json.loads, filter(str.strip, f))
        ]


def visual_tokens(processor, path):
    with PILImage.open(path) as im:
        grid = processor.image_processor(images=[im.convert("RGB")], return_tensors="pt")["image_grid_thw"][0]
    return int(grid.prod()) // processor.image_processor.merge_size**2


def drop_too_long(processor, rows, image_tokens, split):
    """The collator cuts sequences at MAX_LEN from the end, which would drop the end of a completion (the loss target)."""
    texts = [processor.apply_chat_template(r["prompt"] + r["completion"], tokenize=False) for r in rows]
    text_tokens = [len(ids) for ids in processor.tokenizer(texts, add_special_tokens=False)["input_ids"]]
    # Each <|image_pad|> in the rendered text expands to image_tokens placeholders in the processor.
    lengths = [n + len(r["images"]) * (image_tokens - 1) for n, r in zip(text_tokens, rows)]
    kept = [r for r, n in zip(rows, lengths) if n <= MAX_LEN]
    s = sorted(lengths)
    print(
        f"{split}: {len(rows)} rows, tokens/row p50={s[len(s) // 2]} p95={s[int(len(s) * 0.95)]} max={s[-1]}; "
        f"dropped {len(rows) - len(kept)} longer than MAX_LEN={MAX_LEN}"
    )
    return kept


def to_dataset(rows):
    return Dataset.from_list(rows).cast_column("images", List(Image(mode="RGB")))


def dataloader_workers():
    """Workers overlap PNG decoding and image preprocessing (done in the collator) with GPU compute. They hand batches
    over through /dev/shm (~200 MB of float32 pixels per batch of 8), which containers often cap at 64 MB."""
    if "NUM_WORKERS" in os.environ:
        return int(os.environ["NUM_WORKERS"])
    if CPU_DRY_RUN:
        return 0
    try:
        return 4 if shutil.disk_usage("/dev/shm").total >= 8 * 2**30 else 0
    except OSError:
        return 0


processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=IMAGE_PIXELS, max_pixels=IMAGE_PIXELS)
train_rows, val_rows = load_rows("train.jsonl"), load_rows("val.jsonl")
if not train_rows:
    raise SystemExit(f"no rows in {os.path.join(DATA_DIR, 'train.jsonl')}")
image_tokens = visual_tokens(processor, train_rows[0]["images"][0])
print(f"image budget {IMAGE_PIXELS} px -> {image_tokens} visual tokens per image")
train = to_dataset(drop_too_long(processor, train_rows, image_tokens, "train"))
val = to_dataset(drop_too_long(processor, val_rows, image_tokens, "val")) if val_rows else None

# Multi-GPU: run_vlm.sh starts one process per GPU with torchrun (NPROC>1); each holds a full copy on its own GPU (DDP).
WORLD = int(os.getenv("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float32 if CPU_DRY_RUN else torch.bfloat16,
    device_map="cpu" if CPU_DRY_RUN else ({"": LOCAL_RANK} if WORLD > 1 else "auto"),
    attn_implementation=os.getenv("ATTN_IMPL", "sdpa"),
)
# Gradient checkpointing + LoRA: the frozen embedding output must require grad for gradients to reach the adapters
# through checkpointed blocks. This hooks the language model's embed_tokens only, so no backward pass runs through the
# vision tower. SFTTrainer repeats the call for PEFT models; the second hook is a no-op.
model.enable_input_require_grads()

peft_config = LoraConfig(
    r=RANK,
    lora_alpha=ALPHA,
    target_modules=LM_TARGETS,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
# INIT_ADAPTER: keep training an existing adapter (e.g. a second epoch) instead of starting a new one. "auto" = the newest
# adapter under $BT_LOAD_CHECKPOINT_DIR (config_vlm.py loads it when CONTINUE_JOB/CONTINUE_CHECKPOINT are set at push).
INIT_ADAPTER = os.getenv("INIT_ADAPTER", "")
if INIT_ADAPTER == "auto":
    found = sorted(Path(os.getenv("BT_LOAD_CHECKPOINT_DIR", "/nonexistent")).rglob("adapter_config.json"))
    if not found:
        raise SystemExit("INIT_ADAPTER=auto but no adapter under $BT_LOAD_CHECKPOINT_DIR")
    INIT_ADAPTER = str(found[-1].parent)
if INIT_ADAPTER:
    from peft import PeftModel

    print(f"continuing adapter {INIT_ADAPTER}")
    model = PeftModel.from_pretrained(model, INIT_ADAPTER, is_trainable=True)
    peft_config = None

args = SFTConfig(
    learning_rate=float(os.getenv("LR", "2e-4")),
    data_seed=int(os.getenv("DATA_SEED", "42")),  # a continued epoch should see a new order
    ddp_find_unused_parameters=False,
    num_train_epochs=float(os.getenv("EPOCHS", "2")),
    max_steps=int(os.getenv("MAX_STEPS", "-1")),
    per_device_train_batch_size=int(os.getenv("BATCH", "8")),
    per_device_eval_batch_size=int(os.getenv("BATCH", "8")),
    gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "2")),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=MAX_LEN,
    packing=False,  # TRL does not pack image rows
    completion_only_loss=True,  # prompt tokens, image tokens included, get label -100
    warmup_steps=int(os.getenv("WARMUP_STEPS", "10")),
    lr_scheduler_type="cosine",
    logging_steps=int(os.getenv("LOG_STEPS", "5")),
    save_steps=SAVE_STEPS,
    save_only_model=os.getenv("SAVE_ONLY_MODEL", "1") == "1",  # adapter only; we never resume, we deploy
    eval_strategy="steps" if val is not None else "no",
    eval_steps=SAVE_STEPS,
    dataloader_num_workers=dataloader_workers(),
    bf16=not CPU_DRY_RUN,
    use_cpu=CPU_DRY_RUN,
    report_to="none",
    output_dir=OUTPUT_DIR,
)

class TimeBudget(TrainerCallback):
    """Prints the projected run time after a few steps; stops training once TIME_BUDGET_H is spent."""

    def __init__(self, hours: float):
        self.hours, self.t0 = hours, None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_h = (time.time() - self.t0) / 3600
        if state.global_step == 20:
            total_h = elapsed_h / 20 * state.max_steps
            print(f"[time] {elapsed_h * 180:.1f} s/step over 20 steps -> {state.max_steps} steps take ~{total_h:.2f} h"
                  + (f" (budget {self.hours} h: stops near step {int(self.hours / elapsed_h * 20)})" if self.hours and total_h > self.hours else ""), flush=True)
        stop = bool(self.hours) and elapsed_h >= self.hours
        if self.hours and torch.distributed.is_available() and torch.distributed.is_initialized():
            # every rank must stop on the same step, or the next all-reduce hangs: rank 0 decides
            flag = torch.tensor([int(stop)], device=torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else "cpu")
            torch.distributed.broadcast(flag, src=0)
            stop = bool(flag.item())
        if stop:
            print(f"[time] budget of {self.hours} h reached at step {state.global_step}/{state.max_steps}: stopping and saving", flush=True)
            control.should_training_stop = True
            control.should_save = True
        return control


trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=val,
    processing_class=processor,
    peft_config=peft_config,
    callbacks=[TimeBudget(TIME_BUDGET_H)],
)

trainable = {n: p.numel() for n, p in trainer.model.named_parameters() if p.requires_grad}
stray = [n for n in trainable if "lora_" not in n or "language_model" not in n]
if not trainable or stray:
    raise SystemExit(f"LoRA must train language-model adapters only; unexpected trainable params: {stray[:5]}")
print(
    f"LoRA r={RANK} on {len({n.split('.lora_')[0] for n in trainable})} language-model linear layers, "
    f"{sum(trainable.values()) / 1e6:.1f}M trainable params; vision tower and merger frozen"
)
# Completion-only loss on a real row: labels must cover the assistant's code and nothing from the prompt or the image.
check = trainer.data_collator([train[0]])
labels = check["labels"][0]
supervised = labels[labels != -100]
if model.config.image_token_id in supervised.tolist():
    raise SystemExit("image tokens are in the loss")
print(
    f"loss on {len(supervised)} of {int(check['attention_mask'][0].sum())} tokens of row 0, "
    f"starting {processor.tokenizer.decode(supervised[:6])!r}"
)

trainer.train()
trainer.save_model(OUTPUT_DIR)  # final adapter + processor at the top level, as in ../train.py (main process only)

if trainer.is_world_process_zero():
    with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    print(f"Training complete. LoRA adapters saved to {OUTPUT_DIR}")
    if MERGE_AT_END:
        save_merged(trainer.accelerator.unwrap_model(trainer.model), os.path.join(OUTPUT_DIR, "merged"))
# The benchmark eval (EVAL_BENCH=1) runs after this script exits, as its own process (run_vlm.sh -> bench_eval.py): its
# grading workers are spawned, and a spawned child re-imports __main__, which here is this whole training script.
