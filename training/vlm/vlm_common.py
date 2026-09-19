"""Shared by train_vlm.py (SFT) and grpo_vlm.py (RL): message format, image paths, and the merged-weights export."""

import json
import os
import shutil

from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.getenv("BASE_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
DATA_DIR = os.getenv("DATA_DIR", os.path.join(HERE, "data"))
CPU_DRY_RUN = os.getenv("CPU_DRY_RUN") == "1"  # local smoke test on a laptop; never set on Baseten

# Image budget. Qwen3-VL cuts an image into 16x16 px patches and merges each 2x2 group into one LLM token, so a token
# covers 32x32 px. Pinning min_pixels = max_pixels = IMAGE_PIXELS resizes every image to that area (aspect ratio kept,
# sides rounded to multiples of 32), so the token count never depends on the render size. The default, 1024*1024,
# leaves a 1024x1024 sheet untouched: 64x64 patches -> exactly 1024 visual tokens, 256 per view of the 2x2 sheet.
# 768*768 (576 tokens) trains faster but blurs small holes and fillets; a bigger budget only upsamples the render.
# merged/preprocessor_config.json carries the same budget, so vLLM resizes identically at serving time.
IMAGE_PIXELS = int(os.getenv("IMAGE_PIXELS", str(1024 * 1024)))


def as_parts(message):
    content = message["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    for part in content:
        if part["type"] not in ("text", "image"):
            raise ValueError(f"unsupported content part type {part['type']!r}")
    return {"role": message["role"], "content": [{"type": p["type"], "text": p.get("text")} for p in content]}


def image_path(path):
    if os.path.isabs(path):
        return path
    for base in (DATA_DIR, HERE):
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"image {path} not found under {DATA_DIR} or {HERE}")


def save_merged(peft_model, out_dir):
    """Fold the LoRA deltas into the base weights and write a folder vLLM serves exactly like BASE_MODEL."""
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir)  # safetensors with the same tensor names as the base checkpoint
    # Non-weight files are the base repo's own, not save_pretrained's. transformers 5 writes config.json with
    # rope_parameters instead of rope_scaling, which transformers 4.57 fails to load (servers built on 4.x break), and
    # moves the processor configs into processor_config.json. LoRA changes no config, so the base files are exact, and
    # they load wherever the base model does (checked on 4.57.6 and 5.17). Only the image budget is updated to the one
    # used in training.
    src = MODEL_ID if os.path.isdir(MODEL_ID) else snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.jinja", "*.txt"])
    for name in os.listdir(src):
        if name.endswith((".json", ".jinja", ".txt")) and not name.endswith(".index.json"):
            shutil.copy(os.path.join(src, name), out_dir)
    path = os.path.join(out_dir, "preprocessor_config.json")
    with open(path) as f:
        cfg = json.load(f)
    cfg["size"] = {"shortest_edge": IMAGE_PIXELS, "longest_edge": IMAGE_PIXELS}
    for key in ("min_pixels", "max_pixels"):  # older Qwen-VL repos also carry these
        if key in cfg:
            cfg[key] = IMAGE_PIXELS
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Merged model written to {out_dir}: {sorted(os.listdir(out_dir))}")
