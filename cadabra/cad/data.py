"""Load CAD-Coder (gudo7208/CAD-Coder, Apache-2.0; derived from Text2CAD/DeepCAD) and parse its specs.

Each record: {"id", "source_id", "split", "spec", "gold_code", "n_parts", "stated_dims", "identity_transforms"}.
Download the raw JSON files with `python -m cadabra.cad.data --download`.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

from ..config import ROOT

RAW_DIR = ROOT / "data" / "raw" / "cadcoder"
HF_BASE = "https://huggingface.co/datasets/gudo7208/CAD-Coder/resolve/main"
FILES = {
    "train_high": "cad_data_train_high.json",
    "train_middle": "cad_data_train_middle.json",
    "validation": "cad_data_validation.json",
    "test": "cad_data_test_cot.json",
}

_PREAMBLE = "Please based on the following description, create a CAD-Query Code to generate a model:"
_CJK = re.compile(r"[぀-ヿ一-鿿]")
_SHOW = re.compile(r"^\s*#?\s*show_object\(.*$", re.M)
_NUM = r"(-?\d+(?:\.\d+)?)"
_TRIPLE = rf"[\[\(]\s*{_NUM}\s*°?\s*,\s*{_NUM}\s*°?\s*,\s*{_NUM}\s*°?\s*[\]\)]"
_EULER = re.compile(rf"Euler angles[^.\[\(]{{0,40}}?{_TRIPLE}", re.I)
_TRANSLATION = re.compile(rf"translation vector[^.\[\(]{{0,40}}?{_TRIPLE}", re.I)
_DIM = {k: re.compile(rf"\b{k}\b(?: of)?(?: approximately| about)?\s*{_NUM}", re.I) for k in ("length", "width", "height")}


def download(splits: tuple[str, ...] = ("train_high", "validation", "test")) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split in splits:
        path = RAW_DIR / FILES[split]
        if not path.exists():
            print(f"downloading {FILES[split]} ...")
            urllib.request.urlretrieve(f"{HF_BASE}/{FILES[split]}", path)


def extract_spec(user_content: str) -> str:
    if "\ndescription:\n" in user_content:
        return user_content.split("\ndescription:\n", 1)[1].strip()
    if user_content.startswith(_PREAMBLE):
        return user_content[len(_PREAMBLE):].strip()
    return user_content.strip()


def clean_code(code: str) -> str | None:
    """Reference code without display calls; None for the dataset's placeholder rows."""
    if _CJK.search(code) or ("cadquery" not in code and "cq." not in code):
        return None
    from .geometry import strip_io

    code = strip_io(_SHOW.sub("", code))
    code = re.sub(r"\n{3,}", "\n\n", code).strip()
    return code + "\n"


def parse_spec(spec: str) -> dict:
    eulers = [tuple(float(x) for x in m.groups()) for m in _EULER.finditer(spec)]
    translations = [tuple(float(x) for x in m.groups()) for m in _TRANSLATION.finditer(spec)]
    identity = all(abs(v) < 1e-9 for t in eulers + translations for v in t)
    lengths = [float(m.group(1)) for m in _DIM["length"].finditer(spec)]
    widths = [float(m.group(1)) for m in _DIM["width"].finditer(spec)]
    heights = [float(m.group(1)) for m in _DIM["height"].finditer(spec)]
    dims = list(zip(lengths, widths, heights))
    systems = len(re.findall(r"\b(?:new|another|second|third|fourth|fifth|sixth) (?:local )?coordinate system", spec, re.I))
    ordinals = set(re.findall(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth) part\b", spec, re.I))
    n_parts = max(1, systems, len(ordinals))
    return {"n_parts": n_parts, "stated_dims": dims, "identity_transforms": identity, "eulers": eulers, "translations": translations}


def load_split(split: str, limit: int | None = None) -> list[dict]:
    path = RAW_DIR / FILES[split]
    if not path.exists():
        download((split,))
    rows = json.loads(path.read_text())
    out = []
    for i, row in enumerate(rows[:limit]):
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        answer = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        code = clean_code(answer)
        if code is None:
            continue
        spec = extract_spec(user)
        out.append(
            {
                "id": f"{split}:{i:05d}",
                "source_id": Path(row.get("model_path", f"{split}-{i}")).stem,
                "split": split,
                "spec": spec,
                "gold_code": code,
                **parse_spec(spec),
            }
        )
    return out


_LIT = re.compile(r"(?<![\w.])-?\d+\.\d+|(?<![\w.])-?\d+(?![\w.])")


def _numbers(text: str) -> set[float]:
    return {round(abs(float(x)), 4) for x in _LIT.findall(text)}


def spec_coverage(spec: str, code: str) -> float:
    """Share of the reference code's non-trivial numbers that the spec states (a well-posedness check:
    if the answer uses numbers the spec never gives, the spec can't determine the part)."""
    trivial = {0.0, 1.0, 2.0, 90.0, 180.0, 270.0, 360.0}
    needed = {x for x in _numbers(code) if x not in trivial}
    if not needed:
        return 1.0
    given = _numbers(spec)
    tol = lambda a: any(abs(a - b) <= max(1e-4, 0.002 * abs(a)) for b in given)  # noqa: E731
    return sum(1 for x in needed if tol(x)) / len(needed)


def normalized_code(code: str) -> str:
    """Whitespace- and comment-insensitive form, for duplicate detection."""
    lines = [re.sub(r"#.*$", "", ln).strip() for ln in code.splitlines()]
    return "\n".join(ln for ln in lines if ln)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()
    for split in ("train_high", "validation", "test"):
        recs = load_split(split)
        print(split, len(recs), "usable rows")
