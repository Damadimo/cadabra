"""Deterministic helpers for binary checks: normalization, grounding against the source text, parsing."""

from __future__ import annotations

import re

CURRENT_YEAR = 2026

_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.S)
_NUM = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(billion|million|thousand|bn|mm|[kmb])?(?![a-z])",
    re.I,
)
_MULTIPLIER = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


def strip_think(text: str) -> str:
    """Drop a leading <think>...</think> block (Qwen3 emits one, often empty)."""
    return _THINK.sub("", text, count=1)


def extract_json(text: str) -> str | None:
    t = _FENCE.sub("", strip_think(text).strip())
    start, end = t.find("{"), t.rfind("}")
    return t[start : end + 1] if start != -1 and end > start else None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.&-]", " ", s.lower())).strip()


def numbers_in(text: str) -> list[float]:
    """Every number in the text, with k/M/B suffixes applied: "$2.4M" -> 2400000.0."""
    out = []
    for whole, frac, suffix in _NUM.findall(text):
        value = float(whole.replace(",", "") + (frac or ""))
        out.append(value * _MULTIPLIER[suffix.lower()] if suffix else value)
    return out


def grounded_str(value: str | None, source: str) -> bool:
    return value is None or norm(value) in norm(source)


def grounded_number(value: float | None, source: str, rel_tol: float = 0.005) -> bool:
    if value is None:
        return True
    return any(abs(n - value) <= max(abs(value) * rel_tol, 0.01) for n in numbers_in(source))


def grounded_value(value: str, source: str) -> bool:
    """A free-form value ("$18M", "18,000,000", "Newark") is grounded as text or as numbers."""
    if grounded_str(value, source):
        return True
    nums = numbers_in(value)
    return bool(nums) and all(grounded_number(n, source) for n in nums)


def grounded_years(value: float | None, source: str) -> bool:
    """Years in business may be stated directly or derived from a founding year."""
    if value is None or grounded_number(value, source):
        return True
    return any(abs((CURRENT_YEAR - int(y)) - value) <= 1 for y in _YEAR.findall(source))


def same(pred, gold, rel_tol: float = 0.005) -> bool:
    """Field-level match used for scoring against the gold set."""
    if pred is None or gold is None:
        return pred is None and gold is None
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return abs(pred - gold) <= max(abs(gold) * rel_tol, 1e-9)
    if isinstance(gold, list) and isinstance(pred, list):
        return sorted(norm(str(x)) for x in pred) == sorted(norm(str(x)) for x in gold)
    return norm(str(pred)) == norm(str(gold))
