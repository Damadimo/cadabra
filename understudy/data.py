"""JSONL IO, leakage-safe splits and cheap deterministic perturbations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_of(row: dict) -> str:
    return str(row.get("source_id") or row.get("id"))


def split_by_source(rows: list[dict], test_frac: float = 0.2, seed: int = 13) -> tuple[list[dict], list[dict]]:
    """Split so every variant of one source document lands on the same side (no train/test leakage)."""
    sources = sorted({source_of(r) for r in rows})
    random.Random(seed).shuffle(sources)
    n_test = max(1, round(len(sources) * test_frac)) if len(sources) > 1 else 0
    test_sources = set(sources[:n_test])
    train = [r for r in rows if source_of(r) not in test_sources]
    test = [r for r in rows if source_of(r) in test_sources]
    return train, test


def _swap_typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def perturb(record: dict, k: int, seed: int = 13) -> dict:
    """Surface noise that keeps every fact: letter-swap typos in words and shuffled blocks.
    Deterministic in (seed, id, k) so labeling runs can resume."""
    rng = random.Random(f"{seed}-{record['id']}-{k}")

    def noisy(line: str) -> str:
        return " ".join(_swap_typo(w, rng) if w.isalpha() and rng.random() < 0.06 else w for w in line.split(" "))

    variant = {**record, "id": f"{record['id']}~p{k}", "source_id": source_of(record), "derived_from": record["id"]}
    if "lines" in record:  # line-oriented records (logs): keep line order, it carries meaning
        variant["lines"] = [noisy(line) for line in record["lines"]]
        return variant
    blocks = "\n".join(noisy(line) for line in record["text"].splitlines()).split("\n\n")
    if len(blocks) > 2:
        head, rest = blocks[0], blocks[1:]
        rng.shuffle(rest)
        blocks = [head, *rest]
    variant["text"] = "\n\n".join(blocks)
    return variant
