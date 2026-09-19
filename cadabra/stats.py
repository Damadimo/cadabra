"""Percentiles and bootstrap confidence intervals, no numpy needed."""

from __future__ import annotations

import random
from statistics import mean
from typing import Iterable


def pct(values: Iterable[float | None], q: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def bootstrap_ci(xs: Iterable[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple:
    xs = list(xs)
    if not xs:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(iters))
    return means[int(iters * alpha / 2)], means[int(iters * (1 - alpha / 2)) - 1]


def safe_mean(xs: Iterable[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None
