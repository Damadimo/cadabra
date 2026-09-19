from __future__ import annotations

import os
from importlib import import_module

from .base import Check, TaskSpec

__all__ = ["Check", "TaskSpec", "load_task"]


def load_task(name: str | None = None) -> TaskSpec:
    """Task module by name, defaulting to $UNDERSTUDY_TASK (insurance)."""
    return import_module(f"{__name__}.{name or os.getenv('UNDERSTUDY_TASK', 'insurance')}").TASK
