"""Render-and-compare: score a candidate program against the INPUT drawing sheet (no reference solid needed).

Each candidate is built, rendered with the same renderer that drew the input sheet, and compared view by view:
the score is the mean IoU of the part's silhouette-and-edge masks in the FRONT, TOP and RIGHT views. A candidate that
builds the drawn part in the drawn frame scores ~1.0. Used at inference time to pick the best of N samples from our
model; it never looks at the answer key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_renderer = None
CELL = 512
MARGIN = 26  # skip the view label strip


def _masks(sheet: np.ndarray) -> list[np.ndarray]:
    """Foreground (non-white) mask of each orthographic quadrant: FRONT, TOP, RIGHT."""
    out = []
    for idx in range(3):
        r, c = divmod(idx, 2)
        tile = sheet[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL]
        tile = tile[MARGIN : CELL - 4, 4 : CELL - 4]
        out.append(tile.min(axis=2) < 235)
    return out


def sheet_array(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    scores = []
    for ma, mb in zip(_masks(a), _masks(b)):
        union = np.logical_or(ma, mb).sum()
        scores.append(float(np.logical_and(ma, mb).sum() / union) if union else 1.0)
    return float(np.mean(scores))


def render_score(code: str, sheet_path: str) -> dict:
    """Build `code`, render it like the input sheet, and compare. Runs inside a CAD worker process."""
    import tempfile

    from .geometry import fused, run_code
    from .render import SheetRenderer

    global _renderer
    shape, err, _ = run_code(code)
    if shape is None:
        return {"runs": False, "error": err, "render_score": 0.0}
    if _renderer is None:
        _renderer = SheetRenderer(CELL)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidate.png"
        _renderer.sheet(fused(shape), out)
        score = similarity(sheet_array(out), sheet_array(sheet_path))
    return {"runs": True, "error": None, "render_score": score}
