"""Render-and-compare: score a candidate program against the INPUT drawing sheet (no reference solid needed).

Each candidate is built, rendered with the same renderer that drew the input sheet, and compared view by view in the
FRONT, TOP and RIGHT quadrants. Score = silhouette x size:
- silhouette: mean IoU of the part's footprint (non-white pixels) over the three views,
- size: agreement of the candidate's bounding box with the one stated in the prompt (the renderer normalizes scale,
  so a uniformly mis-scaled part would otherwise look perfect).
- edges (reported, not scored): F1 of the drawn visible edges within TOL px. On 689 frontier answers it only added
  noise (AUC 0.81 alone, 0.92 mixed 50/50), since the same solid built another way has other seams and splits.
Validation (scripts/validate_verifier.py, 689 answers, 237 parts): AUC 0.959; among several answers for one part it
picks a correct one 98.1% of the time (random pick 76.3%). A candidate that builds the drawn part in the drawn frame
scores ~1.0. It only sees what the model saw (the sheet and the bounding box), never the answer key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_renderer = None
CELL = 512
MARGIN = 26  # skip the view label strip
EDGE = 70  # drawn B-rep edges are near-black; the shaded body is lighter
TOL = 3  # px


def _tiles(sheet: np.ndarray) -> list[np.ndarray]:
    """The orthographic quadrants: FRONT, TOP, RIGHT."""
    out = []
    for idx in range(3):
        r, c = divmod(idx, 2)
        tile = sheet[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL]
        out.append(tile[MARGIN : CELL - 4, 4 : CELL - 4])
    return out


def sheet_array(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _edge_f1(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.ndimage import distance_transform_edt

    if not a.any() or not b.any():
        return float(a.any() == b.any())
    near_b = distance_transform_edt(~b) <= TOL
    near_a = distance_transform_edt(~a) <= TOL
    precision, recall = float(near_b[a].mean()), float(near_a[b].mean())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compare(candidate: np.ndarray, target: np.ndarray) -> dict:
    sil, edg = [], []
    for ta, tb in zip(_tiles(candidate), _tiles(target)):
        sil.append(_iou(ta.min(axis=2) < 235, tb.min(axis=2) < 235))
        edg.append(_edge_f1(ta.max(axis=2) < EDGE, tb.max(axis=2) < EDGE))
    return {"silhouette": float(np.mean(sil)), "edges": float(np.mean(edg))}


def size_agreement(extents: list[float], bbox: list[float] | None) -> float:
    """Mean over axes of min/max between the candidate's extents and the stated bounding box (1.0 = exact)."""
    if not bbox:
        return 1.0
    return float(np.mean([min(a, b) / max(a, b) if max(a, b) > 0 else 1.0 for a, b in zip(extents, bbox)]))


def combine(parts: dict) -> float:
    return parts["silhouette"] * parts["size"]


def render_score(code: str, sheet_path: str, bbox: list[float] | None = None) -> dict:
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
        drawn = _renderer.sheet(fused(shape), out)
        parts = compare(sheet_array(out), sheet_array(sheet_path))
    parts["size"] = size_agreement(drawn["bbox"], bbox)
    return {"runs": True, "error": None, "render_score": combine(parts), **parts}
