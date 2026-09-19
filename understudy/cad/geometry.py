"""Run CadQuery code and score the resulting solid against a reference.

Everything here runs inside a worker process (see pool.py): model-written code is screened, executed with a
restricted namespace, and the resulting shape is compared to the reference shape with exact OCC booleans.

Metrics
- runs: the code parsed, passed screening, executed, and produced a solid with positive volume.
- iou: volumetric IoU in the reference's own coordinates (placement and orientation must match).
- iou_aligned: best IoU over the 24 axis-aligned rotations after centering both solids. It ignores where a
  single part sits and which way it faces, but not its shape or size. It is the headline metric because the
  dataset's reference code is inconsistent about applying global transforms (see scripts/audit_cadcoder.py).
- chamfer: symmetric mean surface distance after the same alignment, as a fraction of the reference's bbox diagonal.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import math
import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import cadquery as cq
import numpy as np

ALLOWED_IMPORTS = {"cadquery", "math", "numpy"}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input", "breakpoint",
    "getattr", "setattr", "delattr", "exit", "quit", "help", "memoryview", "__builtins__", "__loader__", "__spec__",
}
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list",
        "map", "max", "min", "pow", "range", "reversed", "round", "set", "sorted", "sum", "tuple", "zip",
        "ValueError", "Exception", "ZeroDivisionError", "TypeError", "IndexError", "KeyError", "True", "False", "None",
    )
    if hasattr(builtins, name)
}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name} is not allowed")
    return __import__(name, globals, locals, fromlist, level)


_IO_LINE = re.compile(r"^.*\b(?:exporters|importers|show_object)\b.*$|^.*\.(?:export\w*|save)\(.*$", re.M)
IO_ATTRS = {"exporters", "importers", "occ_impl", "vis", "jupyter_tools", "cq_directive", "save", "toSvg"}


def strip_io(code: str) -> str:
    """Drop display/export/import lines: they don't change geometry and must not touch the disk."""
    return _IO_LINE.sub("", code)


def screen(code: str) -> str | None:
    """Return why the code can't be run, or None if it may run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of {alias.name} is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from {node.module} is not allowed"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"use of {node.id} is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attribute access is not allowed"
        elif isinstance(node, ast.Attribute) and (node.attr in IO_ATTRS or node.attr.lower().startswith(("export", "import"))):
            return f"file access ({node.attr}) is not allowed"
    return None


def to_shape(obj) -> cq.Shape | None:
    """Collapse whatever the code produced into one solid-bearing shape."""
    if isinstance(obj, cq.Assembly):
        obj = obj.toCompound()
    if isinstance(obj, cq.Workplane):
        vals = [v for v in obj.vals() if isinstance(v, cq.Shape)]
        solids = [s for v in vals for s in v.Solids()]
        if not solids:
            try:
                found = obj.findSolid()
                solids = list(found.Solids()) if found is not None else []
            except ValueError:
                solids = []
        if not solids:
            return None
        return solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
    if isinstance(obj, cq.Shape):
        solids = obj.Solids()
        if not solids:
            return None
        return solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
    return None


def run_code(code: str) -> tuple[cq.Shape | None, str | None, float]:
    """Execute CadQuery code; the result is `r`, else `result`, else the last Workplane/Shape defined."""
    t0 = time.perf_counter()
    code = strip_io(code)
    problem = screen(code)
    if problem:
        return None, problem, time.perf_counter() - t0
    namespace = {
        "__builtins__": {**SAFE_BUILTINS, "__import__": _restricted_import, "print": lambda *a, **k: None},
        "__name__": "__cad__",
        "cq": cq,
        "cadquery": cq,
        "math": math,
        "show_object": lambda *args, **kwargs: None,
    }
    try:
        exec(compile(code, "<cad>", "exec"), namespace)
    except Exception as e:  # noqa: BLE001 - model code can raise anything
        return None, f"{type(e).__name__}: {str(e)[:300]}", time.perf_counter() - t0
    candidates = [namespace[k] for k in ("r", "result") if k in namespace]
    if not candidates:
        candidates = [v for v in reversed(list(namespace.values())) if isinstance(v, (cq.Workplane, cq.Shape, cq.Assembly))]
    for obj in candidates:
        try:
            shape = to_shape(obj)
        except Exception as e:  # noqa: BLE001
            return None, f"could not extract a solid: {type(e).__name__}: {e}", time.perf_counter() - t0
        if shape is not None:
            try:
                volume = shape.Volume()
            except Exception:  # noqa: BLE001
                volume = 0.0
            if volume > 1e-12:
                return shape, None, time.perf_counter() - t0
            return None, "the result has zero volume", time.perf_counter() - t0
    return None, "no solid found (assign the final model to r)", time.perf_counter() - t0


def fused(shape: cq.Shape) -> cq.Shape:
    """Fuse a multi-solid compound so volumes don't double-count overlaps."""
    solids = shape.Solids()
    if len(solids) <= 1:
        return shape
    try:
        return solids[0].fuse(*solids[1:]).clean()
    except Exception:  # noqa: BLE001
        return shape


def bbox(shape: cq.Shape) -> tuple[np.ndarray, np.ndarray]:
    bb = shape.BoundingBox()
    return np.array([bb.xmin, bb.ymin, bb.zmin]), np.array([bb.xmax, bb.ymax, bb.zmax])


def iou(a: cq.Shape, b: cq.Shape) -> float:
    va, vb = a.Volume(), b.Volume()
    inter = a.intersect(b).Volume()
    union = va + vb - inter
    return float(min(max(inter / union, 0.0), 1.0)) if union > 0 else 0.0


@lru_cache(maxsize=1)
def rotations() -> list[np.ndarray]:
    """The 24 proper rotations that map axes onto axes."""
    out = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)):
            m = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            if round(np.linalg.det(m)) == 1:
                out.append(m)
    return out


def transformed(shape: cq.Shape, rot: np.ndarray, shift: np.ndarray) -> cq.Shape:
    """Rigid transform x -> rot @ x + shift (built as a gp_Trsf directly; cq.Matrix reads nested lists differently)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    trsf = gp_Trsf()
    trsf.SetValues(*(float(v) for i in range(3) for v in (rot[i, 0], rot[i, 1], rot[i, 2], shift[i])))
    return cq.Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, trsf, True).Shape())


def aligned_candidates(pred: cq.Shape, gold: cq.Shape, rel_tol: float = 0.25) -> list[cq.Shape]:
    """pred rotated by each axis-aligned rotation whose bbox extents roughly match gold's, centered on gold."""
    gmin, gmax = bbox(gold)
    gcenter, gext = (gmin + gmax) / 2, gmax - gmin
    pmin, pmax = bbox(pred)
    pcenter, pext = (pmin + pmax) / 2, pmax - pmin
    scale = max(float(gext.max()), 1e-9)
    out = []
    for rot in rotations():
        ext = np.abs(rot) @ pext
        if np.all(np.abs(ext - gext) <= rel_tol * scale + 1e-9):
            out.append(transformed(pred, rot, gcenter - rot @ pcenter))
    if not out:  # extents disagree everywhere: fall back to plain centering
        out.append(transformed(pred, np.eye(3), gcenter - pcenter))
    return out


def surface_points(shape: cq.Shape, n: int, seed: int = 0) -> np.ndarray:
    import trimesh

    lo, hi = bbox(shape)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    verts, tris = shape.tessellate(0.002 * diag, 0.2)
    mesh = trimesh.Trimesh(vertices=[(v.x, v.y, v.z) for v in verts], faces=tris, process=False)
    points, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(points)


def chamfer(a: cq.Shape, b: cq.Shape, n: int = 2048) -> float:
    """Symmetric mean nearest-surface distance, as a fraction of b's bbox diagonal."""
    from scipy.spatial import cKDTree

    pa, pb = surface_points(a, n), surface_points(b, n)
    lo, hi = bbox(b)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    da, _ = cKDTree(pb).query(pa)
    db, _ = cKDTree(pa).query(pb)
    return float((da.mean() + db.mean()) / 2 / diag)


def mesh_export(shape: cq.Shape, fmt: str = "stl") -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"part.{fmt}"
        lo, hi = bbox(shape)
        diag = float(np.linalg.norm(hi - lo)) or 1.0
        cq.exporters.export(shape, str(path), tolerance=0.002 * diag, angularTolerance=0.2)
        return path.read_bytes()


def stats(shape: cq.Shape) -> dict:
    lo, hi = bbox(shape)
    return {
        "volume": float(shape.Volume()),
        "bbox_min": lo.round(6).tolist(),
        "bbox_max": hi.round(6).tolist(),
        "extents": sorted((hi - lo).round(6).tolist()),
        "n_solids": len(shape.Solids()),
        "n_faces": len(shape.Faces()),
    }


_GOLD_CACHE: dict[str, tuple[cq.Shape | None, str | None]] = {}


def _gold(code: str) -> tuple[cq.Shape | None, str | None]:
    key = hashlib.sha1(code.encode()).hexdigest()
    if key not in _GOLD_CACHE:
        if len(_GOLD_CACHE) > 512:
            _GOLD_CACHE.clear()
        shape, err, _ = run_code(code)
        _GOLD_CACHE[key] = (fused(shape) if shape is not None else None, err)
    return _GOLD_CACHE[key]


def evaluate(code: str, gold_code: str | None = None, want_mesh: bool = False, want_chamfer: bool = True) -> dict:
    """Run `code`; if `gold_code` is given, score it against the reference solid."""
    out: dict = {"runs": False, "error": None, "iou": None, "iou_aligned": None, "chamfer": None}
    shape, err, exec_s = run_code(code)
    out["exec_s"] = round(exec_s, 4)
    if shape is None:
        out["error"] = err
        return out
    shape = fused(shape)
    out["runs"] = True
    out["stats"] = stats(shape)
    if want_mesh:
        out["mesh_stl"] = mesh_export(shape)
    if gold_code is None:
        return out
    gold, gold_err = _gold(gold_code)
    if gold is None:
        out["gold_error"] = gold_err
        return out
    t0 = time.perf_counter()
    try:
        out["iou"] = iou(shape, gold)
    except Exception as e:  # noqa: BLE001
        out["metric_error"] = f"iou: {type(e).__name__}: {e}"
        out["iou"] = 0.0
    best, best_shape = out["iou"], shape
    try:
        for cand in aligned_candidates(shape, gold):
            score = iou(cand, gold)
            if score > best:
                best, best_shape = score, cand
    except Exception as e:  # noqa: BLE001
        out["metric_error"] = f"iou_aligned: {type(e).__name__}: {e}"
    out["iou_aligned"] = best
    if want_chamfer:
        try:
            out["chamfer"] = chamfer(best_shape, gold)
        except Exception as e:  # noqa: BLE001
            out["metric_error"] = f"chamfer: {type(e).__name__}: {e}"
    if want_mesh:
        out["gold_mesh_stl"] = mesh_export(gold)
        out["aligned_mesh_stl"] = mesh_export(best_shape) if best_shape is not shape else out["mesh_stl"]
    out["metric_s"] = round(time.perf_counter() - t0, 4)
    return out
