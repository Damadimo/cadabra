"""Render a part as a 2x2 drawing sheet: FRONT, TOP, RIGHT (orthographic, one shared scale) and ISOMETRIC.

  uv run python -m understudy.cad.render --split bench --workers 12      # -> data/cad/img/<id>.png + index.json
  uv run python -m understudy.cad.render --split train --workers 12

Views are rendered in the reference solid's own frame, so the axes in the picture are the axes the code builds in:
FRONT looks along +Y (X right, Z up), TOP looks down -Z (X right, Y up), RIGHT looks along -X (Y right, Z up).
The index stores each part's bounding-box size per axis; the prompt states it because pixels alone can't carry
exact sizes for thin parts.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..config import ROOT
from ..data import read_jsonl

OUT = ROOT / "data" / "cad" / "img"
VIEWS = (  # name, camera direction (from focal point), view-up
    ("FRONT", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ("TOP", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ("RIGHT", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("ISOMETRIC", (1.0, -1.0, 0.8), (0.0, 0.0, 1.0)),
)

_renderer = None


class SheetRenderer:
    def __init__(self, cell: int = 512):
        import vtk

        self.vtk = vtk
        self.cell = cell
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1, 1, 1)
        self.win = vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.AddRenderer(self.ren)
        self.win.SetSize(cell, cell)
        self.win.SetMultiSamples(4)

    def _actors(self, shape, diag: float):
        vtk = self.vtk
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        verts, tris = shape.tessellate(0.002 * diag, 0.15)
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(np.array([v.toTuple() for v in verts], dtype=np.float64), deep=True))
        t = np.asarray(tris, dtype=np.int64)
        cells = np.hstack([np.full((len(t), 1), 3, dtype=np.int64), t]).ravel()
        polys = vtk.vtkCellArray()
        polys.SetCells(len(t), numpy_to_vtkIdTypeArray(cells, deep=True))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(polys)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(pd)
        normals.SetFeatureAngle(30)
        normals.SplittingOn()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        body = vtk.vtkActor()
        body.SetMapper(mapper)
        body.GetProperty().SetColor(0.66, 0.74, 0.86)

        epts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        for edge in shape.Edges():
            samples = [edge.positionAt(i / 31) for i in range(32)] if edge.geomType() != "LINE" else [edge.positionAt(0), edge.positionAt(1)]
            ids = [epts.InsertNextPoint(*p.toTuple()) for p in samples]
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(ids))
            for k, i in enumerate(ids):
                line.GetPointIds().SetId(k, i)
            lines.InsertNextCell(line)
        epd = vtk.vtkPolyData()
        epd.SetPoints(epts)
        epd.SetLines(lines)
        emapper = vtk.vtkPolyDataMapper()
        emapper.SetInputData(epd)
        edges = vtk.vtkActor()
        edges.SetMapper(emapper)
        edges.GetProperty().SetColor(0.05, 0.05, 0.08)
        edges.GetProperty().SetLineWidth(1.6)
        return body, edges

    def sheet(self, shape, path: Path) -> dict:
        from PIL import Image, ImageDraw
        from vtkmodules.util.numpy_support import vtk_to_numpy

        vtk = self.vtk
        bb = shape.BoundingBox()
        center = np.array([bb.center.x, bb.center.y, bb.center.z])
        size = np.array([bb.xlen, bb.ylen, bb.zlen])
        diag = max(float(np.linalg.norm(size)), 1e-9)
        body, edges = self._actors(shape, diag)
        self.ren.RemoveAllViewProps()
        self.ren.AddActor(body)
        self.ren.AddActor(edges)
        cam = self.ren.GetActiveCamera()
        sheet = Image.new("RGB", (2 * self.cell, 2 * self.cell), "white")
        half = 0.58 * float(size.max())  # one shared scale for the three orthographic views
        for idx, (name, direction, up) in enumerate(VIEWS):
            d = np.asarray(direction) / np.linalg.norm(direction)
            cam.SetFocalPoint(*center)
            cam.SetPosition(*(center + d * diag * 4))
            cam.SetViewUp(*up)
            cam.SetParallelProjection(1)
            if name == "ISOMETRIC":
                self.ren.ResetCamera()
            else:
                cam.SetParallelScale(half)
            self.ren.ResetCameraClippingRange()
            self.win.Render()
            grab = vtk.vtkWindowToImageFilter()
            grab.SetInput(self.win)
            grab.ReadFrontBufferOff()
            grab.Update()
            img = grab.GetOutput()
            w, h, _ = img.GetDimensions()
            arr = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3]
            tile = Image.fromarray(arr.astype(np.uint8))
            ImageDraw.Draw(tile).text((10, 8), name, fill=(40, 40, 60))
            sheet.paste(tile, ((idx % 2) * self.cell, (idx // 2) * self.cell))
        draw = ImageDraw.Draw(sheet)
        draw.line([(self.cell, 0), (self.cell, 2 * self.cell)], fill=(200, 200, 210), width=2)
        draw.line([(0, self.cell), (2 * self.cell, self.cell)], fill=(200, 200, 210), width=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, optimize=True)
        return {"bbox": [round(float(v), 6) for v in size]}


def _init():
    global _renderer
    os.environ.setdefault("VTK_SILENCE_GET_VOID_POINTER_WARNINGS", "1")
    _renderer = SheetRenderer()


def _render_one(args: tuple[str, str, str]) -> tuple[str, dict | None, str | None]:
    rec_id, code, out_dir = args
    from .geometry import fused, run_code

    shape, err, _ = run_code(code)
    if shape is None:
        return rec_id, None, err
    try:
        info = _renderer.sheet(fused(shape), Path(out_dir) / f"{rec_id.replace(':', '_')}.png")
    except Exception as e:  # noqa: BLE001
        return rec_id, None, f"render: {type(e).__name__}: {e}"
    return rec_id, info, None


def image_path(rec_id: str, out_dir: Path = OUT) -> Path:
    return out_dir / f"{rec_id.replace(':', '_')}.png"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="bench", help="bench | train | test_verified | shots")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    recs = read_jsonl(ROOT / "data" / "cad" / f"{args.split}.jsonl")[: args.limit]
    index_path = args.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    todo = [(r["id"], r["gold_code"], str(args.out)) for r in recs if r["id"] not in index or not image_path(r["id"], args.out).exists()]
    print(f"{len(todo)} to render ({len(recs) - len(todo)} cached) with {args.workers} workers", flush=True)
    t0, failed = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for n, (rec_id, info, err) in enumerate(ex.map(_render_one, todo, chunksize=8), 1):
            if info is None:
                failed += 1
            else:
                index[rec_id] = info
            if n % 500 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index))
    print(f"rendered {len(todo) - failed}, failed {failed}, in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
