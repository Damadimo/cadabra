"""Freeze the demo into a folder of files any static host can serve.

  uv run python -m uvicorn app.server:app --port 8000 &     # the exporter reads from a running demo
  uv run python scripts/export_static.py                    # -> web/

The published build has no Python behind it, so there is nothing to keep an API key in and nothing to run CadQuery:
every solid is built once here and written out as STL. The Compare view is therefore the real thing, and the race
replays each part's recorded answers at their measured speed instead of calling the models live.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from cadabra.config import ROOT

API = "http://127.0.0.1:8000"


def get(path: str) -> bytes:
    with urllib.request.urlopen(API + path) as f:
        return f.read()


def try_get(path: str) -> bytes | None:
    try:
        return get(path)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def safe(rec_id: str) -> str:
    return rec_id.replace(":", "_").replace("/", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "web")
    ap.add_argument("--limit", type=int, default=0, help="export only the first N parts (for a quick check)")
    args = ap.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    for sub in ("data/compare", "data/mesh", "data/sheet"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    parts = json.loads(get("/api/parts"))
    if args.limit:
        parts = parts[: args.limit]
    (out / "data/scoreboard.json").write_bytes(get("/api/scoreboard"))

    meshes: set[str] = set()
    skipped: list[str] = []
    kept: list[dict] = []
    lanes: dict[str, dict] = {}
    examples = []
    for n, part in enumerate(parts, 1):
        rec_id = part["id"]
        raw = try_get(f"/api/compare/{urllib.parse.quote(rec_id)}")
        sheet_png = try_get(f"/api/sheet/{urllib.parse.quote(rec_id)}")
        if raw is None or sheet_png is None:
            skipped.append(rec_id)
            continue
        payload = json.loads(raw)
        payload["sheet"] = f"data/sheet/{safe(rec_id)}.png"   # no /api/sheet to serve it
        for lane in payload["lanes"]:
            lanes.setdefault(lane["key"], {
                "key": lane["key"], "label": lane["label"], "model": lane["model"],
                "effort": "high" if "@high" in lane["lane"] else None, "dedicated": bool(lane["ours"]),
            })
        # Pull this part's solids now: the demo keeps only its last few hundred in memory, so a second pass at the
        # end would ask for keys it has already evicted.
        for key in [payload["reference"].get("mesh")] + [l.get("mesh") for l in payload["lanes"]]:
            if not key or key in meshes:
                continue
            stl = try_get(f"/api/mesh/{key}")
            if stl is None:
                continue
            (out / "data/mesh" / f"{key}.stl").write_bytes(stl)
            meshes.add(key)
        (out / "data/compare" / f"{safe(rec_id)}.json").write_bytes(json.dumps(payload).encode())
        (out / "data/sheet" / f"{safe(rec_id)}.png").write_bytes(sheet_png)
        kept.append(part)
        examples.append({"id": rec_id, "title": payload["title"], "n_parts": payload["n_parts"],
                         "n_faces": payload["tier_faces"], "spec": payload.get("spec"),
                         "sheet": payload["sheet"], "bbox": payload.get("bbox")})
        if n % 50 == 0 or n == len(parts):
            print(f"  {n}/{len(parts)} parts, {len(meshes)} solids", flush=True)

    (out / "data/parts.json").write_bytes(json.dumps(kept).encode())
    if skipped:
        print(f"  skipped {len(skipped)} parts with no sheet or no saved answers: {skipped[:5]}")

    # The race lanes have to be the ones the saved answers carry, or a replay has no card to write into.
    (out / "data/config.json").write_bytes(json.dumps(
        {"lanes": list(lanes.values()), "default_modality": "image"}).encode())
    (out / "data/examples.json").write_bytes(json.dumps(examples).encode())

    # The page, with the API swapped for the files above. One source of truth: app/static/index.html drives both.
    src = (ROOT / "app" / "static" / "index.html").read_text()
    shim = """<script>
// Static build: there is no API, so point the page's own calls at the files exported beside it.
window.CADABRA_STATIC = true;
const _fetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url = typeof input === "string" ? input : input.url;
  const m = /^\\/api\\/(.*)$/.exec(url || "");
  if (!m) return _fetch(input, init);
  const rest = m[1];
  const one = (p) => _fetch("data/" + p, init);
  if (rest === "parts") return one("parts.json");
  if (rest === "scoreboard") return one("scoreboard.json");
  if (rest === "config") return one("config.json");
  if (rest === "examples") return one("examples.json");
  if (rest.startsWith("compare/")) return one("compare/" + decodeURIComponent(rest.slice(8)).replace(/[:/]/g, "_") + ".json");
  if (rest.startsWith("mesh/")) return one("mesh/" + rest.slice(5) + ".stl");
  if (rest.startsWith("sheet/")) return one("sheet/" + decodeURIComponent(rest.slice(6)).replace(/[:/]/g, "_") + ".png");
  return Promise.resolve(new Response("not in the static build", { status: 404 }));
};
</script>
"""
    marker = '<script type="importmap">'
    assert marker in src, "importmap not found in index.html"
    src = src.replace(marker, shim + marker, 1)
    # everything is served from one folder here, so the module map cannot point at the app's /static mount
    assert "/static/vendor/" in src, "vendor paths not found in index.html"
    src = src.replace("/static/vendor/", "./vendor/")
    src = src.replace('<select id="modality"><option value="image">Input: drawing sheet</option>'
                      '<option value="text">Input: written spec</option></select>',
                      '<select id="modality"><option value="image">Input: drawing sheet</option></select>', 1)
    src = src.replace("Every model gets the same sheet and bounding box. Parts are held out.",
                      "Answers recorded in the benchmark run, replayed at their measured speed. Parts are held out.", 1)
    (out / "index.html").write_text(src)
    shutil.copytree(ROOT / "app" / "static" / "vendor", out / "vendor")

    # Static host config: no build step, and the exported data is content-addressed or part-keyed, so cache it hard.
    (out / "vercel.json").write_text(json.dumps({
        "cleanUrls": True,
        "headers": [{"source": "/data/mesh/(.*)",
                     "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}]},
                    {"source": "/vendor/(.*)",
                     "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}]}],
    }, indent=2) + "\n")

    files = sum(1 for _ in out.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n{out}: {files} files, {size / 1e6:.1f} MB, {len(kept)} parts, {len(meshes)} solids")


if __name__ == "__main__":
    main()
