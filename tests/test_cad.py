"""Checks for the parts a judge would poke at: the geometry grader, the sandbox, the prompt, and leakage."""

from pathlib import Path

import pytest

from cadabra.cad import geometry as g
from cadabra.cad import prompts
from cadabra.data import read_jsonl
from cadabra.stats import bootstrap_ci, pct

ROOT = Path(__file__).resolve().parent.parent
PLATE = "import cadquery as cq\nr = cq.Workplane('XY').box(0.6, 0.375, 0.075, centered=False)\n"


def test_reference_scores_itself_perfectly():
    res = g.evaluate(PLATE, PLATE)
    assert res["runs"] and res["iou"] == pytest.approx(1.0) and res["iou_aligned"] == pytest.approx(1.0)


def test_aligned_iou_ignores_placement_and_axis_rotation_but_not_size():
    moved = PLATE + "r = r.rotate((0, 0, 0), (1, 0, 0), 90).translate((3, -1, 2))\n"
    res = g.evaluate(moved, PLATE)
    assert res["iou"] < 0.01 and res["iou_aligned"] == pytest.approx(1.0, abs=1e-6)
    thicker = PLATE.replace("0.075", "0.09")
    assert g.evaluate(thicker, PLATE)["iou_aligned"] == pytest.approx(0.075 / 0.09, abs=1e-3)
    bigger = PLATE + "r = r.val().scale(1.1)\n"
    assert g.evaluate(bigger, PLATE)["iou_aligned"] < 0.8


def test_sandbox_blocks_imports_dunders_and_file_io():
    assert "not allowed" in g.evaluate("import os\nr = os.listdir('.')")["error"]
    assert "not allowed" in g.evaluate("r = ().__class__.__bases__")["error"]
    assert "not allowed" in g.evaluate("r = open('x', 'w')")["error"]
    sneaky = PLATE + "f = r.val().exportStl\n"
    assert "not allowed" in g.evaluate(sneaky)["error"]


def test_export_lines_are_stripped_not_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = g.evaluate(PLATE + "cq.exporters.export(r, 'leak.stl')\nshow_object(r)\n")
    assert res["runs"] and not list(tmp_path.iterdir())


def test_failures_are_reported():
    assert g.evaluate("import cadquery as cq\nr = cq.Workplane('XY')\n")["error"].startswith("no solid")
    assert "SyntaxError" in g.evaluate("r = (")["error"]


def test_prompt_and_code_extraction():
    msgs = prompts.messages("a spec", [{"spec": "s", "gold_code": PLATE}])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    reply = "<think>plan</think>Sure:\n```python\n" + PLATE + "```\nDone."
    assert prompts.extract_code(reply).strip() == PLATE.strip()
    assert prompts.extract_code("no code here") is None
    assert prompts.extract_code(prompts.completion(PLATE)).strip() == PLATE.strip()


@pytest.mark.skipif(not (ROOT / "data/cad/bench.jsonl").exists(), reason="splits not built")
def test_bench_and_train_share_no_source_part():
    bench = read_jsonl(ROOT / "data/cad/bench.jsonl")
    train_path = ROOT / "data/cad/train.jsonl"
    if not train_path.exists():
        pytest.skip("train split not built")
    train = read_jsonl(train_path)
    assert not ({r["source_id"] for r in bench} & {r["source_id"] for r in train})
    shots = {r["id"] for r in read_jsonl(ROOT / "data/cad/shots.jsonl")}
    assert not (shots & {r["id"] for r in bench})


def test_stats():
    assert pct([1, 2, 3, 4], 50) == 2.5
    lo, hi = bootstrap_ci([1.0] * 8 + [0.0] * 2)
    assert 0 <= lo <= 0.8 <= hi <= 1
