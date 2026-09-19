from pathlib import Path

import pytest

from understudy.checks import extract_json, grounded_number, grounded_years, numbers_in, same
from understudy.data import perturb, read_jsonl, split_by_source
from understudy.stats import bootstrap_ci, pct
from understudy.tasks import load_task

ROOT = Path(__file__).resolve().parent.parent


def test_numbers_in_handles_money_suffixes():
    assert 2.4e6 in numbers_in("revenue about $2.4M last FY")
    assert 62e6 in numbers_in("TOTAL TIV ............ $62,000,000")
    assert 1.5 in numbers_in("about 1.5 years in business")
    assert 240e3 in numbers_in("ammonia leak $240k;")
    assert 41e6 in numbers_in("rev approx $41M;")
    assert 1110 not in numbers_in("technique T1110")


def test_grounding():
    assert grounded_number(16.5e6, "please use $16.5M")
    assert not grounded_number(17e6, "please use $16.5M")
    assert grounded_years(12, "operating since 2014")
    assert not grounded_years(30, "operating since 2014")


def test_extract_json_strips_think_block_and_fences():
    assert extract_json('<think>\n\n</think>\n\n{"a": 1}') == '{"a": 1}'
    assert extract_json('```json\n{"a": {"b": 2}}\n```') == '{"a": {"b": 2}}'
    assert extract_json("no json here") is None


@pytest.mark.parametrize("task_name", ["insurance", "seclogs"])
def test_reference_labels_pass_every_check_and_match_gold(task_name):
    task = load_task(task_name)
    rows = read_jsonl(ROOT / "data" / "samples" / f"{task_name}_gold_SAMPLE.jsonl")
    assert rows
    for row in rows:
        output = task.output_model.model_validate(row["label"])
        checks = task.run_checks(row, output)
        assert all(checks.values()), (row["id"], checks)
        assert all(task.field_matches(output, row["gold"]).values()), row["id"]
        assert same(getattr(output, task.decision_field), row["gold"][task.decision_field])


def test_checks_catch_hallucinated_value_and_wrong_decision():
    task = load_task("insurance")
    row = read_jsonl(ROOT / "data" / "samples" / "insurance_gold_SAMPLE.jsonl")[0]
    hallucinated = task.output_model.model_validate({**row["label"], "total_insured_value_usd": 9_999_999})
    assert not task.run_checks(row, hallucinated)["values_grounded"]
    wrong = task.output_model.model_validate({**row["label"], "decision": "decline"})
    assert not task.run_checks(row, wrong)["decision_follows_rules"]


def test_parse_accepts_qwen_output_and_rejects_bad_schema():
    task = load_task("insurance")
    row = read_jsonl(ROOT / "data" / "samples" / "insurance_gold_SAMPLE.jsonl")[1]
    output, error = task.parse("<think>\n\n</think>\n\n" + task.output_model.model_validate(row["label"]).model_dump_json())
    assert error is None and output.decision == "refer"
    output, error = task.parse('{"rationale": "x"}')
    assert output is None and error.startswith("schema")


def test_strict_schema_closes_every_object():
    schema = load_task("insurance").json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    conflict = schema["$defs"]["Conflict"]
    assert conflict["additionalProperties"] is False


def test_split_keeps_sources_together():
    rows = [{"id": f"{s}-{k}", "source_id": s} for s in "abcdefghij" for k in range(3)]
    train, test = split_by_source(rows, 0.3)
    assert {r["source_id"] for r in train}.isdisjoint({r["source_id"] for r in test})
    assert len(train) + len(test) == len(rows) and test


def test_perturb_is_deterministic_and_keeps_source():
    row = {"id": "x", "source_id": "src", "text": "Hello brave world\n\nSecond block here\n\nThird block"}
    a, b = perturb(row, 0, seed=1), perturb(row, 0, seed=1)
    assert a == b and a["id"] == "x~p0" and a["source_id"] == "src"
    logs = {"id": "y", "lines": ["Failed password for root", "Accepted password for deploy"]}
    assert len(perturb(logs, 0)["lines"]) == 2


def test_stats():
    assert pct([1, 2, 3, 4], 50) == 2.5
    lo, hi = bootstrap_ci([1.0] * 8 + [0.0] * 2)
    assert 0 <= lo <= 0.8 <= hi <= 1
