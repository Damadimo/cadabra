"""A task is an output schema, a prompt and deterministic checks. Add a module here to switch domains."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from ..checks import extract_json, same


@dataclass(frozen=True)
class Check:
    name: str
    fn: Callable[[dict, Any], bool]  # (input record, parsed output) -> pass?
    description: str = ""


@dataclass
class TaskSpec:
    name: str
    instructions: str
    output_model: type[BaseModel]
    checks: list[Check]
    render_input: Callable[[dict], str]
    score_fields: list[str]
    decision_field: str | None = None
    example: dict = field(default_factory=dict)  # one valid output, used by the mock server

    def system_prompt(self) -> str:
        schema = json.dumps(self.output_model.model_json_schema(), separators=(",", ":"))
        return (
            f"{self.instructions}\n\n"
            "Respond with exactly one JSON object that matches this JSON Schema. No prose, no code fences. "
            "Write `rationale` first: one sentence on how you decided.\n"
            f"{schema}"
        )

    def messages(self, record: dict) -> list[dict]:
        # Same prompt for teacher, baselines and the fine-tuned model: training and serving must match.
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.render_input(record)},
        ]

    def json_schema(self) -> dict:
        return strict_schema(self.output_model.model_json_schema())

    def parse(self, text: str) -> tuple[BaseModel | None, str | None]:
        raw = extract_json(text)
        if raw is None:
            return None, "no JSON object in output"
        try:
            return self.output_model.model_validate_json(raw), None
        except ValidationError as e:
            err = e.errors()[0]
            return None, f"schema: {err['msg']} at {'.'.join(map(str, err['loc']))}"

    def run_checks(self, record: dict, output: BaseModel) -> dict[str, bool]:
        results = {}
        for check in self.checks:
            try:
                results[check.name] = bool(check.fn(record, output))
            except Exception:
                results[check.name] = False
        return results

    def field_matches(self, output: BaseModel, gold: dict) -> dict[str, bool]:
        return {f: same(getattr(output, f), gold.get(f)) for f in self.score_fields}

    @property
    def correctable_fields(self) -> list[str]:
        return self.score_fields + ([self.decision_field] if self.decision_field else [])


def strict_schema(schema: dict) -> dict:
    """OpenAI-style strict JSON schema: every object closed, every property required."""

    def fix(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                fix(value)
        elif isinstance(node, list):
            for value in node:
                fix(value)

    fix(schema)
    return schema
