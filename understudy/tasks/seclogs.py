"""Security-log triage (fallback task): decide whether a window of log lines shows an attack.

Records carry either "lines" (a list) or "text" (newline-separated). Adapt INSTRUCTIONS and the action
policy to the sponsor's dataset and labels.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .base import Check, TaskSpec

INSTRUCTIONS = """You are a SOC analyst. Read a window of numbered log lines and decide whether it shows an attack.
Cite the evidence line numbers, list indicators of compromise exactly as they appear (IPs, domains, usernames, \
hashes), map malicious or suspicious behaviour to one MITRE ATT&CK technique ID, and pick one action.
Action policy: benign -> none or monitor; suspicious -> monitor or escalate; \
malicious -> block_ip, disable_account, isolate_host or escalate."""

ALLOWED_ACTIONS = {
    "benign": {"none", "monitor"},
    "suspicious": {"monitor", "escalate"},
    "malicious": {"block_ip", "disable_account", "isolate_host", "escalate"},
}


class Finding(BaseModel):
    rationale: str = Field(description="One sentence on how you decided. Written first.")
    verdict: Literal["benign", "suspicious", "malicious"]
    technique_id: Optional[str] = Field(description="MITRE ATT&CK ID like T1110 or T1110.001; null if benign.")
    evidence_lines: list[int] = Field(description="Line numbers (the N in LN) that support the verdict.")
    iocs: list[str] = Field(description="Indicators exactly as they appear in the logs.")
    recommended_action: Literal["none", "monitor", "block_ip", "disable_account", "isolate_host", "escalate"]


def _lines(record: dict) -> list[str]:
    return record.get("lines") or record["text"].splitlines()


def _render(record: dict) -> str:
    return "Log window:\n" + "\n".join(f"L{i}: {line}" for i, line in enumerate(_lines(record), 1))


def _evidence_valid(record: dict, o: Finding) -> bool:
    n = len(_lines(record))
    in_range = all(1 <= x <= n for x in o.evidence_lines)
    return in_range and (o.verdict == "benign" or bool(o.evidence_lines))


def _iocs_grounded(record: dict, o: Finding) -> bool:
    text = "\n".join(_lines(record)).lower()
    return all(ioc.strip() and ioc.strip().lower() in text for ioc in o.iocs)


def _technique_valid(record: dict, o: Finding) -> bool:
    if o.verdict == "benign":
        return o.technique_id is None
    return bool(o.technique_id and re.fullmatch(r"T\d{4}(\.\d{3})?", o.technique_id.strip()))


def _action_matches_policy(record: dict, o: Finding) -> bool:
    return o.recommended_action in ALLOWED_ACTIONS[o.verdict]


def _rationale_brief(record: dict, o: Finding) -> bool:
    return 0 < len(o.rationale.strip()) <= 300


TASK = TaskSpec(
    name="seclogs",
    instructions=INSTRUCTIONS,
    output_model=Finding,
    checks=[
        Check("evidence_valid", _evidence_valid, "Evidence lines exist, and are present unless benign"),
        Check("iocs_grounded", _iocs_grounded, "Every IOC appears in the logs"),
        Check("technique_valid", _technique_valid, "Valid ATT&CK ID when not benign, null when benign"),
        Check("action_matches_policy", _action_matches_policy, "Action allowed for the verdict"),
        Check("rationale_brief", _rationale_brief, "One short rationale"),
    ],
    render_input=_render,
    score_fields=["technique_id", "recommended_action"],
    decision_field="verdict",
    example={
        "rationale": "Repeated failed SSH logins from one IP followed by a success indicate brute force.",
        "verdict": "malicious",
        "technique_id": "T1110",
        "evidence_lines": [1, 2, 3],
        "iocs": ["203.0.113.45"],
        "recommended_action": "block_ip",
    },
)
