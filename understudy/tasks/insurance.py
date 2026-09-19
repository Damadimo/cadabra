"""Commercial-insurance submission triage (default task).

REPLACE the placeholder GUIDELINES with the appetite guide that ships with the sponsor dataset, and keep
`apply_rules` in sync with it: the `decision_follows_rules` check re-derives the decision from the extracted
fields, which is what lets us reject bad teacher labels and escalate bad specialist answers automatically.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..checks import grounded_number, grounded_str, grounded_value, grounded_years, norm
from .base import Check, TaskSpec

GUIDELINES = """\
G1 Decline: NAICS code in an excluded class (prefixes 2111 oil & gas extraction, 3251 basic chemical manufacturing, 7132 gambling).
G2 Refer: total insured value (TIV) above $50,000,000.
G3 Refer: more than 3 claims in the past 5 years.
G4 Decline: fewer than 2 years in business.
G5 Refer: a required field is missing (insured name, NAICS code, state, annual revenue, TIV).
G6 Refer: sources disagree on a field and nothing in the packet settles it.
G7 Accept: none of the above apply."""

EXCLUDED_NAICS = ("2111", "3251", "7132")
TIV_LIMIT = 50_000_000
MAX_CLAIMS = 3
MIN_YEARS = 2
REQUIRED = ("insured_name", "naics_code", "state", "annual_revenue_usd", "total_insured_value_usd")

INSTRUCTIONS = f"""You are an underwriting assistant. Read a messy commercial-insurance submission packet \
(broker emails, form excerpts, statements of values, loss runs) and triage it.

Take every field from the packet only; use null when the packet doesn't state it. When sources disagree, record a \
conflict, and set resolved_value only if the packet itself settles it (for example a later correction). Then decide \
with these appetite guidelines. Declines take precedence over refers. Start every reason with the guideline ID it \
applies, e.g. "G2: TIV $62M exceeds $50M".
{GUIDELINES}"""


class Conflict(BaseModel):
    field_name: str
    values: list[str] = Field(description="Each conflicting value exactly as written in the packet.")
    resolved_value: Optional[str] = Field(description="The value the packet itself settles on, or null.")


class Triage(BaseModel):
    rationale: str = Field(description="One sentence on how you decided. Written first.")
    insured_name: Optional[str]
    naics_code: Optional[str] = Field(description="6-digit NAICS code as written in the packet.")
    state: Optional[str] = Field(description="2-letter US state of the insured's main location.")
    annual_revenue_usd: Optional[float]
    total_insured_value_usd: Optional[float]
    years_in_business: Optional[float]
    prior_claims_5y: Optional[int] = Field(description="Number of claims in the past 5 years.")
    lines_requested: list[str] = Field(description="Coverage lines requested, e.g. General Liability, Property.")
    missing_fields: list[str] = Field(description=f"Which of these required fields are null: {', '.join(REQUIRED)}.")
    conflicts: list[Conflict]
    decision: Literal["accept", "refer", "decline"]
    reasons: list[str]


def apply_rules(o: Triage) -> tuple[str, set[str]]:
    """The guidelines as code. Returns (decision, guideline IDs that fired)."""
    if o.naics_code and o.naics_code.strip().startswith(EXCLUDED_NAICS):
        return "decline", {"G1"}
    if o.years_in_business is not None and o.years_in_business < MIN_YEARS:
        return "decline", {"G4"}
    refer = set()
    if any(getattr(o, f) is None for f in REQUIRED):
        refer.add("G5")
    if any(c.resolved_value is None for c in o.conflicts):
        refer.add("G6")
    if o.total_insured_value_usd is not None and o.total_insured_value_usd > TIV_LIMIT:
        refer.add("G2")
    if o.prior_claims_5y is not None and o.prior_claims_5y > MAX_CLAIMS:
        refer.add("G3")
    return ("refer", refer) if refer else ("accept", {"G7"})


STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california", "CO": "colorado",
    "CT": "connecticut", "DE": "delaware", "DC": "district of columbia", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
    "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland", "MA": "massachusetts",
    "MI": "michigan", "MN": "minnesota", "MS": "mississippi", "MO": "missouri", "MT": "montana",
    "NE": "nebraska", "NV": "nevada", "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico",
    "NY": "new york", "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma",
    "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina", "SD": "south dakota",
    "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington",
    "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming",
}


def _state_grounded(code: str | None, text: str) -> bool:
    if code is None:
        return True
    code = code.strip().upper()
    if code not in STATES:
        return False
    # Match the code case-sensitively so words like "in" or "or" don't count as Indiana or Oregon.
    return bool(re.search(rf"\b{code}\b", text)) or STATES[code] in norm(text)


def _values_grounded(record: dict, o: Triage) -> bool:
    text = record["text"]
    return (
        grounded_str(o.insured_name, text)
        and grounded_str(o.naics_code, text)
        and _state_grounded(o.state, text)
        and grounded_number(o.annual_revenue_usd, text)
        and grounded_number(o.total_insured_value_usd, text)
        and grounded_years(o.years_in_business, text)
    )


def _field_key(s: str) -> str:
    return re.sub(r"[\s-]+", "_", s.strip().lower())


def _missing_fields_consistent(record: dict, o: Triage) -> bool:
    return {_field_key(f) for f in o.missing_fields} == {f for f in REQUIRED if getattr(o, f) is None}


def _decision_follows_rules(record: dict, o: Triage) -> bool:
    return apply_rules(o)[0] == o.decision


def _reasons_cite_rules(record: dict, o: Triage) -> bool:
    cited = set()
    for reason in o.reasons:
        m = re.match(r"\s*(G[1-7])\b", reason)
        if not m:
            return False
        cited.add(m.group(1))
    return bool(cited) and cited >= apply_rules(o)[1]


def _conflicts_grounded(record: dict, o: Triage) -> bool:
    return all(grounded_value(v, record["text"]) for c in o.conflicts for v in c.values)


def _rationale_brief(record: dict, o: Triage) -> bool:
    return 0 < len(o.rationale.strip()) <= 300


CHECKS = [
    Check("values_grounded", _values_grounded, "Every extracted value appears in the packet"),
    Check("missing_fields_consistent", _missing_fields_consistent, "missing_fields lists exactly the null required fields"),
    Check("decision_follows_rules", _decision_follows_rules, "The decision re-derived from the fields matches"),
    Check("reasons_cite_rules", _reasons_cite_rules, "Every reason cites a guideline, including the one that fired"),
    Check("conflicts_grounded", _conflicts_grounded, "Every conflicting value appears in the packet"),
    Check("rationale_brief", _rationale_brief, "One short rationale"),
]

EXAMPLE = {
    "rationale": "All required fields are present and consistent, and no guideline triggers a refer or decline.",
    "insured_name": "Brightline Bakery LLC",
    "naics_code": "311811",
    "state": "OH",
    "annual_revenue_usd": 2400000,
    "total_insured_value_usd": 3100000,
    "years_in_business": 12,
    "prior_claims_5y": 1,
    "lines_requested": ["Commercial General Liability", "Commercial Property"],
    "missing_fields": [],
    "conflicts": [],
    "decision": "accept",
    "reasons": ["G7: within appetite (TIV $3.1M, 1 claim in 5 years, 12 years in business)."],
}

TASK = TaskSpec(
    name="insurance",
    instructions=INSTRUCTIONS,
    output_model=Triage,
    checks=CHECKS,
    render_input=lambda record: "Submission packet:\n\n" + record["text"],
    score_fields=[
        "insured_name",
        "naics_code",
        "state",
        "annual_revenue_usd",
        "total_insured_value_usd",
        "years_in_business",
        "prior_claims_5y",
    ],
    decision_field="decision",
    example=EXAMPLE,
)
