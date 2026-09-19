"""The one prompt every lane gets: frontier baselines, the base model and our fine-tune.

The conventions block spells out how these specs are written, so no model loses points for not knowing them.
"""

from __future__ import annotations

import re

SYSTEM = """You are an expert mechanical CAD engineer. Write CadQuery (Python) code that builds exactly the part described.

How the descriptions are written:
- All numbers share one unitless scale. Build everything at exactly the stated sizes.
- Sketch coordinates are already at final size. A sentence such as "apply a scale factor of 0.75 to the sketch" restates the sketch's overall size; do not scale the coordinates again. The dimensions stated for each part are authoritative.
- Each part begins with a coordinate system: Euler angles in degrees, then a translation vector. Draw that part's sketch on the XY plane of this coordinate system, then place it.
- "Extrude along the normal by d" means extrude d along the sketch plane's normal; "opposite direction" means the negative normal. A part that "adds material" or makes a "new body" is unioned with the model so far; one that "removes material" or "cuts" is subtracted.
- A loop drawn inside another loop on the same face is a hole through that face.

Output rules:
- Reply with the complete program in one ```python code block and nothing else.
- Use `import cadquery as cq` (and `math` if needed); import nothing else.
- Assign the final solid to a variable named `r`. Do not export files or call show_object."""

REPAIR = "Your code failed: {error}\nReply with the complete corrected program in one ```python code block."

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def messages(spec: str, shots: list[dict] | None = None) -> list[dict]:
    """Chat messages for one spec; `shots` are {"spec", "gold_code"} worked examples shown as prior turns."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for shot in shots or []:
        msgs.append({"role": "user", "content": shot["spec"]})
        msgs.append({"role": "assistant", "content": f"```python\n{shot['gold_code'].strip()}\n```"})
    msgs.append({"role": "user", "content": spec})
    return msgs


def extract_code(text: str) -> str | None:
    """The last fenced block that looks like CadQuery; falls back to a bare program."""
    text = _THINK.sub("", text)
    blocks = [b for b in _FENCE.findall(text) if "cq" in b or "cadquery" in b]
    if blocks:
        return blocks[-1].strip() + "\n"
    if "import cadquery" in text:
        return text[text.index("import cadquery"):].strip() + "\n"
    return None


def completion(code: str) -> str:
    """Training target: exactly the format we ask for at inference."""
    return f"```python\n{code.strip()}\n```"
