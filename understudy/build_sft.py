"""Turn accepted teacher labels and human corrections into Baseten training data.

  uv run python -m understudy.build_sft --gold data/gold/gold.jsonl

Writes training/data/{train,val}.jsonl in TRL's conversational prompt/completion format, so the loss is on
the answer only. Any source document that appears in the gold set is dropped (leakage guard), and prompts use
exactly the messages the eval and the race send.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .config import ROOT
from .data import read_jsonl, source_of, split_by_source, write_jsonl
from .tasks import TaskSpec, load_task


def example(task: TaskSpec, record: dict, label: dict) -> dict:
    answer = task.output_model.model_validate(label).model_dump_json()  # field order puts rationale first
    return {
        "id": record["id"],
        "source_id": source_of(record),
        "prompt": task.messages(record),
        "completion": [{"role": "assistant", "content": answer}],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels/teacher.jsonl")
    ap.add_argument("--corrections", default="data/corrections.jsonl")
    ap.add_argument("--gold", default="data/gold/gold.jsonl")
    ap.add_argument("--out", default=str(ROOT / "training" / "data"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-chars", type=int, default=13000,
                    help="drop examples (system + input + answer) that could overflow MAX_LEN=4096 and get the answer truncated")
    ap.add_argument("--correction-weight", type=int, default=2)
    ap.add_argument("--format", choices=["prompt-completion", "messages"], default="prompt-completion")
    args = ap.parse_args()

    task = load_task()
    gold_sources = {source_of(r) for r in read_jsonl(args.gold)}
    if not gold_sources:
        print(f"WARNING: no gold set at {args.gold}; can't guard against train/test leakage")

    rows, dropped = [], Counter()
    for row in read_jsonl(args.labels):
        record = row["record"]
        if row.get("status") != "accepted":
            dropped["not accepted"] += 1
        elif source_of(record) in gold_sources:
            dropped["source is in the gold set"] += 1
        else:
            ex = example(task, record, row["label"])
            if sum(len(m["content"]) for m in ex["prompt"] + ex["completion"]) > args.max_chars:
                dropped["too long for MAX_LEN"] += 1
            else:
                rows.append(ex)
    for c in read_jsonl(args.corrections):
        if source_of(c["record"]) in gold_sources:
            dropped["correction source is in the gold set"] += 1
            continue
        rows += [example(task, c["record"], c["label"])] * args.correction_weight

    train, val = split_by_source(rows, args.val_frac)
    if args.format == "messages":
        shape = lambda r: {"messages": r["prompt"] + r["completion"]}  # noqa: E731
    else:
        shape = lambda r: {"prompt": r["prompt"], "completion": r["completion"]}  # noqa: E731
    train, val = [shape(r) for r in train], [shape(r) for r in val]
    write_jsonl(f"{args.out}/train.jsonl", train)
    write_jsonl(f"{args.out}/val.jsonl", val)
    stats = {"train": len(train), "val": len(val), "dropped": dict(dropped), "format": args.format}
    (Path(args.out) / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
