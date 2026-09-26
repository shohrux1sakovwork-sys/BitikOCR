"""Evaluate predictions against human-reviewed fact annotations."""

import argparse
import json
from pathlib import Path
from typing import Any

from bitikocr.train.annotated_facts import score_annotated_facts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSON objects from a JSONL file."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"No records found in {path}.")
    return rows


def read_annotations(directory: Path) -> dict[str, list[dict[str, Any]]]:
    """Read fact annotations keyed by document ID."""
    annotations = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        annotations[record["id"]] = record["facts"]
    if not annotations:
        raise ValueError(f"No fact annotations found in {directory}.")
    return annotations


def main() -> None:
    """Write summary and per-fact results next to benchmark predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--facts-dir",
        type=Path,
        default=Path("data/uzbek-htr-bench/facts"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85)
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    metrics, details = score_annotated_facts(
        rows,
        read_annotations(args.facts_dir),
        args.fuzzy_threshold,
    )
    output = args.output or args.predictions.with_name("fact_results.json")
    details_output = args.details_output or args.predictions.with_name(
        "fact_details.jsonl"
    )
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    details_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details)
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
