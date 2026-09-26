"""Tests for choosing the model a full run trains or evaluates."""

import json
from pathlib import Path

from scripts.run_full_real_eval import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    model_revision,
    parse_args,
    prepare_real_eval,
)


def test_the_default_model_keeps_its_pinned_snapshot() -> None:
    options = parse_args([])
    assert options.model == DEFAULT_MODEL
    assert model_revision(options) == DEFAULT_MODEL_REVISION
    assert not options.evaluation_only


def test_another_model_loads_its_latest_snapshot() -> None:
    options = parse_args(
        ["--model", "Qwen/Qwen3-VL-8B-Instruct", "--evaluation-only"]
    )
    assert model_revision(options) is None
    assert options.evaluation_only


def test_an_explicit_revision_wins() -> None:
    options = parse_args(["--model-revision", "abc123"])
    assert model_revision(options) == "abc123"


def test_a_downloaded_benchmark_needs_no_index(tmp_path: Path) -> None:
    (tmp_path / "annotations").mkdir()
    (tmp_path / "facts").mkdir()
    annotation = {
        "id": "doc_000001",
        "image": "images/doc_000001.jpg",
        "text": "Ариза",
        "metadata": {
            "document_type": "ariza",
            "scripts": ["latin", "cyrillic"],
        },
    }
    (tmp_path / "annotations/doc_000001.json").write_text(
        json.dumps(annotation)
    )
    facts = {
        "id": "doc_000001",
        "facts": [{"kind": "doc_type", "value": "ariza"}],
    }
    (tmp_path / "facts/doc_000001.json").write_text(json.dumps(facts))

    output = tmp_path / "eval.jsonl"
    assert prepare_real_eval(output, tmp_path) == 1
    row = json.loads(output.read_text())
    assert row["script"] == "cyrillic+latin"
    assert row["facts"] == facts["facts"]
