"""Tests for choosing the model a full run trains or evaluates."""

from scripts.run_full_real_eval import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    model_revision,
    parse_args,
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
