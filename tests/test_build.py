"""Tests for bitikocr.data.synthetic.build and .validate."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic.augment import AugmentationProfile
from bitikocr.data.synthetic.build import (
    CorpusSpec,
    PlannedDocument,
    build_corpus,
    pending_documents,
    plan_corpus,
    read_plan,
    record_fingerprint,
    templates_for,
    write_plan,
)
from bitikocr.data.synthetic.validate import check_corpus


@pytest.fixture(scope="module")
def plan(config: SyntheticConfig) -> list[PlannedDocument]:
    """A small plan over every document type."""
    spec = CorpusSpec(
        counts={
            "ariza": 3,
            "birth_certificate": 2,
            "consent_letter": 2,
            "death_certificate": 4,
            "explanatory_letter": 2,
        },
        seed=5,
        profile=AugmentationProfile.varied(),
    )
    return plan_corpus(spec, config)


def test_a_plan_never_repeats_content(plan: list[PlannedDocument]) -> None:
    fingerprints = [entry.fingerprint for entry in plan]
    assert len(set(fingerprints)) == len(plan)
    assert all(
        entry.fingerprint == record_fingerprint(entry.record) for entry in plan
    )


def test_a_plan_gives_every_page_its_own_id(
    plan: list[PlannedDocument],
) -> None:
    assert len({entry.id for entry in plan}) == len(plan)
    assert plan[0].id == "ariza_000000"


def test_a_plan_is_reproducible(
    plan: list[PlannedDocument], config: SyntheticConfig
) -> None:
    spec = CorpusSpec(
        counts={
            "ariza": 3,
            "birth_certificate": 2,
            "consent_letter": 2,
            "death_certificate": 4,
            "explanatory_letter": 2,
        },
        seed=5,
    )
    again = plan_corpus(spec, config)
    assert [e.to_dict() for e in again] == [e.to_dict() for e in plan]


def test_a_plan_spreads_pages_across_form_variants(
    plan: list[PlannedDocument], config: SyntheticConfig
) -> None:
    variants = templates_for("death_certificate", config)
    assert len(variants) == 2
    used = Counter(
        entry.template
        for entry in plan
        if entry.record.document_type == "death_certificate"
    )
    assert set(used) == set(variants)
    assert set(used.values()) == {2}


def test_a_cyrillic_only_form_is_filled_in_cyrillic(
    plan: list[PlannedDocument],
) -> None:
    for entry in plan:
        if entry.template == "death_certificate_cyrillic_single":
            assert entry.record.script == "cyrillic"


def test_letters_use_no_template(config: SyntheticConfig) -> None:
    assert templates_for("ariza", config) == (None,)


def test_a_plan_round_trips_through_its_file(
    plan: list[PlannedDocument], tmp_path: Path
) -> None:
    path = tmp_path / "plan.jsonl"
    write_plan(plan, path)
    assert [e.to_dict() for e in read_plan(path)] == [e.to_dict() for e in plan]


def test_a_built_corpus_passes_its_checks_and_resumes(
    plan: list[PlannedDocument], config: SyntheticConfig, tmp_path: Path
) -> None:
    subset = [plan[0], plan[4], plan[8]]
    first = build_corpus(subset[:2], tmp_path, config, workers=1)
    assert first.done == 2 and not first.failed
    assert pending_documents(subset, tmp_path) == [subset[2]]

    second = build_corpus(subset, tmp_path, config, workers=1)
    assert (second.done, second.skipped) == (1, 2)

    report = check_corpus(tmp_path)
    assert report.pages == 3
    assert report.ok, report.problems
    index = (tmp_path / "index.jsonl").read_text(encoding="utf-8")
    assert [json.loads(line)["id"] for line in index.splitlines()] == sorted(
        entry.id for entry in subset
    )


def test_the_checker_catches_a_box_that_misses_its_outline(
    plan: list[PlannedDocument], config: SyntheticConfig, tmp_path: Path
) -> None:
    build_corpus([plan[0]], tmp_path, config, workers=1)
    path = tmp_path / "annotations" / f"{plan[0].id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["target"]["parts"][0]["bbox"][0] += 5
    record["metadata"]["quality"]["capture"] = "fax"
    path.write_text(json.dumps(record), encoding="utf-8")

    problems = [problem for _, problem in check_corpus(tmp_path).problems]
    assert any("bbox" in problem for problem in problems)
    assert any("capture" in problem for problem in problems)
