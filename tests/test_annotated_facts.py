"""Test scoring against structured human-reviewed facts."""

import pytest

from bitikocr.train.annotated_facts import (
    normalize_fact_text,
    score_annotated_facts,
    substring_similarity,
)


def test_normalization_and_substring_similarity() -> None:
    """Match across case, punctuation, whitespace, and a small typo."""
    assert normalize_fact_text(" № С-1025\n") == "no с 1025"
    assert (
        substring_similarity("Мадаминова Гулнора", "Мадаминова\nГулнора") == 1
    )
    assert substring_similarity("Гулнора", "Гулнара") == pytest.approx(6 / 7)


def test_score_annotated_facts() -> None:
    """Apply exact and fuzzy rules and aggregate each category."""
    rows = [{"id": "doc_1", "prediction": "Ариза № 653 Гулнара"}]
    annotations = {
        "doc_1": [
            {"category": "doc_type", "value": "ариза", "fuzzy": True},
            {"category": "number", "value": "№ 658", "fuzzy": False},
            {"category": "name", "value": "Гулнора", "fuzzy": True},
        ]
    }
    metrics, details = score_annotated_facts(rows, annotations)
    assert metrics["all_accuracy"] == pytest.approx(2 / 3)
    assert metrics["name_accuracy"] == 1
    assert metrics["number_accuracy"] == 0
    assert metrics["document_accuracy"] == 0
    assert [detail["matched"] for detail in details] == [True, False, True]


def test_missing_annotation_is_rejected() -> None:
    """Never silently omit a prediction from fact scoring."""
    with pytest.raises(ValueError, match="Missing fact annotations"):
        score_annotated_facts([{"id": "missing", "prediction": ""}], {})
