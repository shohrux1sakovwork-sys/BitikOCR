"""Tests for the corpus schema's own record types.

How a generated page is described *in* this schema is covered beside the
generator, in bitikocr/data/synthetic/tests/test_export.py.
"""

from __future__ import annotations

from typing import Any

from bitikocr.models.schema import (
    AnnotationInfo,
    DocumentMetadata,
    Fact,
    Part,
    QualityInfo,
    SourceInfo,
)


def test_unknown_provenance_is_left_out_rather_than_nulled() -> None:
    payload = SourceInfo(
        origin="real", collection="archive_batch_03", era="modern"
    ).to_dict()
    assert "seed" not in payload
    assert payload["year_approx"] is None


def test_a_part_without_lines_omits_the_key() -> None:
    assert "lines" not in Part(role="body", text="x").to_dict()


def test_a_fact_without_a_subtype_omits_the_key() -> None:
    payload = Fact(category="place", value="x", evidence_text="x").to_dict()
    assert "subtype" not in payload


def test_a_default_annotation_is_unreviewed() -> None:
    payload = AnnotationInfo().to_dict()
    assert payload["status"] == "auto"
    assert payload["revision"] == 1


def test_document_metadata_reports_every_content_flag() -> None:
    payload: dict[str, Any] = DocumentMetadata(
        language=["uz"],
        scripts=["cyrillic"],
        primary_script="cyrillic",
        document_type="ariza",
        text_mode="handwritten",
        layout="single_column",
        quality=QualityInfo(),
    ).to_dict()
    for flag in (
        "has_table",
        "has_formula",
        "has_diagram",
        "has_handwriting",
        "has_printed_text",
        "has_stamp",
        "has_signature",
    ):
        assert flag in payload
