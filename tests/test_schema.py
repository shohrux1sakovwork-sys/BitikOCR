"""Tests for the corpus schema's own record types.

How a generated page is described *in* this schema is covered in
tests/test_export.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from bitikocr.data.models.geometry import polygon_bounds
from bitikocr.data.models.schema import (
    AnnotationInfo,
    DocumentMetadata,
    Fact,
    Part,
    QualityInfo,
    SourceInfo,
    TranscriptionRecord,
    UncertainSpan,
)

#: The region from the schema's own example.
TITLE_POLYGON = ((120, 80), (900, 84), (898, 160), (118, 156))


def record(**overrides: Any) -> TranscriptionRecord:
    """Build a small valid record, overriding whatever a test needs."""
    fields: dict[str, Any] = {
        "id": "doc_000002",
        "image": "images/doc_000002.jpg",
        "image_size": (2480, 3508),
        "source": SourceInfo(origin="real", collection="archive_batch_03"),
        "metadata": DocumentMetadata(
            language=["uz-cyrillic", "ru"],
            document_type="ariza",
            text_mode="mixed",
            layout="single_column",
            quality=QualityInfo(),
        ),
        "text": "Ариза\nЯнгиарик тумани ҳокими М. Саидовга",
        "parts": (Part(role="title", polygon=TITLE_POLYGON, text="Ариза"),),
    }
    fields.update(overrides)
    return TranscriptionRecord(**fields)


# -- the record ------------------------------------------------------------


def test_a_record_has_exactly_the_schema_keys() -> None:
    payload = record().to_dict()
    assert list(payload) == [
        "id",
        "image",
        "image_size",
        "source",
        "metadata",
        "target",
        "annotation",
    ]
    assert payload["image_size"] == [2480, 3508]
    assert set(payload["target"]) == {"text", "parts"}


def test_the_metadata_names_languages_not_alphabets() -> None:
    metadata = record().to_dict()["metadata"]
    assert metadata["language"] == ["uz-cyrillic", "ru"]
    assert "scripts" not in metadata
    assert "primary_script" not in metadata
    for flag in (
        "has_table",
        "has_formula",
        "has_diagram",
        "has_handwriting",
        "has_printed_text",
        "has_stamp",
        "has_signature",
    ):
        assert flag in metadata


@pytest.mark.parametrize("language", ["uz", "en", "cyrillic"])
def test_a_language_outside_the_set_is_refused(language: str) -> None:
    with pytest.raises(ValueError, match="metadata.language"):
        DocumentMetadata(
            language=[language],  # type: ignore[list-item]
            document_type=None,
            text_mode=None,
            layout="single_column",
            quality=QualityInfo(),
        )


def test_what_a_real_scan_may_not_know_can_be_null() -> None:
    payload = record(
        source=SourceInfo(origin="real", collection="batch"),
        metadata=DocumentMetadata(
            language=["uz-latin"],
            document_type=None,
            text_mode=None,
            layout="single_column",
            quality=QualityInfo(),
        ),
        parts=(),
    ).to_dict()
    assert payload["source"]["era"] is None
    assert payload["source"]["year_approx"] is None
    assert payload["metadata"]["document_type"] is None
    assert payload["metadata"]["text_mode"] is None
    assert payload["target"]["parts"] == []


def test_the_source_records_the_file_it_came_from() -> None:
    source = SourceInfo(
        origin="real",
        collection="archive_batch_03",
        era="modern",
        year_approx=2006,
        original_file="scan_17.pdf#page=2",
    ).to_dict()
    assert source == {
        "origin": "real",
        "collection": "archive_batch_03",
        "era": "modern",
        "year_approx": 2006,
        "original_file": "scan_17.pdf#page=2",
    }


def test_an_unknown_origin_is_refused() -> None:
    with pytest.raises(ValueError, match="source.origin"):
        SourceInfo(origin="scanned", collection="x")  # type: ignore[arg-type]


# -- uncertain spans -------------------------------------------------------


def test_uncertain_spans_are_absent_when_there_are_none() -> None:
    assert "uncertain_spans" not in record().to_dict()


def test_an_uncertain_span_is_written_when_there_is_one() -> None:
    span = UncertainSpan(text="Саидовга", note="surname unclear")
    payload = record(uncertain_spans=(span,)).to_dict()
    assert payload["uncertain_spans"] == [
        {"text": "Саидовга", "note": "surname unclear"}
    ]


def test_an_uncertain_span_must_be_part_of_the_transcription() -> None:
    span = UncertainSpan(text="Сапаевга", note="not on this page")
    with pytest.raises(ValueError, match="not in the transcription"):
        record(uncertain_spans=(span,))


# -- parts -----------------------------------------------------------------


def test_a_part_matches_the_schema_example() -> None:
    part = Part(
        role="title", polygon=TITLE_POLYGON, text="...", hand="handwritten"
    )
    assert part.to_dict() == {
        "role": "title",
        "polygon": [[120, 80], [900, 84], [898, 160], [118, 156]],
        "bbox": [118, 80, 782, 80],
        "text": "...",
        "hand": "handwritten",
    }


def test_a_parts_box_always_encloses_its_outline() -> None:
    x, y, width, height = Part(role="body", polygon=TITLE_POLYGON).bbox
    for px, py in TITLE_POLYGON:
        assert x <= px <= x + width
        assert y <= py <= y + height


@pytest.mark.parametrize("role", ["footer", "recipient", "surname", ""])
def test_a_role_outside_the_set_is_refused(role: str) -> None:
    """``footer`` was dropped from the set; field names never belonged."""
    with pytest.raises(ValueError, match="part.role"):
        Part(role=role, polygon=TITLE_POLYGON)  # type: ignore[arg-type]


def test_a_part_may_leave_its_hand_unknown() -> None:
    assert Part(role="other", polygon=TITLE_POLYGON).to_dict()["hand"] is None


def test_a_hand_outside_the_set_is_refused() -> None:
    with pytest.raises(ValueError, match="part.hand"):
        Part(
            role="body", polygon=TITLE_POLYGON, hand="typed"  # type: ignore[arg-type]
        )


def test_an_outline_needs_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3 points"):
        Part(role="body", polygon=((0, 0), (10, 10)))


def test_a_part_off_the_page_is_refused() -> None:
    stray = Part(role="body", polygon=((0, 0), (3000, 0), (3000, 40)))
    with pytest.raises(ValueError, match="lies off"):
        record(parts=(stray,))


def test_the_bounds_of_a_polygon_touch_its_extreme_points() -> None:
    box = polygon_bounds([(5, 9), (1, 4), (7, 2)])
    assert box.to_list() == [1, 2, 7, 9]


# -- the rest --------------------------------------------------------------


def test_a_fact_without_a_subtype_omits_the_key() -> None:
    payload = Fact(category="place", value="x", evidence_text="x").to_dict()
    assert "subtype" not in payload


def test_a_default_annotation_is_unreviewed() -> None:
    payload = AnnotationInfo().to_dict()
    assert payload["status"] == "auto"
    assert payload["revision"] == 1


def test_an_unknown_review_status_is_refused() -> None:
    with pytest.raises(ValueError, match="annotation.status"):
        AnnotationInfo(status="done")  # type: ignore[arg-type]
