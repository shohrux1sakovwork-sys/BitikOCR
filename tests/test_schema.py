"""Tests for the corpus schema and the records exported into it."""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic.augment import AugmentationReport
from bitikocr.data.synthetic.export import (
    build_facts_record,
    build_transcription_record,
)
from bitikocr.data.synthetic.facts import (
    FACT_CATEGORIES,
    FIELD_FACTS,
    build_facts,
)
from bitikocr.data.synthetic.generators import create_generator
from bitikocr.data.synthetic.records import (
    available_record_types,
    sample_record,
)
from bitikocr.models.geometry import BoundingBox
from bitikocr.models.schema import (
    AnnotationInfo,
    DocumentMetadata,
    Fact,
    FactsRecord,
    Part,
    QualityInfo,
    SourceInfo,
    TranscriptionRecord,
)


@pytest.fixture(params=available_record_types())
def exported(
    request: pytest.FixtureRequest, config: SyntheticConfig
) -> tuple[TranscriptionRecord, FactsRecord]:
    """One document of each type, described in the corpus schema."""
    document_type = request.param
    generator = create_generator(document_type, config)
    record = sample_record(document_type, random.Random(21), "cyrillic")
    document = generator.generate(record.fields, seed=record.seed)

    transcription = build_transcription_record(
        record=record,
        annotation=document.annotation,
        generator=generator,
        document_id="doc_000000",
        image_path="images/doc_000000.png",
        collection="test_batch",
        quality=AugmentationReport(blur=True, rotation=-0.4, noise="medium"),
    )
    facts = build_facts_record(record, "doc_000000", "images/doc_000000.png")
    return transcription, facts


# -- shape ----------------------------------------------------------------


def test_both_records_share_an_id_and_an_image(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    transcription, facts = exported
    assert transcription.id == facts.id
    assert transcription.image == facts.image


def test_a_transcription_record_has_the_schema_shape(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    payload = exported[0].to_dict()
    assert set(payload) == {
        "id",
        "image",
        "source",
        "metadata",
        "target",
        "annotation",
    }
    assert set(payload["target"]) == {"text", "parts"}
    assert {"origin", "collection", "era", "year_approx"} <= set(
        payload["source"]
    )


def test_a_facts_record_has_the_schema_shape(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    payload = exported[1].to_dict()
    assert set(payload) == {"id", "image", "facts"}
    for fact in payload["facts"]:
        assert {"category", "value", "fuzzy", "evidence_text"} <= set(fact)


def test_both_records_serialise_to_json(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    for record in exported:
        assert json.loads(json.dumps(record.to_dict(), ensure_ascii=False))


# -- provenance and metadata ----------------------------------------------


def test_a_generated_page_says_it_is_synthetic(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    """The three-stage split depends on this being right."""
    source = exported[0].to_dict()["source"]
    assert source["origin"] == "synthetic"
    assert source["collection"] == "test_batch"
    assert source["seed"] is not None


def test_the_era_follows_the_year_the_document_is_dated(
    config: SyntheticConfig,
) -> None:
    for seed in range(20):
        record = sample_record("death_certificate", random.Random(seed))
        assert record.year is not None
        assert record.era == ("modern" if record.year >= 2000 else "old")


def test_capture_quality_comes_from_what_was_done_to_the_page(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    quality = exported[0].to_dict()["metadata"]["quality"]
    assert quality["blur"] is True
    assert quality["rotation"] == -0.4
    assert quality["skew"] is True
    assert quality["noise"] == "medium"
    assert quality["capture"] == "scanner"


def test_a_form_is_mixed_text_and_a_letter_is_handwritten(
    config: SyntheticConfig,
) -> None:
    assert create_generator("ariza", config).text_mode == "handwritten"
    assert create_generator("birth_certificate", config).text_mode == "mixed"


def test_a_bilingual_form_reports_both_alphabets(
    config: SyntheticConfig,
) -> None:
    """The form's own printing is bilingual whatever the clerk wrote in."""
    generator = create_generator("death_certificate", config)
    record = sample_record("death_certificate", random.Random(2), "cyrillic")
    document = generator.generate(record.fields, seed=record.seed)

    metadata = build_transcription_record(
        record, document.annotation, generator, "doc_1", "images/x.png", "b"
    ).to_dict()["metadata"]

    assert metadata["primary_script"] == "cyrillic"
    assert set(metadata["scripts"]) == {"cyrillic", "latin"}


def test_a_blank_sheet_reports_only_its_own_alphabet(
    config: SyntheticConfig,
) -> None:
    generator = create_generator("ariza", config)
    record = sample_record("ariza", random.Random(2), "cyrillic")
    document = generator.generate(record.fields, seed=record.seed)

    metadata = build_transcription_record(
        record, document.annotation, generator, "doc_1", "images/x.png", "b"
    ).to_dict()["metadata"]

    assert metadata["scripts"] == ["cyrillic"]
    assert metadata["has_printed_text"] is False


def test_marks_are_reported_from_what_was_drawn(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    metadata = exported[0].to_dict()["metadata"]
    assert metadata["has_handwriting"] is True
    assert isinstance(metadata["has_stamp"], bool)
    assert isinstance(metadata["has_signature"], bool)


def test_a_synthetic_transcription_is_gold(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    """It was written onto the page, not read off it."""
    annotation = exported[0].to_dict()["annotation"]
    assert annotation["status"] == "gold"
    assert annotation["pre_annotator"].startswith("generator:bitikocr@")
    assert annotation["reviewer"] is None


# -- parts ----------------------------------------------------------------


def test_parts_carry_boxes_as_x_y_width_height(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    """The schema's spelling, not the renderer's two-corner one."""
    parts = exported[0].to_dict()["target"]["parts"]
    assert parts
    for part in parts:
        if part["bbox"] is None:
            continue
        _, _, width, height = part["bbox"]
        assert width > 0 and height > 0, part["role"]


def test_a_part_keeps_the_lines_inside_it(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    """Line boxes are what line-level HTR training needs."""
    parts = exported[0].to_dict()["target"]["parts"]
    assert any(part.get("lines") for part in parts)


def test_marks_are_not_parts_of_the_text(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    roles = {part["role"] for part in exported[0].to_dict()["target"]["parts"]}
    assert "stamp" not in roles
    assert "signature" not in roles


def test_a_box_converts_between_both_spellings() -> None:
    box = BoundingBox(10, 20, 40, 60)
    assert box.to_list() == [10, 20, 40, 60]
    assert box.to_xywh() == [10, 20, 30, 40]


# -- facts ----------------------------------------------------------------


def test_every_declared_category_is_in_the_vocabulary() -> None:
    for document_type, mapping in FIELD_FACTS.items():
        for name, declared in mapping.items():
            where = f"{document_type}.{name}"
            assert declared.category in FACT_CATEGORIES, where
            if declared.subtype is not None:
                allowed = FACT_CATEGORIES[declared.category]
                assert declared.subtype in allowed, where


def test_a_synthetic_fact_is_never_fuzzy(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    """The page was written from the fact, so the fact is exact."""
    assert all(not fact.fuzzy for fact in exported[1].facts)


def test_every_fact_can_be_traced_back_to_the_page(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    """Nothing in a facts record may be invented.

    A date spread over three cells is joined into one evidence string, and
    those cells sit on separate lines, so the check is per word rather than
    on the whole phrase.
    """
    transcription, facts = exported
    assert facts.facts

    written = set(transcription.text.split())
    for fact in facts.facts:
        assert fact.evidence_text, fact.category
        # The seal's lettering is pressed on, not written, so it is the one
        # fact whose evidence is not part of the transcription.
        if fact.category == "stamp_text":
            continue
        for word in fact.evidence_text.split():
            assert word in written, f"{fact.field}: {word!r}"


def test_a_date_spread_over_three_fields_becomes_one_fact() -> None:
    record = sample_record("death_certificate", random.Random(3), "latin")
    facts = build_facts(record.document_type, record.fields, record.dates)

    dated = [fact for fact in facts if fact.field == "death"]
    assert len(dated) == 1
    assert dated[0].value == record.dates["death"]
    assert record.fields["death_month"] in dated[0].evidence_text


def test_a_year_written_out_normalises_to_the_year() -> None:
    record = sample_record("birth_certificate", random.Random(3), "latin")
    facts = build_facts(record.document_type, record.fields, record.dates)

    spelled = next(
        fact for fact in facts if fact.field == "child_birth_year_words"
    )
    assert spelled.value == record.dates["birth"][:4]
    assert spelled.evidence_text == record.fields["child_birth_year_words"]


def test_an_unmapped_document_type_is_rejected() -> None:
    with pytest.raises(KeyError, match="No fact mapping"):
        build_facts("passport", {})


def test_an_unknown_category_is_refused_at_import_time() -> None:
    """The vocabulary is versioned, so a typo must not slip into a corpus."""
    from bitikocr.data.synthetic.facts import _fact

    with pytest.raises(ValueError, match="Unknown fact category"):
        _fact("not_a_category")
    with pytest.raises(ValueError, match="has no subtype"):
        _fact("number", "not_a_subtype")


# -- the pieces on their own ----------------------------------------------


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
