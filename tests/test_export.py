"""Tests for bitikocr.data.synthetic.export and the facts it reads."""

from __future__ import annotations

import json
import random
from typing import get_args

import pytest

from bitikocr.config import SyntheticConfig
from bitikocr.data.models.schema import (
    FactsRecord,
    Role,
    TranscriptionRecord,
)

#: The six regions the schema allows a part to be.
SCHEMA_ROLES = frozenset(get_args(Role))
from bitikocr.data.synthetic.augment import AugmentationReport, rotate_page
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
from bitikocr.data.synthetic.generators.death_certificate import (
    SINGLE_TEMPLATE,
)
from bitikocr.data.synthetic.records import (
    available_record_types,
    sample_record,
)
from bitikocr.data.synthetic.scripts import Script

#: One page of every document type, plus every further form variant.
_PAGES: tuple[tuple[str, str | None], ...] = (
    *((document_type, None) for document_type in available_record_types()),
    ("death_certificate", SINGLE_TEMPLATE),
)


@pytest.fixture(params=_PAGES, ids=lambda page: page[1] or page[0])
def exported(
    request: pytest.FixtureRequest, config: SyntheticConfig
) -> tuple[TranscriptionRecord, FactsRecord]:
    """One document of each kind, described in the corpus schema."""
    document_type, template = request.param
    generator = create_generator(document_type, config, template=template)
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
    facts = build_facts_record(
        record, "doc_000000", "images/doc_000000.png", document.annotation
    )
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
    # A synthetic page is certain of every word, so it never carries the
    # optional uncertain_spans key.
    assert list(payload) == [
        "id",
        "image",
        "image_size",
        "source",
        "metadata",
        "target",
        "annotation",
    ]
    assert set(payload["target"]) == {"text", "parts"}
    assert set(payload["source"]) == {
        "origin",
        "collection",
        "era",
        "year_approx",
        "original_file",
    }
    assert "scripts" not in payload["metadata"]
    assert "primary_script" not in payload["metadata"]


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
    assert source["era"] in ("old", "modern")
    assert isinstance(source["year_approx"], int)


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


def test_a_bilingual_form_reports_the_languages_it_is_printed_in(
    config: SyntheticConfig,
) -> None:
    """The form is printed in Uzbek Latin and Russian whatever the clerk
    wrote in, and the clerk's own language comes first."""
    generator = create_generator("death_certificate", config)
    record = sample_record("death_certificate", random.Random(2), "cyrillic")
    document = generator.generate(record.fields, seed=record.seed)

    metadata = build_transcription_record(
        record, document.annotation, generator, "doc_1", "images/x.png", "b"
    ).to_dict()["metadata"]

    assert metadata["language"] == ["uz-cyrillic", "uz-latin", "ru"]


def test_a_blank_sheet_reports_only_its_own_language(
    config: SyntheticConfig,
) -> None:
    generator = create_generator("ariza", config)
    record = sample_record("ariza", random.Random(2), "cyrillic")
    document = generator.generate(record.fields, seed=record.seed)

    metadata = build_transcription_record(
        record, document.annotation, generator, "doc_1", "images/x.png", "b"
    ).to_dict()["metadata"]

    assert metadata["language"] == ["uz-cyrillic"]
    assert metadata["has_printed_text"] is False


def test_a_single_language_form_adds_only_what_the_clerk_wrote(
    config: SyntheticConfig,
) -> None:
    """The single-page certificate is printed in Uzbek Cyrillic, so a
    Cyrillic record puts one language on the page, and a Latin one two."""
    generator = create_generator(
        "death_certificate", config, template=SINGLE_TEMPLATE
    )
    cases: tuple[tuple[Script, list[str]], ...] = (
        ("cyrillic", ["uz-cyrillic"]),
        ("latin", ["uz-latin", "uz-cyrillic"]),
    )
    for script, expected in cases:
        record = sample_record("death_certificate", random.Random(2), script)
        document = generator.generate(record.fields, seed=record.seed)
        metadata = build_transcription_record(
            record, document.annotation, generator, "doc_1", "images/x.png", "b"
        ).to_dict()["metadata"]
        assert metadata["language"] == expected
        assert metadata["has_printed_text"] is True


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
        _, _, width, height = part["bbox"]
        assert width > 0 and height > 0, part["role"]


def test_every_part_has_exactly_the_schema_keys(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    for part in exported[0].to_dict()["target"]["parts"]:
        assert list(part) == ["role", "polygon", "bbox", "text", "hand"]
        assert part["role"] in SCHEMA_ROLES, part["role"]
        assert part["hand"] in ("handwritten", "printed", "mixed", None)


def test_a_parts_box_encloses_its_outline_on_the_page(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    payload = exported[0].to_dict()
    page_width, page_height = payload["image_size"]
    for part in payload["target"]["parts"]:
        x, y, width, height = part["bbox"]
        assert len(part["polygon"]) >= 3
        for px, py in part["polygon"]:
            assert x <= px <= x + width and y <= py <= y + height
            assert 0 <= px <= page_width and 0 <= py <= page_height


def test_the_page_size_is_the_saved_images(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    width, height = exported[0].to_dict()["image_size"]
    assert width > 0 and height > 0


def test_marks_are_regions_of_their_own(
    exported: tuple[TranscriptionRecord, FactsRecord],
) -> None:
    """The schema has regions for a stamp and a signature, and a page that
    carries one reports it as a part with the matching role."""
    payload = exported[0].to_dict()
    roles = {part["role"] for part in payload["target"]["parts"]}
    assert ("stamp" in roles) == payload["metadata"]["has_stamp"]
    scribbles = [
        part
        for part in payload["target"]["parts"]
        if part["role"] == "signature" and not part["text"]
    ]
    assert bool(scribbles) == payload["metadata"]["has_signature"]
    for part in payload["target"]["parts"]:
        if part["role"] == "stamp":
            assert part["hand"] == "printed"


def test_a_letter_is_divided_into_header_title_body_and_signature(
    config: SyntheticConfig,
) -> None:
    generator = create_generator("ariza", config)
    record = sample_record("ariza", random.Random(4), "latin")
    document = generator.generate(record.fields, seed=record.seed)
    parts = build_transcription_record(
        record, document.annotation, generator, "doc_1", "images/x.png", "b"
    ).to_dict()["target"]["parts"]

    # A wrapped block keeps its line breaks, so compare the words.
    by_role: dict[str, list[str]] = {}
    for part in parts:
        by_role.setdefault(part["role"], []).append(
            " ".join(part["text"].split())
        )
    assert {"header", "title", "body", "signature"} <= set(by_role)
    assert record.fields["recipient"] in by_role["header"]
    assert by_role["body"] == [record.fields["body"]]


def test_what_a_form_typesets_is_printed_matter(
    config: SyntheticConfig,
) -> None:
    """A serial number is printed; everything in the cells is written."""
    generator = create_generator("death_certificate", config)
    record = sample_record("death_certificate", random.Random(4), "latin")
    document = generator.generate(record.fields, seed=record.seed)
    parts = build_transcription_record(
        record, document.annotation, generator, "doc_1", "images/x.png", "b"
    ).to_dict()["target"]["parts"]

    printed = [part for part in parts if part["hand"] == "printed"]
    assert {part["role"] for part in printed} == {"other", "stamp"}
    assert any(
        part["text"] == record.fields["serial_number"] for part in printed
    )
    body = [part for part in parts if part["role"] == "body"]
    assert body and all(part["hand"] == "handwritten" for part in body)


def test_a_skewed_page_outlines_its_regions_as_tilted_shapes(
    config: SyntheticConfig,
) -> None:
    """A page laid crookedly on the scanner has tilted regions. The polygon
    keeps that tilt; only the box is square."""
    generator = create_generator("ariza", config)
    record = sample_record("ariza", random.Random(4), "latin")
    document = generator.generate(record.fields, seed=record.seed)
    _, turned = rotate_page(document.image, document.annotation, 4.0)
    parts = build_transcription_record(
        record, turned, generator, "doc_1", "images/x.png", "b"
    ).to_dict()["target"]["parts"]

    tilted = 0
    for part in parts:
        xs = {x for x, _ in part["polygon"]}
        ys = {y for _, y in part["polygon"]}
        if len(xs) > 2 or len(ys) > 2:
            tilted += 1
    assert tilted, "every outline stayed an upright rectangle"


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


def test_facts_describe_only_what_reached_the_page(
    config: SyntheticConfig,
) -> None:
    """A record holds every spelling its variants use. The facts beside a
    page must not claim the ones this variant had no cell for."""
    generator = create_generator(
        "death_certificate", config, template=SINGLE_TEMPLATE
    )
    record = sample_record("death_certificate", random.Random(4), "cyrillic")
    document = generator.generate(record.fields, seed=record.seed)
    facts = build_facts_record(
        record, "doc_1", "images/x.png", document.annotation
    )

    by_field = {fact.field: fact for fact in facts.facts}
    assert "citizenship" not in by_field
    assert by_field["issue"].value == record.dates["issue"]
    assert by_field["issue"].evidence_text == (
        f"{record.fields['issue_year']} {record.fields['issue_day_month']}"
    )
    assert by_field["form_series"].value == record.fields["form_series"]
    assert by_field["stamp_ring"].category == "stamp_text"


def test_the_bilingual_form_keeps_three_cells_of_issue_evidence(
    config: SyntheticConfig,
) -> None:
    generator = create_generator("death_certificate", config)
    record = sample_record("death_certificate", random.Random(4), "cyrillic")
    document = generator.generate(record.fields, seed=record.seed)
    facts = build_facts_record(
        record, "doc_1", "images/x.png", document.annotation
    )

    by_field = {fact.field: fact for fact in facts.facts}
    fields = record.fields
    assert by_field["issue"].evidence_text == (
        f"{fields['issue_year']} {fields['issue_month']} {fields['issue_day']}"
    )
    assert "form_series" not in by_field
    assert "citizenship" in by_field


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
