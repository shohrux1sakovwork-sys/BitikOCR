"""Tests for bitikocr.synthetic.templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bitikocr.config import SyntheticConfig
from bitikocr.synthetic.templates import FormTemplate

MINIMAL_LAYOUT: dict[str, Any] = {
    "template_name": "toy",
    "image": "toy.png",
    "width": 100,
    "height": 200,
    "fields": [
        {
            "id": "name",
            "baseline_y": 50,
            "bbox_xyxy": [10, 20, 90, 54],
            "text_type": "uzbek_latin_word",
        },
        {
            "id": "year",
            "baseline_y": 80,
            "bbox_xyxy": [10, 50, 40, 84],
            "text_type": "digits_4",
        },
        {
            "id": "address_line2",
            "baseline_y": 120,
            "bbox_xyxy": [20, 90, 90, 124],
            "text_type": "uzbek_latin_phrase",
        },
        {
            "id": "address_line1",
            "baseline_y": 100,
            "bbox_xyxy": [10, 70, 90, 104],
            "text_type": "uzbek_latin_phrase",
        },
    ],
}


def test_a_layout_round_trips_into_a_template() -> None:
    template = FormTemplate.from_layout(MINIMAL_LAYOUT)
    assert template.name == "toy"
    assert template.background == "toy.png"
    assert template.native_size == (100, 200)


def test_digit_fields_are_marked_numeric() -> None:
    template = FormTemplate.from_layout(MINIMAL_LAYOUT)
    assert template.field("year").is_numeric
    assert not template.field("name").is_numeric


def test_numbered_lines_merge_into_one_field_in_order() -> None:
    """``_line1``/``_line2`` are one logical field, whatever order they
    appear in the layout."""
    template = FormTemplate.from_layout(MINIMAL_LAYOUT)
    assert "address_line1" not in template.field_names
    baselines = [
        segment.baseline_y for segment in template.field("address").segments
    ]
    assert baselines == [100, 120]


def test_a_segment_knows_its_width() -> None:
    template = FormTemplate.from_layout(MINIMAL_LAYOUT)
    assert template.field("year").segments[0].width == 30


def test_an_unknown_field_is_rejected() -> None:
    template = FormTemplate.from_layout(MINIMAL_LAYOUT)
    with pytest.raises(KeyError, match="has no field 'nope'"):
        template.field("nope")


def test_a_layout_without_required_keys_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs 'width'"):
        FormTemplate.from_layout({"fields": []})


def test_a_layout_without_writable_fields_is_rejected() -> None:
    payload = {**MINIMAL_LAYOUT, "fields": []}
    with pytest.raises(ValueError, match="at least one writable field"):
        FormTemplate.from_layout(payload)


def test_a_writable_field_needs_a_baseline() -> None:
    payload = {
        **MINIMAL_LAYOUT,
        "fields": [
            {
                "id": "name",
                "baseline_y": None,
                "bbox_xyxy": [0, 0, 10, 10],
                "text_type": "uzbek_latin_word",
            }
        ],
    }
    with pytest.raises(ValueError, match="has no baseline_y"):
        FormTemplate.from_layout(payload)


def test_malformed_json_is_reported_as_a_bad_layout(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed layout JSON"):
        FormTemplate.load(path)


# -- the shipped bilingual certificate layout ------------------------------


@pytest.fixture()
def bilingual(config: SyntheticConfig) -> FormTemplate:
    """The measured layout of the two-page bilingual certificate."""
    return FormTemplate.load(config.layout("death_certificate_bilingual"))


def test_the_shipped_layout_matches_its_background(
    bilingual: FormTemplate, config: SyntheticConfig
) -> None:
    """Coordinates measured at another size would land off the printed rules."""
    from PIL import Image

    with Image.open(config.background(bilingual.background)) as background:
        assert background.size == bilingual.native_size


def test_the_shipped_layout_defines_every_certificate_field(
    bilingual: FormTemplate,
) -> None:
    assert bilingual.field_names == (
        "surname",
        "given_name_patronymic",
        "citizenship",
        "death_year",
        "death_month",
        "death_day",
        "death_year_in_words",
        "age_at_death",
        "record_year",
        "record_month",
        "record_day",
        "record_number",
        "cause_of_death",
        "death_place_country",
        "death_place_region",
        "death_place_district",
        "death_place_settlement",
        "registration_office",
        "issue_year",
        "issue_month",
        "issue_day",
    )


def test_the_shipped_layout_has_a_seal_a_serial_and_a_signature(
    bilingual: FormTemplate,
) -> None:
    assert bilingual.seal is not None
    assert bilingual.signature is not None
    assert bilingual.printed_names == ("serial_number",)


def test_the_two_double_ruled_fields_carry_two_segments(
    bilingual: FormTemplate,
) -> None:
    assert len(bilingual.field("cause_of_death").segments) == 2
    assert len(bilingual.field("registration_office").segments) == 2


def test_the_seal_fills_its_printed_circle(bilingual: FormTemplate) -> None:
    """The reference calls the seal ~260px across on the 1419px-wide form."""
    assert bilingual.seal is not None
    assert 120 <= bilingual.seal.radius <= 140


def test_every_field_stays_inside_the_form(bilingual: FormTemplate) -> None:
    width, height = bilingual.native_size
    for field in bilingual.fields:
        for segment in field.segments:
            assert 0 <= segment.x_start < segment.x_end <= width, field.name
            assert 0 < segment.baseline_y < height, field.name


def test_the_shipped_layout_is_valid_json(config: SyntheticConfig) -> None:
    payload = json.loads(
        config.layout("death_certificate_bilingual").read_text(encoding="utf-8")
    )
    assert payload["coordinate_system"].startswith("pixels")


# -- the second layout dialect ---------------------------------------------

OBJECT_DIALECT_LAYOUT: dict[str, Any] = {
    "image": "toy.png",
    "width": 100,
    "height": 200,
    "fields": [
        {
            "id": "name",
            "underline_y": 50,
            "baseline_y": 46,
            "bbox": {"x1": 10, "y1": 20, "x2": 90, "y2": 48},
            "text_type": "surname_uppercase_latin",
        },
        {
            "id": "year",
            "underline_y": 80,
            "bbox": {"x1": 10, "y1": 50, "x2": 40, "y2": 78},
            "text_type": "year_4digit",
        },
        {
            "id": "no_rule",
            "baseline_y": 120,
            "bbox": {"x1": 10, "y1": 95, "x2": 90, "y2": 118},
            "text_type": "initials_surname",
        },
    ],
}


def test_a_box_may_be_an_object_instead_of_an_array() -> None:
    template = FormTemplate.from_layout(OBJECT_DIALECT_LAYOUT, "toy")
    segment = template.field("name").segments[0]
    assert (segment.x_start, segment.x_end) == (10, 90)


def test_the_printed_rule_wins_over_the_text_baseline() -> None:
    """``underline_y`` is the rule itself; ``baseline_y`` sits above it."""
    template = FormTemplate.from_layout(OBJECT_DIALECT_LAYOUT, "toy")
    assert template.field("name").segments[0].baseline_y == 50


def test_a_field_with_only_a_text_baseline_still_loads() -> None:
    template = FormTemplate.from_layout(OBJECT_DIALECT_LAYOUT, "toy")
    assert template.field("no_rule").segments[0].baseline_y == 120


def test_the_layout_name_falls_back_to_the_file_stem() -> None:
    template = FormTemplate.from_layout(OBJECT_DIALECT_LAYOUT, "toy")
    assert template.name == "toy"


# -- role classification ---------------------------------------------------


@pytest.mark.parametrize(
    ("text_type", "expected"),
    [
        ("digits_4", "numeric"),
        ("year_4digit", "numeric"),
        ("day_2digit", "numeric"),
        ("uzbek_latin_word", "text"),
        ("month_name_uzbek", "text"),
        ("record_no (e.g. 1-2108-20-T-003 pattern)", "text"),
    ],
)
def test_a_field_is_numeric_only_when_its_type_says_digits(
    text_type: str, expected: str
) -> None:
    payload = {
        **MINIMAL_LAYOUT,
        "fields": [
            {
                "id": "f",
                "baseline_y": 10,
                "bbox_xyxy": [0, 0, 50, 12],
                "text_type": text_type,
            }
        ],
    }
    template = FormTemplate.from_layout(payload)
    assert template.field("f").is_numeric is (expected == "numeric")


@pytest.mark.parametrize(
    "text_type",
    [
        "printed_digits_7 (typewriter/serif, NOT handwritten)",
        "digits_7 (typographic, red/black)",
        "series (roman numeral + 2 letters, e.g. III-XX)",
    ],
)
def test_typeset_values_are_printed_not_handwritten(text_type: str) -> None:
    """A digit count in the type must not outrank 'typographic'."""
    payload = {
        **MINIMAL_LAYOUT,
        "fields": [
            MINIMAL_LAYOUT["fields"][0],
            {
                "id": "printed",
                "bbox_xyxy": [0, 0, 50, 12],
                "text_type": text_type,
            },
        ],
    }
    template = FormTemplate.from_layout(payload)
    assert template.printed_names == ("printed",)
    assert "printed" not in template.field_names


@pytest.mark.parametrize(
    ("text_type", "attribute"),
    [
        ("round_stamp (ink seal)", "seal"),
        ("stamp", "seal"),
        ("signature (scribble) + optional surname", "signature"),
    ],
)
def test_marks_are_recognised_from_their_type(
    text_type: str, attribute: str
) -> None:
    payload = {
        **MINIMAL_LAYOUT,
        "fields": [
            MINIMAL_LAYOUT["fields"][0],
            {
                "id": "mark",
                "bbox_xyxy": [0, 0, 50, 50],
                "text_type": text_type,
            },
        ],
    }
    template = FormTemplate.from_layout(payload)
    assert getattr(template, attribute) is not None


def test_a_qr_region_becomes_a_keep_out_zone() -> None:
    payload = {
        **MINIMAL_LAYOUT,
        "fields": [
            MINIMAL_LAYOUT["fields"][0],
            {
                "id": "qr_code",
                "bbox_xyxy": [0, 0, 50, 50],
                "text_type": "qr",
            },
        ],
    }
    template = FormTemplate.from_layout(payload)
    assert [area.name for area in template.keep_out] == ["qr_code"]
    assert "qr_code" not in template.field_names


# -- the shipped birth certificate layout ----------------------------------


@pytest.fixture()
def birth(config: SyntheticConfig) -> FormTemplate:
    """The measured layout of the two-page bilingual birth certificate."""
    return FormTemplate.load(config.layout("birth_certificate_bilingual"))


def test_the_birth_layout_matches_its_background(
    birth: FormTemplate, config: SyntheticConfig
) -> None:
    from PIL import Image

    with Image.open(config.background(birth.background)) as background:
        assert background.size == birth.native_size


def test_the_birth_layout_covers_child_parents_and_registration(
    birth: FormTemplate,
) -> None:
    names = set(birth.field_names)
    assert {"child_surname", "child_given_name", "child_birth_date"} <= names
    assert {"father_surname", "mother_surname"} <= names
    assert {"registry_office", "registry_head_name"} <= names


def test_the_birth_layout_merges_the_two_office_rules(
    birth: FormTemplate,
) -> None:
    assert len(birth.field("registry_office").segments) == 2
    assert "registry_office_line1" not in birth.field_names


def test_the_birth_layout_has_two_printed_areas(birth: FormTemplate) -> None:
    assert set(birth.printed_names) == {"form_number", "form_series"}


def test_the_birth_layout_reserves_the_printed_qr_code(
    birth: FormTemplate,
) -> None:
    assert [area.name for area in birth.keep_out] == ["qr_code"]


def test_the_birth_signature_zone_has_no_printed_rule(
    birth: FormTemplate,
) -> None:
    """It is an open zone, so the head's name has a field of its own."""
    assert birth.signature is not None
    assert birth.signature.baseline_y is None
    assert "registry_head_name" in birth.field_names


def test_every_birth_field_stays_inside_the_form(birth: FormTemplate) -> None:
    width, height = birth.native_size
    for field in birth.fields:
        for segment in field.segments:
            assert 0 <= segment.x_start < segment.x_end <= width, field.name
            assert 0 < segment.baseline_y < height, field.name
