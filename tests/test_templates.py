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
    assert bilingual.serial is not None
    assert bilingual.signature is not None


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
