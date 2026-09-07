"""Tests for bitikocr.synthetic.records and the corpus behind them."""

from __future__ import annotations

import random

import pytest

from bitikocr.config import SyntheticConfig
from bitikocr.synthetic import corpus
from bitikocr.synthetic.fonts import FontLibrary
from bitikocr.synthetic.generators import create_generator
from bitikocr.synthetic.records import (
    DocumentRecord,
    available_record_types,
    sample_record,
    sample_records,
)
from bitikocr.synthetic.scripts import SCRIPTS

CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяўқғҳ")
LATIN = set("abcdefghijklmnopqrstuvwxyz")


def record_text(record: DocumentRecord, names: set[str] | None = None) -> str:
    """Join the record's field values, optionally only the named ones."""
    return " ".join(
        value if isinstance(value, str) else " ".join(value)
        for name, value in record.fields.items()
        if names is None or name in names
    )


def letters(record: DocumentRecord, names: set[str] | None = None) -> set[str]:
    """Every alphabetic character the record writes, lower-cased."""
    return {
        char for char in record_text(record, names).lower() if char.isalpha()
    }


# -- sampling --------------------------------------------------------------


def test_every_document_type_can_be_sampled() -> None:
    assert available_record_types() == (
        "ariza",
        "birth_certificate",
        "death_certificate",
    )


def test_an_unknown_document_type_is_rejected() -> None:
    with pytest.raises(KeyError, match="No record sampler"):
        sample_record("passport", random.Random(1))


@pytest.mark.parametrize("document_type", available_record_types())
def test_the_same_seed_samples_the_same_record(document_type: str) -> None:
    first = sample_record(document_type, random.Random(5))
    second = sample_record(document_type, random.Random(5))
    assert first == second


@pytest.mark.parametrize("document_type", available_record_types())
def test_records_vary(document_type: str) -> None:
    """Thirty records should not be thirty copies of one."""
    records = sample_records(document_type, 30, random.Random(6))
    surnames = {next(iter(record.fields.values())) for record in records}
    assert len(surnames) > 15
    assert len({record.seed for record in records}) == 30


@pytest.mark.parametrize("document_type", available_record_types())
@pytest.mark.parametrize("script", SCRIPTS)
def test_a_record_stays_in_one_alphabet(
    document_type: str, script: str, config: SyntheticConfig
) -> None:
    """Handwritten values are all in one script.

    Machine-printed marks are excluded: a form series is typeset beside a
    roman numeral, which belongs to neither alphabet.
    """
    generator = create_generator(document_type, config)
    template = getattr(generator, "template", None)
    handwritten = set(generator.field_names) - {"stamp_ring", "stamp_center"}
    if template is not None:
        handwritten -= set(template.printed_names)

    record = sample_record(document_type, random.Random(7), script)  # type: ignore[arg-type]
    used = letters(record, handwritten)
    assert used, "expected some text"
    if script == "cyrillic":
        assert used <= CYRILLIC, sorted(used - CYRILLIC)
    else:
        assert used <= LATIN, sorted(used - LATIN)


def test_both_alphabets_appear_when_neither_is_forced() -> None:
    records = sample_records("death_certificate", 20, random.Random(8))
    assert {record.script for record in records} == {"latin", "cyrillic"}


@pytest.mark.parametrize("document_type", available_record_types())
def test_every_sampled_field_is_one_the_generator_knows(
    document_type: str, config: SyntheticConfig
) -> None:
    generator = create_generator(document_type, config)
    known = set(generator.field_names)
    for record in sample_records(document_type, 10, random.Random(9)):
        assert set(record.fields) <= known, sorted(set(record.fields) - known)


@pytest.mark.parametrize("document_type", available_record_types())
def test_a_shipped_font_can_write_every_sampled_record(
    document_type: str, library: FontLibrary
) -> None:
    """A record no font can draw would be silently skipped at render time."""
    for record in sample_records(document_type, 20, random.Random(10)):
        text = record_text(record)
        assert any(
            font.can_render(text) for font in library
        ), f"no font can write {record.script} record {record.seed}"


def test_a_zero_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        sample_records("ariza", 0, random.Random(1))


# -- internal consistency --------------------------------------------------


def test_a_death_is_registered_after_it_happened() -> None:
    for record in sample_records("death_certificate", 20, random.Random(11)):
        fields = record.fields
        assert int(fields["record_year"]) >= int(fields["death_year"])


def test_an_age_matches_the_year_of_death() -> None:
    for record in sample_records("death_certificate", 20, random.Random(12)):
        assert 40 <= int(record.fields["age_at_death"]) <= 99


def test_a_family_shares_a_surname_stem() -> None:
    for record in sample_records("birth_certificate", 20, random.Random(13)):
        fields = record.fields
        child = fields["child_surname"]
        father = fields["father_surname"]
        mother = fields["mother_surname"]
        stem = min(len(child), len(father), len(mother)) - 3
        assert child[:stem] == father[:stem] == mother[:stem]


def test_a_serial_number_is_seven_digits() -> None:
    for record in sample_records("birth_certificate", 10, random.Random(14)):
        assert record.fields["form_number"].isdigit()
        assert len(record.fields["form_number"]) == 7


# -- records on the wire ---------------------------------------------------


def test_a_record_round_trips_through_its_dict() -> None:
    record = sample_record("ariza", random.Random(15))
    assert DocumentRecord.from_dict(record.to_dict()) == record


def test_a_record_without_a_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="Malformed record"):
        DocumentRecord.from_dict({"document_type": "ariza", "fields": {}})


# -- spelled-out years -----------------------------------------------------


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2015, "ikki ming o'n beshinchi"),
        (2000, "ikki minginchi"),
        (2022, "ikki ming yigirma ikkinchi"),
        (1975, "bir ming to'qqiz yuz yetmish beshinchi"),
    ],
)
def test_years_are_spelled_out_in_uzbek(year: int, expected: str) -> None:
    assert corpus.year_in_words(year, "latin") == expected


def test_a_year_outside_the_supported_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cannot spell"):
        corpus.year_in_words(1500, "latin")
