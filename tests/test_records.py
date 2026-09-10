"""Tests for bitikocr.data.synthetic.records and the corpus behind them."""

from __future__ import annotations

import random

import pytest

from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic import corpus
from bitikocr.data.synthetic.fonts import FontLibrary
from bitikocr.data.synthetic.generators import create_generator
from bitikocr.data.synthetic.records import (
    DocumentRecord,
    available_record_types,
    sample_record,
    sample_records,
)
from bitikocr.data.synthetic.scripts import SCRIPTS

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
        "consent_letter",
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
def test_every_sampled_field_is_one_a_generator_knows(
    document_type: str, config: SyntheticConfig
) -> None:
    """A record carries every spelling its form variants use, so it is
    checked against the union of what those variants can write."""
    variants: tuple[str | None, ...] = config.layouts_for(document_type) or (
        None,
    )
    known: set[str] = set()
    for template in variants:
        generator = create_generator(document_type, config, template=template)
        known |= set(generator.field_names)
    for record in sample_records(document_type, 10, random.Random(9)):
        assert set(record.fields) <= known, sorted(set(record.fields) - known)


@pytest.mark.parametrize("document_type", available_record_types())
def test_every_variant_of_a_form_is_named_after_its_document_type(
    document_type: str, config: SyntheticConfig
) -> None:
    """Variants are found by name, so each must start with the type."""
    for name in config.layouts_for(document_type):
        assert name.startswith(f"{document_type}_")
        create_generator(document_type, config, template=name)


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


def test_the_issue_date_is_spelled_both_ways() -> None:
    """The bilingual form gives the issue day and month a cell each; the
    single-page form one shared cell. Both spellings must agree."""
    for record in sample_records("death_certificate", 10, random.Random(16)):
        fields = record.fields
        assert fields["issue_day_month"] == (
            f"{fields['issue_day']} {fields['issue_month']}"
        )


def test_a_death_certificate_carries_a_series_beside_its_serial() -> None:
    for record in sample_records("death_certificate", 10, random.Random(17)):
        assert record.fields["serial_number"].isdigit()
        numeral, letters = record.fields["form_series"].split("-")
        assert numeral in ("I", "II", "III", "IV", "V")
        assert len(letters) == 2


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


def test_only_an_author_with_a_seal_of_their_own_puts_one_on_the_page() -> None:
    """This is the rule the whole document type turns on.

    A private citizen has no seal. Their letter carries their signature and
    nothing more, unless they took it to a notary or the mahalla, whose
    seal it then is. A legal entity has a seal and is required to press it
    on anything it signs, so its letters always carry one.
    """
    records = sample_records("consent_letter", 120, random.Random(18))
    seen = set()

    for record in records:
        kind = record.notes["author_kind"]
        certified = bool(record.fields.get("certifier_name"))
        sealed = bool(record.fields.get("stamp_ring"))
        seen.add((kind, certified))

        if kind == "organisation":
            assert sealed, "a legal entity must seal what it signs"
            assert not certified, "an entity does not certify its own letter"
        else:
            assert (
                sealed == certified
            ), "a citizen's letter is sealed only by whoever certified it"

    assert ("individual", False) in seen, "no plain citizen's letter was drawn"
    assert ("individual", True) in seen, "no certified letter was drawn"
    assert ("organisation", False) in seen, "no entity's letter was drawn"


def test_a_citizen_identifies_themselves_by_passport_and_phone() -> None:
    """A letter acted on by an office has to say who wrote it."""
    for record in sample_records("consent_letter", 60, random.Random(24)):
        fields = record.fields
        if record.notes["author_kind"] != "individual":
            continue
        assert fields["passport"]
        assert fields["phone"].startswith("+998")
        series, number = fields["passport"].split()
        assert len(series) == 2 and series.isalpha()
        assert len(number) == 7 and number.isdigit()


def test_an_organisation_carries_no_passport() -> None:
    """A legal entity is not a person; it has a name and a seal instead."""
    for record in sample_records("consent_letter", 60, random.Random(25)):
        if record.notes["author_kind"] != "organisation":
            continue
        assert "passport" not in record.fields
        assert "phone" not in record.fields
        assert record.fields["stamp_ring"]


def test_every_consent_letter_is_signed() -> None:
    for record in sample_records("consent_letter", 30, random.Random(26)):
        assert record.fields["signature_name"]
        assert record.fields["title"].lower() in (
            "rozilik xati",
            "розилик хати",
        )


def test_an_organisation_consents_to_what_an_organisation_can() -> None:
    """An entity has no courtyard and no family. It consents in the plural,
    to work being done and to its premises being used."""
    records = sample_records("consent_letter", 90, random.Random(27), "latin")
    subjects = set()
    for record in records:
        body = record.fields["body"]
        if record.notes["author_kind"] != "organisation":
            continue
        assert "Menga tegishli" not in body, "an entity claimed a home"
        assert "hovlim" not in body, "an entity claimed a courtyard"
        assert "bildiramiz" in body or "beramiz" in body, body
        if "qurilish ishlariga" in body:
            subjects.add("works")
        if "binodan" in body:
            subjects.add("premises")
    assert subjects == {"works", "premises"}, subjects


def test_a_consent_letter_consents_to_something() -> None:
    """Each of the subjects the scans carry has to be reachable, and each
    has to name the address it is about."""
    bodies = [
        record.fields["body"]
        for record in sample_records(
            "consent_letter", 90, random.Random(19), "latin"
        )
    ]
    assert all(body.endswith(".") for body in bodies)
    assert any("chegarasi" in body for body in bodies), "no boundary letter"
    assert any(
        "xususiylashtirishga" in body for body in bodies
    ), "none privatise"
    assert any("turar joy" in body for body in bodies), "no housing letter"


def test_a_citizens_letter_says_where_everyone_lives() -> None:
    """A consent about a boundary is meaningless without the address, and
    both parties are in the same city."""
    for record in sample_records(
        "consent_letter", 30, random.Random(20), "latin"
    ):
        if record.notes["author_kind"] != "individual":
            continue
        applicant = record.fields["applicant"]
        assert "ko'chasi" in applicant
        assert "yashovchi" in applicant
        city = applicant.split(" shahar")[0]
        assert city in record.fields["body"], record.fields["body"]


def test_a_seal_names_whoever_it_belongs_to() -> None:
    """Whose seal it is follows from who put it there — a notary, a
    neighbourhood, or the entity that wrote the letter. It is never the
    civil registry's, which has nothing to do with a consent."""
    kinds = set()
    for record in sample_records(
        "consent_letter", 80, random.Random(21), "latin"
    ):
        ring = record.fields.get("stamp_ring")
        if not ring:
            continue
        centre = record.fields["stamp_center"]
        assert "FHDYO" not in ring, "a registry seal has no place on a consent"
        assert len(centre) == 1 and centre[0]

        if "MAHALLA FUQAROLAR YIG'INI" in ring:
            kinds.add("mahalla")
            # A neighbourhood's seal names the neighbourhood twice over.
            assert centre[0].upper() in ring
        elif "NOTARIAL" in ring:
            kinds.add("notary")
        else:
            kinds.add("organisation")
            assert centre[0].upper() in ring
    assert kinds == {"mahalla", "notary", "organisation"}, kinds


def test_an_address_names_a_street_and_a_house() -> None:
    latin = corpus.sample_address(random.Random(22), "latin")
    assert latin.street in corpus.STREETS
    assert latin.house
    # The short form stops before the dwelling word so a sentence can
    # inflect it; the header's form does not.
    assert latin.short("latin").endswith(latin.house)
    assert latin.line("latin").startswith(latin.short("latin"))
    assert " uy" in latin.line("latin")


def test_neighbours_in_one_letter_share_a_city() -> None:
    rng = random.Random(23)
    mine = corpus.sample_address(rng, "latin")
    theirs = corpus.sample_address(rng, "latin", mine.city)
    assert theirs.city == mine.city
