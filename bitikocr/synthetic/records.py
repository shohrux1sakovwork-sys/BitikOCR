"""Sampling the field values a document is filled with.

A record is the *content* of one document, separate from how it is drawn:
names, places, dates and numbers, in one alphabet, with the internal
consistency a real document has — a person is not registered before they
were born, and a family shares a surname.

Records are sampled first and rendered second, so a batch's text can be
reviewed, edited or reused before any image exists.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from bitikocr.synthetic import corpus
from bitikocr.synthetic.scripts import Script, in_script

__all__ = [
    "DEFAULT_LATIN_SHARE",
    "MODERN_FROM",
    "RECORD_SAMPLERS",
    "DocumentRecord",
    "SampledContent",
    "available_record_types",
    "sample_record",
    "sample_records",
]

_MAX_SEED = 2**31

#: The year Uzbekistan's documents are treated as modern from. The
#: alphabet reform and the forms themselves changed either side of it, so it
#: is the axis the corpus balances on.
MODERN_FROM = 2000

#: Share of records written in Latin when no script is forced. The archive
#: holds both alphabets, but the font library is overwhelmingly Cyrillic, so
#: Latin records would otherwise be drawn by the same handful of hands over
#: and over. Raise this as Latin fonts are added.
DEFAULT_LATIN_SHARE = 0.2

_ARIZA_SUBJECTS: tuple[str, ...] = (
    "yashab turgan turar joyimga egalik huquqini belgilab berishingizni",
    (
        "maishiy xizmat ko'rsatish shahobchasi qurish uchun yer maydoni "
        "ajratib berishingizni"
    ),
    "ko'p yillik mehnatim inobatga olinib moddiy yordam ko'rsatishingizni",
    "farzandimni maktabgacha ta'lim muassasasiga qabul qilishingizni",
    "uy joy qurish uchun ruxsat berishingizni",
    "oilamning ijtimoiy holatini hisobga olib yordam berishingizni",
)

_ARIZA_OPENINGS: tuple[str, ...] = (
    "Arizam mazmuni shundan iboratki",
    "Beraman ushbu arizani shu haqda ki",
    "Sizga murojaat qilishimga sabab shuki",
)

_TITLES: tuple[str, ...] = ("Ariza",)


@dataclass(frozen=True)
class SampledContent:
    """What one sampler produced.

    Args:
        fields: The field values, as they will be written on the page.
        year: The year the document is dated, for balancing the corpus.
        dates: Normalised ``YYYY-MM-DD`` dates for its date groups.
    """

    fields: dict[str, Any]
    year: int
    dates: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentRecord:
    """The content of one document, before anything is drawn.

    Args:
        document_type: Which generator renders this record.
        script: The alphabet its text is written in.
        seed: Makes the rendered page reproducible.
        fields: Field name mapped to its value.
        notes: What the sampler knows that the fields do not say — the
            year the document is dated, its era, and its normalised dates.
    """

    document_type: str
    script: Script
    seed: int
    fields: dict[str, Any]
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def year(self) -> int | None:
        """The year the document is dated, if the sampler recorded one."""
        year = self.notes.get("year_approx")
        return int(year) if year is not None else None

    @property
    def era(self) -> str:
        """Whether the document is old or modern."""
        recorded = self.notes.get("era")
        if recorded:
            return str(recorded)
        year = self.year
        return "modern" if year is None or year >= MODERN_FROM else "old"

    @property
    def dates(self) -> dict[str, str]:
        """Normalised dates for the document's date groups."""
        return dict(self.notes.get("dates", {}))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the record."""
        payload: dict[str, Any] = {
            "document_type": self.document_type,
            "script": self.script,
            "seed": self.seed,
            "fields": self.fields,
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DocumentRecord:
        """Rebuild a record from its JSON form.

        Args:
            payload: A decoded record.

        Returns:
            The record.

        Raises:
            ValueError: If a required key is missing.
        """
        try:
            return cls(
                document_type=str(payload["document_type"]),
                script=str(payload.get("script", "latin")),  # type: ignore[arg-type]
                seed=int(payload["seed"]),
                fields=dict(payload["fields"]),
                notes=dict(payload.get("notes", {})),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed record: {payload!r}") from error


# -- shared pieces ---------------------------------------------------------


def _seal_text(rng: random.Random, script: Script) -> dict[str, Any]:
    """Sample the lettering pressed into the office seal."""
    region = rng.choice(list(corpus.DISTRICTS))
    ring = f"O'zbekiston Respublikasi * {region} viloyati FHDYO bo'limi *"
    return {
        "stamp_ring": in_script(ring, script).upper(),
        "stamp_center": [in_script("FHDYO", script)],
    }


def _serial(rng: random.Random) -> str:
    """Sample a seven-digit form serial number."""
    return f"{rng.randrange(10**7):07d}"


def _record_number(rng: random.Random, script: Script) -> str:
    """Sample a civil record number in the registry's pattern.

    The letter in the middle marks the record kind, and a clerk writes it in
    the alphabet the rest of the form uses.

    Args:
        rng: Random source.
        script: Alphabet the letter is written in.

    Returns:
        A number such as "1-2108-20-T-003".
    """
    kind = "T" if script == "latin" else "Т"
    return (
        f"1-{rng.randrange(1000, 4000)}-{rng.randrange(1, 40):02d}"
        f"-{kind}-{rng.randrange(1, 999):03d}"
    )


def _later_date(
    rng: random.Random, year: int, month: int, day: int
) -> tuple[int, int, int]:
    """Sample a date a few days after the given one, without crossing a year."""
    day += rng.randint(1, 20)
    if day > 28:
        day -= 28
        month += 1
    if month > 12:
        month, year = 1, year + 1
    return year, month, day


# -- death certificate -----------------------------------------------------


def _sample_death_certificate(
    rng: random.Random, script: Script
) -> SampledContent:
    """Sample the content of one death certificate."""
    person = corpus.sample_person(rng, script)
    born = rng.randint(1925, 1975)
    died = rng.randint(born + 40, min(born + 95, 2024))
    age = died - born

    month_number, month = corpus.month_name(rng, script)
    day = rng.randint(1, 28)
    reg_year, reg_month_number, reg_day = _later_date(
        rng, died, month_number, day
    )
    reg_month = in_script(corpus.MONTHS[reg_month_number - 1], script)
    place = corpus.sample_place(rng, script)

    fields: dict[str, Any] = {
        "surname": person.surname,
        "given_name_patronymic": person.full_given_name(),
        "citizenship": in_script("O'zbekiston", script),
        "death_year": str(died),
        "death_month": month,
        "death_day": str(day),
        "death_year_in_words": corpus.year_in_words(died, script),
        "age_at_death": str(age),
        "record_year": str(reg_year),
        "record_month": reg_month,
        "record_day": str(reg_day),
        "record_number": str(rng.randint(1, 1999)),
        "cause_of_death": corpus.sample_cause_of_death(rng, script),
        "death_place_country": place["country"],
        "death_place_region": place["region"],
        "death_place_district": place["district"],
        "death_place_settlement": place["settlement"],
        "registration_office": corpus.sample_office(rng, script),
        "issue_year": str(reg_year),
        "issue_month": reg_month,
        "issue_day": str(reg_day),
        "registrar_name": corpus.sample_person(rng, script).initials(),
        "serial_number": _serial(rng),
        **_seal_text(rng, script),
    }
    registered = _iso(reg_year, reg_month_number, reg_day)
    return SampledContent(
        fields=fields,
        year=died,
        dates={
            "death": _iso(died, month_number, day),
            "record": registered,
            "issue": registered,
        },
    )


# -- birth certificate -----------------------------------------------------


def _sample_birth_certificate(
    rng: random.Random, script: Script
) -> SampledContent:
    """Sample the content of one birth certificate."""
    stem = corpus.surname_stem(rng)
    father = corpus.sample_person(
        rng, script, is_female=False, surname_stem=stem
    )
    mother = corpus.sample_person(
        rng, script, is_female=True, surname_stem=stem
    )

    is_daughter = rng.random() < 0.5
    child = corpus.sample_person(
        rng, script, is_female=is_daughter, surname_stem=stem
    )
    suffix = in_script("qizi" if is_daughter else "o'g'li", script)

    born = rng.randint(1995, 2024)
    month_number, month = corpus.month_name(rng, script)
    day = rng.randint(1, 28)
    reg_year, reg_month_number, reg_day = _later_date(
        rng, born, month_number, day
    )
    reg_month = in_script(corpus.MONTHS[reg_month_number - 1], script)
    place = corpus.sample_place(rng, script)
    nationality = corpus.sample_nationality(rng, script)

    fields: dict[str, Any] = {
        "child_surname": child.surname,
        "child_given_name": (
            f"{child.given_name} {father.given_name} {suffix}"
        ),
        "child_birth_date": (
            f"{born} {in_script('yil', script)} {month} {day}"
        ),
        "child_birth_year_words": corpus.year_in_words(born, script),
        "birth_country": place["country"],
        "birth_region": place["region"],
        "birth_district": place["district"],
        "birth_settlement": place["settlement"],
        "record_year": str(reg_year),
        "record_month": reg_month,
        "record_day": str(reg_day),
        "record_number": _record_number(rng, script),
        "father_surname": father.surname,
        "father_given_name": father.full_given_name(),
        "father_nationality": nationality,
        "father_citizenship": in_script("O'zbekiston", script),
        "mother_surname": mother.surname,
        "mother_given_name": mother.full_given_name(),
        "mother_nationality": nationality,
        "mother_citizenship": in_script("O'zbekiston", script),
        "registry_office": corpus.sample_office(rng, script),
        "issue_year": str(reg_year),
        "issue_month": reg_month,
        "issue_day": str(reg_day),
        "registry_head_name": corpus.sample_person(rng, script).initials(),
        "form_series": _form_series(rng, script),
        "form_number": _serial(rng),
        **_seal_text(rng, script),
    }

    # A settlement is often left blank on the real forms.
    if rng.random() < 0.25:
        fields["birth_settlement"] = ""

    registered = _iso(reg_year, reg_month_number, reg_day)
    return SampledContent(
        fields=fields,
        year=born,
        dates={
            "birth": _iso(born, month_number, day),
            "record": registered,
            "issue": registered,
        },
    )


def _form_series(rng: random.Random, script: Script) -> str:
    """Sample a form series: a roman numeral and two letters.

    The letters follow the document's alphabet, as they do on the real
    forms. The numeral does not: roman numerals are written the same way in
    both.

    Args:
        rng: Random source.
        script: Alphabet the letters are drawn from.

    Returns:
        A series such as "III-XX" or "III-ЮА".
    """
    numeral = rng.choice(("I", "II", "III", "IV", "V"))
    alphabet = (
        "ABCDEFGHIKLMNOPRSTUVXYZ"
        if script == "latin"
        else "АБВГДЕЖЗИКЛМНОПРСТУФХЧШЮЯ"
    )
    letters = "".join(rng.choice(alphabet) for _ in range(2))
    return f"{numeral}-{letters}"


# -- ariza -----------------------------------------------------------------


def _sample_ariza(rng: random.Random, script: Script) -> SampledContent:
    """Sample the content of one application letter."""
    official = corpus.sample_person(rng, script, is_female=False)
    applicant = corpus.sample_person(rng, script)
    place = corpus.sample_place(rng, script)

    district = place["district"]
    street = corpus.sample_person(rng, script).surname
    house = rng.randint(1, 120)

    # Hoisted out of the f-strings below: a backslash inside an f-string
    # expression only became legal in Python 3.12, and we support 3.11.
    mayor = in_script("hokimi", script)
    street_word = in_script("ko'chasi", script)
    resident = in_script("uyda yashovchi fuqaro", script)
    to_suffix = in_script("ga", script)
    from_suffix = in_script("dan", script)
    request = in_script("so'rayman", script)

    recipient = f"{district} {mayor} {official.initials()}{to_suffix}"
    applicant_line = (
        f"{district} {street} {street_word} {house} {resident} "
        f"{applicant.surname} {applicant.given_name}{from_suffix}"
    )
    body = (
        f"{in_script(rng.choice(_ARIZA_OPENINGS), script)} "
        f"{in_script(rng.choice(_ARIZA_SUBJECTS), script)} {request}."
    )

    year = rng.randint(1985, 2024)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)

    fields: dict[str, Any] = {
        "recipient": recipient,
        "applicant": applicant_line,
        "title": in_script(rng.choice(_TITLES), script),
        "body": body,
        "signature_name": f"{applicant.surname} {applicant.given_name[0]}.",
        "date": f"{day:02d}.{month:02d}.{year}",
        "reg_number": f"{applicant.surname[0]}-{rng.randint(100, 1999)}",
        "reg_date": f"{day:02d}.{month:02d}.{year}",
        "page_number": str(rng.randint(1, 250)),
    }

    # Not every letter carries a phone number or a clerk's page number.
    if rng.random() < 0.4:
        fields["phone"] = f"9989{rng.randrange(10**8):08d}"
    if rng.random() < 0.3:
        fields.pop("page_number")
    return SampledContent(
        fields=fields, year=year, dates={"filed": _iso(year, month, day)}
    )


def _iso(year: int, month: int, day: int) -> str:
    """Format a date the way the facts records normalise them."""
    return f"{year:04d}-{month:02d}-{day:02d}"


#: Document type mapped to the sampler that fills it.
RECORD_SAMPLERS: dict[
    str, Callable[[random.Random, Script], SampledContent]
] = {
    "ariza": _sample_ariza,
    "birth_certificate": _sample_birth_certificate,
    "death_certificate": _sample_death_certificate,
}


def available_record_types() -> tuple[str, ...]:
    """Return every document type a record can be sampled for, sorted."""
    return tuple(sorted(RECORD_SAMPLERS))


def sample_record(
    document_type: str,
    rng: random.Random,
    script: Script | None = None,
    latin_share: float = DEFAULT_LATIN_SHARE,
) -> DocumentRecord:
    """Sample the content of one document.

    Args:
        document_type: A name from :func:`available_record_types`.
        rng: Random source.
        script: Force an alphabet instead of sampling one.
        latin_share: Chance of drawing Latin when no script is forced.

    Returns:
        The sampled record, carrying its own render seed.

    Raises:
        KeyError: If no sampler is registered for that document type.
    """
    try:
        sampler = RECORD_SAMPLERS[document_type]
    except KeyError as error:
        known = ", ".join(available_record_types())
        raise KeyError(
            f"No record sampler for {document_type!r}. Known types: {known}"
        ) from error

    if script is not None:
        chosen: Script = script
    else:
        chosen = "latin" if rng.random() < latin_share else "cyrillic"
    content = sampler(rng, chosen)
    return DocumentRecord(
        document_type=document_type,
        script=chosen,
        seed=rng.randrange(_MAX_SEED),
        fields=content.fields,
        notes={
            "year_approx": content.year,
            "era": "modern" if content.year >= MODERN_FROM else "old",
            "dates": content.dates,
        },
    )


def sample_records(
    document_type: str,
    count: int,
    rng: random.Random,
    script: Script | None = None,
    latin_share: float = DEFAULT_LATIN_SHARE,
) -> list[DocumentRecord]:
    """Sample several records of one document type.

    Args:
        document_type: A name from :func:`available_record_types`.
        count: How many records to sample. Must be positive.
        rng: Random source.
        script: Force an alphabet instead of sampling one per record.
        latin_share: Share of records written in Latin, when no script is
            forced.

    Returns:
        The sampled records, in order.

    Raises:
        ValueError: If ``count`` is not positive, or ``latin_share`` is not
            a proportion.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if not 0.0 <= latin_share <= 1.0:
        raise ValueError(
            f"latin_share must be between 0 and 1, got {latin_share}"
        )
    return [
        sample_record(document_type, rng, script, latin_share)
        for _ in range(count)
    ]
