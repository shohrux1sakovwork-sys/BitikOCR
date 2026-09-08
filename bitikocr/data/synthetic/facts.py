"""Turning a document's field values into structured facts.

A generated document knows exactly what it says, so its facts are exact:
nothing is inferred and nothing is illegible. What is not automatic is
*what kind* of value each field holds — that a registry office is an
organisation and a settlement is a place — so the mapping is written out
per document type rather than guessed from the field name.

Where a document spells one value across several fields, as certificates do
with a year, a month and a day, the fields are grouped into a single fact:
the value is the normalised date and the evidence is the three fields as
they appear on the page.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bitikocr.models.schema import Fact

__all__ = [
    "CATEGORY_SET_VERSION",
    "DATE_GROUPS",
    "FACT_CATEGORIES",
    "FIELD_FACTS",
    "YEAR_IN_WORDS",
    "build_facts",
]

#: Version of the category vocabulary below. Bump it when the set changes,
#: so a corpus can record which vocabulary it was labelled against.
CATEGORY_SET_VERSION = 1

#: Every category a fact may carry, and the subtypes each allows.
FACT_CATEGORIES: Mapping[str, tuple[str, ...]] = {
    "date": (),
    "person_name": (),
    "organisation": (),
    "number": ("id", "phone", "amount", "reference"),
    "address": (),
    "place": (),
    "signature_owner": (),
    "stamp_text": (),
    # Beyond the initial set, for what these documents actually record.
    "nationality": (),
    "citizenship": (),
    "cause_of_death": (),
    "age": (),
    "subject": (),
}


@dataclass(frozen=True)
class FieldFact:
    """How one document field becomes a fact.

    Args:
        category: The fact category the field belongs to.
        subtype: A finer kind, for categories that define one.
    """

    category: str
    subtype: str | None = None


def _fact(category: str, subtype: str | None = None) -> FieldFact:
    """Declare a field's category, checking it against the vocabulary."""
    allowed = FACT_CATEGORIES.get(category)
    if allowed is None:
        raise ValueError(f"Unknown fact category: {category}")
    if subtype is not None and subtype not in allowed:
        raise ValueError(f"{category} has no subtype {subtype!r}")
    return FieldFact(category, subtype)


#: Document type mapped to the category of each of its fields. Fields absent
#: from the mapping carry no fact — the seal's own lettering, for instance,
#: is document furniture rather than something read off the page.
FIELD_FACTS: Mapping[str, Mapping[str, FieldFact]] = {
    "ariza": {
        "recipient": _fact("person_name"),
        "applicant": _fact("address"),
        "body": _fact("subject"),
        "signature_name": _fact("signature_owner"),
        "date": _fact("date"),
        "phone": _fact("number", "phone"),
        "reg_number": _fact("number", "reference"),
        "reg_date": _fact("date"),
        "page_number": _fact("number", "reference"),
    },
    "death_certificate": {
        "surname": _fact("person_name"),
        "given_name_patronymic": _fact("person_name"),
        "citizenship": _fact("citizenship"),
        "death_year_in_words": _fact("date"),
        "age_at_death": _fact("age"),
        "record_number": _fact("number", "reference"),
        "cause_of_death": _fact("cause_of_death"),
        "death_place_country": _fact("place"),
        "death_place_region": _fact("place"),
        "death_place_district": _fact("place"),
        "death_place_settlement": _fact("place"),
        "registration_office": _fact("organisation"),
        "registrar_name": _fact("signature_owner"),
        "form_series": _fact("number", "id"),
        "serial_number": _fact("number", "id"),
        "stamp_ring": _fact("stamp_text"),
    },
    "birth_certificate": {
        "child_surname": _fact("person_name"),
        "child_given_name": _fact("person_name"),
        "child_birth_year_words": _fact("date"),
        "birth_country": _fact("place"),
        "birth_region": _fact("place"),
        "birth_district": _fact("place"),
        "birth_settlement": _fact("place"),
        "record_number": _fact("number", "reference"),
        "father_surname": _fact("person_name"),
        "father_given_name": _fact("person_name"),
        "father_nationality": _fact("nationality"),
        "father_citizenship": _fact("citizenship"),
        "mother_surname": _fact("person_name"),
        "mother_given_name": _fact("person_name"),
        "mother_nationality": _fact("nationality"),
        "mother_citizenship": _fact("citizenship"),
        "registry_office": _fact("organisation"),
        "registry_head_name": _fact("signature_owner"),
        "form_series": _fact("number", "id"),
        "form_number": _fact("number", "id"),
        "stamp_ring": _fact("stamp_text"),
    },
}

#: Fields spelling a year out in words, mapped to the date group that says
#: which year they mean. The value normalises to the year, the evidence
#: keeps the words as written.
YEAR_IN_WORDS: Mapping[str, Mapping[str, str]] = {
    "death_certificate": {"death_year_in_words": "death"},
    "birth_certificate": {"child_birth_year_words": "birth"},
    "ariza": {},
}

#: Fields that together spell one date, per document type: the note holding
#: the normalised value, then the fields whose text is the evidence for it.
#: A group may list more spellings than any one form uses — the death
#: certificate's issue date is three cells on one variant and two on
#: another — and the evidence is whichever of them reached the page.
DATE_GROUPS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "death_certificate": {
        "death": ("death_year", "death_month", "death_day"),
        "record": ("record_year", "record_month", "record_day"),
        "issue": ("issue_year", "issue_month", "issue_day", "issue_day_month"),
    },
    "birth_certificate": {
        "birth": ("child_birth_date",),
        "record": ("record_year", "record_month", "record_day"),
        "issue": ("issue_year", "issue_month", "issue_day"),
    },
    "ariza": {"filed": ("date",)},
}


def build_facts(
    document_type: str,
    fields: Mapping[str, Any],
    dates: Mapping[str, str] | None = None,
) -> list[Fact]:
    """Read the structured facts off one document's field values.

    Args:
        document_type: Which document the fields belong to.
        fields: The field values as written on the page.
        dates: Normalised ``YYYY-MM-DD`` dates for the document's date
            groups, keyed as in :data:`DATE_GROUPS`.

    Returns:
        The facts, dates first and then the remaining fields in the order
        the mapping declares them.

    Raises:
        KeyError: If no fact mapping is registered for that document type.
    """
    try:
        mapping = FIELD_FACTS[document_type]
    except KeyError as error:
        known = ", ".join(sorted(FIELD_FACTS))
        raise KeyError(
            f"No fact mapping for {document_type!r}. Known types: {known}"
        ) from error

    facts: list[Fact] = []
    grouped: set[str] = set()

    for name, members in DATE_GROUPS.get(document_type, {}).items():
        evidence = [str(fields[part]) for part in members if fields.get(part)]
        if not evidence:
            continue
        grouped.update(members)
        value = (dates or {}).get(name) or " ".join(evidence)
        facts.append(
            Fact(
                category="date",
                value=value,
                evidence_text=" ".join(evidence),
                field=name,
            )
        )

    spelled = YEAR_IN_WORDS.get(document_type, {})
    for name, declared in mapping.items():
        if name in grouped:
            continue
        raw = fields.get(name)
        if not raw:
            continue

        written = " ".join(raw) if isinstance(raw, list) else str(raw)
        value = written
        if name in spelled:
            # "ikki ming o'n beshinchi" means 2015; say so, and keep the
            # words as the evidence they were read from.
            dated = (dates or {}).get(spelled[name], "")
            value = dated[:4] or written

        facts.append(
            Fact(
                category=declared.category,
                value=value,
                evidence_text=written,
                subtype=declared.subtype,
                field=name,
            )
        )
    return facts
