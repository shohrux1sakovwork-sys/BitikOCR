"""Tests for the neighbourhood committee's certificate."""

from __future__ import annotations

import random
from typing import Any

import pytest

from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic.generators import (
    SyntheticDocument,
    create_generator,
)
from bitikocr.data.synthetic.generators.malumotnoma import (
    MAX_FAMILY_ROWS,
    MalumotnomaGenerator,
)
from bitikocr.data.synthetic.records import sample_record, sample_records


def _render(
    config: SyntheticConfig, seed: int
) -> tuple[dict[str, Any], SyntheticDocument]:
    record = sample_record("malumotnoma", random.Random(seed))
    generator = create_generator("malumotnoma", config)
    return record.fields, generator.generate(record.fields, seed=record.seed)


def test_most_certificates_are_filled_in_cyrillic() -> None:
    records = sample_records("malumotnoma", 200, random.Random(1))
    cyrillic = sum(record.script == "cyrillic" for record in records)
    assert cyrillic >= 170


def test_a_household_never_lists_someone_not_yet_born() -> None:
    for record in sample_records("malumotnoma", 200, random.Random(2)):
        for name, value in record.fields.items():
            if name.startswith("member_"):
                years = [int(word) for word in value.split() if word.isdigit()]
                assert all(year <= 2023 for year in years), value


def test_a_household_fits_the_blank() -> None:
    for record in sample_records("malumotnoma", 200, random.Random(3)):
        members = [name for name in record.fields if name.startswith("member_")]
        assert len(members) <= MAX_FAMILY_ROWS


def test_the_page_reads_title_holder_and_family_in_order(
    config: SyntheticConfig,
) -> None:
    fields, document = _render(config, 4)
    text = document.annotation.text
    title = text.index("МАЪЛУМОТНОМА")
    holder = text.index(str(fields["holder"]).split()[0])
    assert title < holder
    if fields.get("member_1"):
        assert holder < text.index("1.")


def test_empty_family_rows_keep_their_printed_numbers(
    config: SyntheticConfig,
) -> None:
    for seed in range(10):
        fields, document = _render(config, seed)
        members = sum(1 for name in fields if name.startswith("member_"))
        lines = document.annotation.text.splitlines()
        numbered = [line for line in lines if line.split(".")[0].isdigit()]
        assert len(numbered) >= members


def test_every_page_carries_the_seal_and_a_signature(
    config: SyntheticConfig,
) -> None:
    for seed in range(5):
        _, document = _render(config, seed)
        text = document.annotation.text
        assert "<stamp>" in text
        assert "<signature>" in text


def test_the_blank_prints_uzbek_letters(config: SyntheticConfig) -> None:
    _, document = _render(config, 6)
    assert "ҳақ" in document.annotation.text


def test_the_page_is_a4_wide(config: SyntheticConfig) -> None:
    for seed in range(5):
        _, document = _render(config, seed)
        width, height = document.image.size
        assert width == 1654
        assert 900 <= height <= 2339


def test_the_same_seed_draws_the_same_page(config: SyntheticConfig) -> None:
    _, first = _render(config, 7)
    _, second = _render(config, 7)
    assert first.image.tobytes() == second.image.tobytes()
    assert first.annotation.text == second.annotation.text


def test_a_template_is_refused(config: SyntheticConfig) -> None:
    with pytest.raises(ValueError, match="drawn, not chosen"):
        MalumotnomaGenerator.with_template(config, None, "anything")


def test_sons_are_often_named_differently_from_their_father() -> None:
    """A male holder's sons usually carry their grandfather's name."""
    differs = total = 0
    for record in sample_records("malumotnoma", 300, random.Random(8), "latin"):
        holder = record.fields["holder"].split()[0]
        if holder.endswith("a"):
            continue
        for name, value in record.fields.items():
            if name.startswith("member_") and "o'g'li" in value:
                total += 1
                # One entry format puts the relation first, in lower case.
                surname = next(w for w in value.split() if w[0].isupper())
                differs += surname != holder
    assert total > 50
    assert 0.4 < differs / total < 0.9
