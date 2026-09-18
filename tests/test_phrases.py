"""Tests for bitikocr.data.synthetic.phrases."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from bitikocr.data.synthetic.phrases import (
    PHRASE_SLOTS,
    PhraseBank,
    normalise_phrase,
    phrase_problem,
)
from bitikocr.data.synthetic.records import sample_record


def test_normalising_fixes_apostrophes_markers_and_spacing() -> None:
    assert normalise_phrase("  2. “O‘gʻil  bola .”") == "O'g'il bola."


@pytest.mark.parametrize(
    ("slot", "text"),
    [
        ("ariza_request", "uyimga gaz o'tkazib berishingizni"),
        ("explanation_reason", "onamning kasal bo'lib qolgani"),
        ("consent_closing", "Ushbu xatni o'z ixtiyorim bilan yozdim."),
    ],
)
def test_well_formed_phrases_are_accepted(slot: str, text: str) -> None:
    assert phrase_problem(text, PHRASE_SLOTS[slot]) is None


@pytest.mark.parametrize(
    ("slot", "text", "reason"),
    [
        ("ariza_request", "uyimga 3 ta gaz berishingizni", "characters"),
        ("ariza_request", "moddiy yordam berib turishingiz", "must end"),
        ("explanation_reason", "bo'lmasligi kani", "stands alone"),
        ("explanation_reason", "Onamning kasalligi", "lower case"),
        ("consent_closing", "Bunga to'liq roziman", "full stop"),
        ("consent_closing", "Men Karimovga roziman.", "proper names"),
        ("consent_closing", "Men cabinetga roziman.", "does not use"),
        ("ariza_detail", "Juda.", "words"),
    ],
)
def test_malformed_phrases_are_rejected(
    slot: str, text: str, reason: str
) -> None:
    problem = phrase_problem(text, PHRASE_SLOTS[slot])
    assert problem is not None and reason in problem


def test_a_bank_keeps_only_new_valid_phrases() -> None:
    bank = PhraseBank()
    assert (
        bank.add("ariza_request", "menga nafaqa tayinlashingizni", "m") is None
    )
    assert bank.add("ariza_request", "Menga nafaqa tayinlashingizni", "m")
    assert bank.add("ariza_request", "nafaqa", "m")
    assert bank.count("ariza_request") == 1


def test_a_bank_survives_a_round_trip(tmp_path: Path) -> None:
    bank = PhraseBank()
    bank.add("explanation_task", "yillik hisobotni", "model-a")
    path = tmp_path / "bank.json"
    bank.save(path)
    loaded = PhraseBank.load(path)
    assert list(loaded) == [("explanation_task", "yillik hisobotni", "model-a")]


def test_samplers_draw_from_the_bank() -> None:
    bank = PhraseBank()
    bank.add("ariza_request", "ko'chamizga yoritgich o'rnatishingizni", "m")
    bodies = {
        str(
            sample_record(
                "ariza", random.Random(seed), "latin", phrases=bank
            ).fields["body"]
        )
        for seed in range(20)
    }
    assert any("yoritgich" in body for body in bodies)


def test_samplers_transliterate_bank_wording() -> None:
    bank = PhraseBank()
    bank.add("ariza_request", "ko'chamizga yoritgich o'rnatishingizni", "m")
    bodies = [
        str(
            sample_record(
                "ariza", random.Random(seed), "cyrillic", phrases=bank
            ).fields["body"]
        )
        for seed in range(20)
    ]
    assert any("ёритгич" in body for body in bodies)
    assert not any("yoritgich" in body for body in bodies)
