"""Tests for bitikocr.data.synthetic.scripts."""

from __future__ import annotations

import random

import pytest

from bitikocr.data.synthetic.scripts import (
    in_script,
    russian_spelling,
    spell_as_writer,
    to_cyrillic,
)

# Latin spelling mapped to the Cyrillic a registry clerk would write.
KNOWN_PAIRS = [
    ("Toshkent", "Тошкент"),
    ("Xorazm", "Хоразм"),
    ("O'zbekiston", "Ўзбекистон"),
    ("Yangiariq", "Янгиариқ"),
    ("Gulnora", "Гулнора"),
    ("To'rayevna", "Тўраевна"),
    ("Shavkat", "Шавкат"),
    ("Urganch", "Урганч"),
    ("Sirdaryo", "Сирдарё"),
    ("Farg'ona", "Фарғона"),
    ("Qashqadaryo", "Қашқадарё"),
    ("hokimi", "ҳокими"),
]


@pytest.mark.parametrize(("latin", "cyrillic"), KNOWN_PAIRS)
def test_uzbek_latin_transliterates_to_cyrillic(
    latin: str, cyrillic: str
) -> None:
    assert to_cyrillic(latin) == cyrillic


def test_yo_before_an_apostrophe_is_two_letters() -> None:
    """ "Yo'ldoshev" is y + o', not yo + tutuq belgisi."""
    assert to_cyrillic("Yo'ldoshev") == "Йўлдошев"


def test_a_word_initial_e_is_the_open_vowel() -> None:
    assert to_cyrillic("Ergashev") == "Эргашев"
    assert to_cyrillic("Bekchanov") == "Бекчанов"


def test_capitalisation_survives() -> None:
    assert to_cyrillic("O'ZBEKISTON") == "ЎЗБЕКИСТОН"
    assert to_cyrillic("o'zbekiston") == "ўзбекистон"
    assert to_cyrillic("O'zbekiston") == "Ўзбекистон"


def test_digits_and_punctuation_pass_through() -> None:
    assert to_cyrillic("1-2108-20-T-003") == "1-2108-20-Т-003"
    assert to_cyrillic("18.03.2018") == "18.03.2018"


def test_spacing_is_preserved() -> None:
    assert to_cyrillic("Urganch shahri") == "Урганч шаҳри"


def test_a_plain_entry_is_transliterated() -> None:
    assert in_script("Toshkent", "latin") == "Toshkent"
    assert in_script("Toshkent", "cyrillic") == "Тошкент"


def test_a_bilingual_entry_uses_its_given_spelling() -> None:
    """Some words the rules get wrong, so both spellings are given."""
    entry = ("sentabr", "сентябр")
    assert in_script(entry, "latin") == "sentabr"
    assert in_script(entry, "cyrillic") == "сентябр"
    assert to_cyrillic("sentabr") != "сентябр"


def test_a_russian_schooled_writer_uses_russian_letters_and_forms() -> None:
    assert russian_spelling("Иброҳимова") == "Ибрагимова"
    assert russian_spelling("Аҳмадова") == "Ахмедова"
    assert russian_spelling("Қосимов") == "Косимов"
    assert russian_spelling("ҳокими") == "хокими"


def test_a_standard_writer_changes_nothing() -> None:
    text = "Гулистон шаҳар ҳокими"
    assert spell_as_writer(text, "cyrillic", 0.0, "'", random.Random(0)) == text


def test_a_latin_writer_types_the_tutuq_their_own_way() -> None:
    written = spell_as_writer(
        "O'zbekiston g'alla", "latin", 1.0, "‘", random.Random(0)
    )
    assert written == "O‘zbekiston g‘alla"


def test_a_mixed_writer_keeps_line_breaks() -> None:
    text = "Қосимов\nҲамроев Ўткир"
    written = spell_as_writer(text, "cyrillic", 0.5, "'", random.Random(2))
    assert written.count("\n") == 1
    assert len(written.split()) == 3
