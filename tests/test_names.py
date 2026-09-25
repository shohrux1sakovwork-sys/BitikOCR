"""Tests for bitikocr.data.synthetic.names."""

from __future__ import annotations

import pytest

from bitikocr.data.synthetic.names import (
    FEMALE_NAMES,
    MALE_NAMES,
    SURNAME_STEMS,
)
from bitikocr.data.synthetic.scripts import to_cyrillic


@pytest.mark.parametrize("names", [MALE_NAMES, FEMALE_NAMES, SURNAME_STEMS])
def test_no_name_is_listed_twice(names: tuple[str, ...]) -> None:
    assert len(names) == len(set(names))


def test_surname_stems_do_not_repeat_the_given_names() -> None:
    """Every male name is a surname stem already."""
    assert not set(SURNAME_STEMS) & set(MALE_NAMES)


def test_the_pool_is_too_wide_to_learn_by_heart() -> None:
    assert len(set(SURNAME_STEMS) | set(MALE_NAMES)) >= 300
    assert len(FEMALE_NAMES) >= 150


@pytest.mark.parametrize("name", [*MALE_NAMES, *FEMALE_NAMES, *SURNAME_STEMS])
def test_every_name_transliterates_to_cyrillic(name: str) -> None:
    cyrillic = to_cyrillic(name)
    assert cyrillic and not any("a" <= char.lower() <= "z" for char in cyrillic)
