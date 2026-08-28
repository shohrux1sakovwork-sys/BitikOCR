"""Tests for bitikocr.synthetic.style."""

from __future__ import annotations

import json
import random

import pytest

from bitikocr.synthetic.fonts import FontLibrary
from bitikocr.synthetic.style import HandwritingStyle, sample_style


def test_the_same_seed_samples_the_same_style(library: FontLibrary) -> None:
    first = sample_style(random.Random(7), library)
    second = sample_style(random.Random(7), library)
    assert first == second


def test_different_seeds_sample_different_styles(library: FontLibrary) -> None:
    first = sample_style(random.Random(7), library)
    second = sample_style(random.Random(8), library)
    assert first != second


def test_the_sampled_font_belongs_to_the_library(
    library: FontLibrary,
) -> None:
    style = sample_style(random.Random(3), library)
    assert library.by_name(style.font) is not None
    assert library.by_name(style.second_font) is not None


def test_replace_leaves_the_original_untouched(
    style: HandwritingStyle,
) -> None:
    changed = style.replace(font_size=11)
    assert changed.font_size == 11
    assert style.font_size != 11


def test_replace_rejects_unknown_fields(style: HandwritingStyle) -> None:
    with pytest.raises(ValueError, match="Unknown style fields: nonsense"):
        style.replace(nonsense=1)


def test_the_style_serialises_to_json(style: HandwritingStyle) -> None:
    payload = json.loads(json.dumps(style.to_dict()))
    assert payload["ink"] == list(style.ink)
    assert payload["paper"] == list(style.paper)
    assert payload["pen"] in {"hard", "soft", "gel"}


def test_sampled_colors_stay_inside_the_byte_range(
    library: FontLibrary,
) -> None:
    for seed in range(20):
        style = sample_style(random.Random(seed), library)
        assert all(0 <= channel <= 255 for channel in style.ink)
        assert all(0 <= channel <= 255 for channel in style.paper)
