"""Tests for bitikocr.data.synthetic.fonts."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from bitikocr.data.synthetic.fonts import (
    FALLBACK_BASE,
    FontInfo,
    FontLibrary,
    pixel_size_for,
)

UZBEK_TEXT = "Ҳақиқий ўзбек ёзуви ғалаба қилди"


def test_packaged_fonts_are_discovered(library: FontLibrary) -> None:
    assert len(library) > 0
    assert all(font.path.is_file() for font in library)


def test_a_font_is_found_by_name(library: FontLibrary) -> None:
    name = library.fonts[0].name
    assert library.by_name(name) is library.fonts[0]


def test_an_unknown_name_resolves_to_nothing(library: FontLibrary) -> None:
    assert library.by_name("NoSuchFont") is None


def test_uzbek_letters_are_rendered_through_their_base_glyph() -> None:
    info = FontInfo(
        path=Path("dummy.ttf"),
        name="dummy",
        codepoints=frozenset(ord(char) for char in FALLBACK_BASE.values()),
    )
    assert info.can_render("".join(FALLBACK_BASE))


def test_a_font_without_the_base_glyph_cannot_render() -> None:
    info = FontInfo(
        path=Path("dummy.ttf"),
        name="dummy",
        codepoints=frozenset({ord("a")}),
    )
    assert not info.can_render("ҳ")


def test_whitespace_never_blocks_rendering() -> None:
    info = FontInfo(
        path=Path("dummy.ttf"),
        name="dummy",
        codepoints=frozenset({ord("a")}),
    )
    assert info.can_render("a a\na")


def test_picking_returns_a_font_covering_the_text(library: FontLibrary) -> None:
    chosen = library.pick(UZBEK_TEXT, random.Random(0))
    assert chosen.can_render(UZBEK_TEXT)


def test_picking_fails_when_no_font_covers_the_text(
    library: FontLibrary,
) -> None:
    with pytest.raises(ValueError, match="No available font"):
        library.pick("漢字とかな", random.Random(0))


def test_picking_an_explicit_font_bypasses_sampling(
    library: FontLibrary,
) -> None:
    wanted = library.fonts[0]
    chosen = library.pick("", random.Random(0), font_path=wanted.path)
    assert chosen.name == wanted.name


def test_an_empty_library_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one font"):
        FontLibrary([])


def test_pixel_size_compensates_for_small_x_heights(
    library: FontLibrary,
) -> None:
    narrow = library.fonts[0]
    squashed = FontInfo(
        path=narrow.path,
        name=narrow.name,
        codepoints=narrow.codepoints,
        xheight_ratio=0.21,
        width_ratio=0.23,
    )
    assert pixel_size_for(squashed, 60) > 60


def test_scanning_a_directory_without_fonts_fails(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No usable fonts"):
        FontLibrary.from_directory(tmp_path)


def test_latin_only_fonts_are_measured_on_latin_glyphs(
    library: FontLibrary,
) -> None:
    """Measuring a Latin font on Cyrillic would read the notdef box instead.

    That used to report a stroke as wide as the em square, which made the
    renderer erode the whole line away.
    """
    latin_only = [
        font for font in library if not font.can_render("мамлакатнинг")
    ]
    assert latin_only, "expected at least one Latin-only packaged font"
    for font in latin_only:
        assert font.stroke_ratio < 0.3, font.name
