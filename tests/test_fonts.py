"""Tests for bitikocr.data.synthetic.fonts."""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from bitikocr.data.synthetic.fonts import (
    FALLBACK_BASE,
    FontInfo,
    FontLibrary,
    pixel_size_for,
)
from bitikocr.data.synthetic.hand import Hand
from bitikocr.data.synthetic.style import sample_style

UZBEK_TEXT = "Ҳақиқий ўзбек ёзуви ғалаба қилди"

#: A title is the largest thing written on a page, and erosion is counted in
#: pixels, so a stroke that survives at body size can still be eroded away
#: here.
TITLE_PIXEL_SIZE = 110


def _ink(rendered: Image.Image) -> float:
    """Return the total opacity of a rendered line."""
    return float(np.asarray(rendered.getchannel("A")).astype(float).sum())


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


def test_scanning_a_directory_without_fonts_fails(tmp_path: Path) -> None:
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


def test_the_pen_never_eats_more_than_half_a_hands_ink(
    library: FontLibrary,
) -> None:
    """A stroke width read off the heaviest downstrokes shreds a hand.

    A cursive font is not one width: it has thick downstrokes and hairline
    joins. Measuring until the ink was gone reported the thickest stroke,
    the renderer then thinned the hand by more than the joins could take,
    and a title came out as disconnected fragments while its ground truth
    still claimed a readable line. MixHand02 lost 72% of its ink that way.

    The size matters: erosion is measured in pixels, so the fault only
    shows at the large sizes a title is written in.
    """
    text = "Розилик хати"
    style = sample_style(random.Random(7), library).replace(ink_strength=1.4)

    for font in library:
        if not font.can_render(text):
            continue
        hand = Hand(
            font,
            TITLE_PIXEL_SIZE,
            style.replace(font=font.name),
            random.Random(7),
        )
        drawn, _ = hand.render_line(text, style.ink)

        with patch.object(Hand, "_erode", lambda self, ink, passes: ink):
            untouched, _ = hand.render_line(text, style.ink)

        kept = _ink(drawn) / _ink(untouched)
        assert kept >= 0.5, f"{font.name} lost {(1 - kept):.0%} of its ink"


def test_a_high_contrast_font_is_measured_by_its_typical_stroke(
    library: FontLibrary,
) -> None:
    """MixHand02 is the library's most contrasted hand, so it is the one
    that shows whether the measurement follows the heavy strokes."""
    font = library.by_name("MixHand02")
    assert font is not None
    assert font.stroke_ratio <= 0.05, (
        "measured on the heaviest strokes again; the thin joins will be "
        "eroded away"
    )
