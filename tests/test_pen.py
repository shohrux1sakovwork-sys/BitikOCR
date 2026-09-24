"""Tests for bitikocr.data.synthetic.pen."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from bitikocr.data.synthetic.pen import centre_line, redraw_ballpoint


def thick_letter() -> Image.Image:
    """Draw an "L" with a broad nib: a thick upright and a thick foot."""
    ink = Image.new("L", (200, 160), 0)
    draw = ImageDraw.Draw(ink)
    draw.rectangle([40, 20, 58, 130], fill=255)
    draw.rectangle([40, 112, 160, 130], fill=255)
    return ink


def test_a_broad_stroke_thins_to_a_one_pixel_line() -> None:
    bar = np.zeros((40, 120), dtype=bool)
    bar[10:30, 10:110] = True
    path = centre_line(bar)
    # Every column the bar crossed keeps exactly one pixel of path, near
    # the middle; the ends may shorten by the bar's half-width.
    columns = path[:, 25:95].sum(axis=0)
    assert (columns == 1).all()
    rows = np.nonzero(path[:, 60])[0]
    assert 17 <= rows[0] <= 22


def test_thinning_keeps_a_stroke_in_one_piece() -> None:
    path = centre_line(np.asarray(thick_letter()) > 0)
    ys, xs = np.nonzero(path)
    assert len(ys) > 0
    # The corner of the "L" survives: the path still reaches both the top
    # of the upright and the end of the foot.
    assert ys.min() < 40 and xs.max() > 140


def test_a_redrawn_line_is_finer_than_the_font_drew_it() -> None:
    ink = thick_letter()
    redrawn = redraw_ballpoint(ink, width=2.0, size=60, rng=random.Random(0))
    before = (np.asarray(ink) > 128).sum()
    after = (np.asarray(redrawn) > 128).sum()
    assert 0 < after < before / 4


def test_a_redrawn_line_stays_where_the_letter_was() -> None:
    ink = thick_letter()
    redrawn = redraw_ballpoint(ink, width=3.0, size=60, rng=random.Random(0))
    assert redrawn.size == ink.size
    before, after = ink.getbbox(), redrawn.getbbox()
    assert before is not None and after is not None
    left, top, right, bottom = before
    new_left, new_top, new_right, new_bottom = after
    assert left <= new_left and top <= new_top
    assert new_right <= right and new_bottom <= bottom


def test_nothing_redrawn_stays_nothing() -> None:
    blank = Image.new("L", (50, 50), 0)
    assert redraw_ballpoint(blank, 2.0, 60, random.Random(0)).getbbox() is None


def test_the_same_seed_redraws_the_same_line() -> None:
    first = redraw_ballpoint(thick_letter(), 2.0, 60, random.Random(3))
    second = redraw_ballpoint(thick_letter(), 2.0, 60, random.Random(3))
    assert first.tobytes() == second.tobytes()


def test_a_hand_shrunk_into_a_narrow_cell_is_still_traced() -> None:
    """Its strokes are too fine for any pixel to reach full ink; the value
    must not vanish while the label still claims it."""
    faint = Image.new("L", (80, 40), 0)
    ImageDraw.Draw(faint).line([(10, 30), (30, 8), (50, 30)], fill=90, width=2)
    redrawn = redraw_ballpoint(faint, 1.0, 20, random.Random(0))
    assert redrawn.getbbox() is not None
    assert int(np.asarray(redrawn).max()) > 60
