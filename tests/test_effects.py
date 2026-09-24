"""Tests for bitikocr.data.synthetic.effects."""

from __future__ import annotations

import random

from PIL import Image

from bitikocr.data.models.geometry import BoundingBox
from bitikocr.data.synthetic.effects import (
    StampBlank,
    box_stamp_size,
    draw_box_stamp,
)
from bitikocr.data.synthetic.fonts import FontLibrary
from bitikocr.data.synthetic.system_fonts import find_print_font

ROWS = ["XORAZM VILOYATI", "URGANCH SHAHAR HOKIMI", "KELGAN №", ""]


def press(
    library: FontLibrary,
) -> tuple[Image.Image, BoundingBox | None, list[StampBlank]]:
    """Press the sample stamp on a blank page."""
    page = Image.new("RGBA", (1200, 800), (255, 255, 255, 255))
    box, blanks = draw_box_stamp(
        page=page,
        top_left=(100, 200),
        rows=ROWS,
        type_size=30,
        blank_width=150,
        color=(60, 30, 130),
        font_path=find_print_font(library),
        rng=random.Random(0),
    )
    return page, box, blanks


def test_a_box_stamp_leaves_room_after_the_number_sign_and_in_its_empty_row(
    library: FontLibrary,
) -> None:
    _, _, blanks = press(library)
    number, date = blanks
    assert number.baseline < date.baseline
    assert number.right - number.left >= 150 * 0.8
    # The empty row spans the stamp; the number's room only follows the sign.
    assert date.left < number.left


def test_a_box_stamp_lands_where_it_was_measured_to(
    library: FontLibrary,
) -> None:
    _, box, _ = press(library)
    width, height = box_stamp_size(ROWS, 30, 150, find_print_font(library))
    assert box is not None
    # A slight tilt may spread the ink by a few pixels, never by a row.
    assert abs(box.left - 100) <= 12 and abs(box.top - 200) <= 12
    assert abs(box.width - width) <= 24 and abs(box.height - height) <= 24


def test_every_room_in_a_box_stamp_is_inside_its_frame(
    library: FontLibrary,
) -> None:
    _, box, blanks = press(library)
    assert box is not None
    for blank in blanks:
        assert box.left <= blank.left < blank.right <= box.right
        assert box.top < blank.baseline < box.bottom
