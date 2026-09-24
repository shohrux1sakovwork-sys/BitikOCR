"""Tests for bitikocr.data.synthetic.transcript."""

from __future__ import annotations

from bitikocr.data.models.geometry import BoundingBox
from bitikocr.data.synthetic.transcript import (
    SIGNATURE_MARK,
    Piece,
    compose,
    stamp_markup,
)


def box(left: int, top: int, right: int, bottom: int) -> BoundingBox:
    """Build a box from its four edges."""
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def test_a_seal_is_wrapped_in_stamp_markup() -> None:
    assert stamp_markup("XORAZM VILOYATI\nKELGAN № 658") == (
        "<stamp>\nXORAZM VILOYATI\nKELGAN № 658\n</stamp>"
    )


def test_a_seal_with_no_lettering_is_an_empty_stamp() -> None:
    assert stamp_markup(" \n") == "<stamp/>"


def test_lines_are_read_top_to_bottom_whatever_order_they_came_in() -> None:
    pieces = [
        Piece("икки", box(10, 200, 200, 240), baseline=235),
        Piece("бир", box(10, 100, 200, 140), baseline=135),
    ]
    assert compose(pieces) == "бир\nикки"


def test_a_value_reads_on_the_row_of_the_label_printed_on_its_rule() -> None:
    """The label sits on the rule; the value was written above it, and its
    ink is much taller than the label's, but they share the rule."""
    pieces = [
        Piece("Tug‘ilgan vaqti:", box(45, 429, 217, 449), "print"),
        Piece("1999 yil fevral 20", box(300, 395, 600, 455), baseline=449),
    ]
    assert compose(pieces) == "Tug‘ilgan vaqti: 1999 yil fevral 20"


def test_a_caption_under_the_rule_is_a_row_of_its_own() -> None:
    """A value's descenders reach into the caption printed under its rule;
    that must not pull the caption onto the value's row."""
    pieces = [
        Piece("Mirjalol Zokir o'g'li", box(800, 150, 1200, 195), baseline=183),
        Piece("(ismi, otasining ismi)", box(922, 181, 1241, 203), "print"),
    ]
    assert compose(pieces) == "Mirjalol Zokir o'g'li\n(ismi, otasining ismi)"


def test_a_short_word_joins_the_row_of_its_rule() -> None:
    """A word with a descender and no ascender sits low; its rule does
    not."""
    pieces = [
        Piece("Bu haqida", box(51, 804, 166, 825), "print"),
        Piece("may", box(300, 808, 380, 840), baseline=824),
        Piece("oyining", box(433, 804, 526, 825), "print"),
        Piece("Об этом", box(50, 823, 151, 840), "print"),
    ]
    assert compose(pieces) == "Bu haqida may oyining\nОб этом"


def test_a_signature_is_marked_where_it_was_signed() -> None:
    pieces = [
        Piece("Ashurov Sh.", box(200, 1000, 500, 1060), baseline=1050),
        Piece(SIGNATURE_MARK, box(900, 960, 1200, 1080), "signature"),
        Piece("13.09.1994", box(900, 1150, 1200, 1200), baseline=1195),
    ]
    assert compose(pieces) == "Ashurov Sh. <signature>\n13.09.1994"


def test_a_seal_is_read_where_it_sits_and_joins_no_row() -> None:
    pieces = [
        Piece("Muhr", box(851, 630, 920, 644), "print"),
        Piece("FHDYO", box(800, 610, 1100, 690), "stamp"),
        Piece("Печать", box(847, 649, 928, 663), "print"),
    ]
    assert compose(pieces) == ("Muhr\n<stamp>\nFHDYO\n</stamp>\nПечать")


def test_facing_sheets_are_read_one_after_the_other() -> None:
    pieces = [
        Piece("Otasi", box(767, 110, 840, 127), "print"),
        Piece("O'LIM HAQIDA GUVOHNOMA", box(178, 234, 535, 258), "print"),
        Piece("Vafot etgan joyi:", box(93, 815, 228, 834), "print"),
    ]
    assert compose(pieces, columns=[(0, 710), (710, 1419)]) == (
        "O'LIM HAQIDA GUVOHNOMA\nVafot etgan joyi:\n\nOtasi"
    )


def test_a_piece_past_every_sheet_goes_to_the_nearest() -> None:
    pieces = [Piece("I-HR №", box(1500, 10, 1600, 30), "print")]
    assert compose(pieces, columns=[(0, 700), (700, 1400)]) == "I-HR №"
