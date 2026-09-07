"""Tests for bitikocr.data.synthetic.layout and the handwriting renderer."""

from __future__ import annotations

import random

from bitikocr.data.synthetic.fonts import FontLibrary
from bitikocr.data.synthetic.hand import Hand
from bitikocr.data.synthetic.layout import Page, wrap_text
from bitikocr.data.synthetic.style import HandwritingStyle

TEXT = "Ҳақиқий ўзбек ёзуви ғалаба қилди ва яна давом этди"


def make_hand(library: FontLibrary, style: HandwritingStyle) -> Hand:
    """Build a hand from the first font that can write the sample text."""
    info = library.pick(TEXT, random.Random(0))
    return Hand(info, 48, style, random.Random(0))


def test_wrapping_keeps_every_word(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    hand = make_hand(library, style)
    lines = wrap_text(TEXT, hand, max_width=300)
    assert " ".join(lines).split() == TEXT.split()


def test_wrapping_respects_the_available_width(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    hand = make_hand(library, style)
    lines = wrap_text(TEXT, hand, max_width=400)
    # A line may only overflow when it holds a single unbreakable word.
    assert all(hand.measure(line) <= 400 or " " not in line for line in lines)


def test_wrapping_splits_on_paragraph_breaks(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    hand = make_hand(library, style)
    lines = wrap_text("бир\n\nикки", hand, max_width=10_000)
    assert lines == ["бир", "икки"]


def test_wrapping_empty_text_yields_no_lines(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    hand = make_hand(library, style)
    assert wrap_text("   \n  ", hand, max_width=500) == []


def test_measuring_grows_with_the_text(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    hand = make_hand(library, style)
    assert hand.measure("аб") > hand.measure("а") > 0


def test_a_page_records_a_line_per_written_line(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    hand = make_hand(library, style)
    page = Page((800, 600), style, random.Random(0))
    page.put_lines("body", ["бир", "икки"], hand, 40, 200)

    assert [line.text for line in page.lines] == ["бир", "икки"]
    assert [block.kind for block in page.blocks] == ["body"]
    assert page.blocks[0].text == "бир\nикки"


def test_a_page_can_skip_recording_a_block(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    hand = make_hand(library, style)
    page = Page((800, 600), style, random.Random(0))
    page.put_lines("body", ["бир"], hand, 40, 200, record_block=False)

    assert page.lines and not page.blocks


def test_a_block_box_encloses_its_line_boxes(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    hand = make_hand(library, style)
    page = Page((800, 600), style, random.Random(0))
    page.put_lines("body", ["бир", "икки"], hand, 40, 200)

    block = page.blocks[0].bbox
    assert block is not None
    for line in page.lines:
        assert line.bbox is not None
        assert block.left <= line.bbox.left
        assert block.right >= line.bbox.right


def test_the_annotation_follows_the_reading_order(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    hand = make_hand(library, style)
    page = Page((800, 600), style, random.Random(0))
    page.put_lines("body", ["икки"], hand, 40, 300)
    page.put_lines("title", ["бир"], hand, 40, 150)

    annotation = page.annotation(("title", "body"))
    assert annotation.text == "бир\nикки"
    assert annotation.size == (800, 600)


def test_blocks_outside_the_reading_order_are_not_transcribed(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    hand = make_hand(library, style)
    page = Page((800, 600), style, random.Random(0))
    page.put_lines("body", ["бир"], hand, 40, 200)
    page.put_lines("scratch", ["икки"], hand, 40, 400)

    assert page.annotation(("body",)).text == "бир"


def test_the_page_uses_the_style_paper_when_given_no_background(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    page = Page((40, 30), style, random.Random(0))
    assert page.render().getpixel((0, 0)) == style.paper


def test_every_font_and_pen_leaves_visible_ink(
    library: FontLibrary, style: HandwritingStyle
) -> None:
    """A line that renders blank would still claim text in the ground truth."""
    for font in library:
        text = "handwriting" if not font.can_render(TEXT) else TEXT
        for pen in ("hard", "soft", "gel"):
            hand = Hand(
                font,
                48,
                style.replace(pen=pen, stroke_px=1.0),
                random.Random(0),
            )
            rendered, _ = hand.render_line(text, style.ink)
            assert (
                rendered.getchannel("A").getbbox() is not None
            ), f"{font.name} wrote nothing with a {pen} pen"
