"""Layout for a handwritten Uzbek "Ariza" (application letter).

The arrangement a letter shares with every other letter — the addressee
block, the title, the body, the signature — lives in
:mod:`bitikocr.data.synthetic.generators.letter`. What is left here is what
belongs to an ariza alone: the marks the receiving office writes on it once
it has been filed.

Most offices press a rectangular incoming stamp and write the filing number
and date into it; the rest write them on the page alone. Either way it is
the clerk's hand, not the applicant's.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import ClassVar

from bitikocr.data.models.geometry import BoundingBox
from bitikocr.data.synthetic.effects import (
    NUMBER_SIGN,
    SEAL_COLORS,
    StampBlank,
    box_stamp_size,
    draw_box_stamp,
)
from bitikocr.data.synthetic.generators.base import FieldValues, PartKind
from bitikocr.data.synthetic.generators.letter import (
    DEFAULT_PAGE_SIZE,
    LetterGenerator,
)
from bitikocr.data.synthetic.hand import Hand
from bitikocr.data.synthetic.layout import Page
from bitikocr.data.synthetic.style import INK_PALETTE, HandwritingStyle
from bitikocr.data.synthetic.system_fonts import find_print_font

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_TITLE",
    "REGISTRATION_STAMP_FIELD",
    "ArizaGenerator",
]

DEFAULT_TITLE = "Ариза"

#: Field holding the rows of the office's incoming stamp.
REGISTRATION_STAMP_FIELD = "reg_stamp"

# The stamp's lettering, as a share of the letter's nominal hand size.
_STAMP_TYPE_SCALE = (0.42, 0.55)

# Where the stamp's left edge lands, as a share of the page width, and how
# close to the right edge its frame may come.
_STAMP_LEFT = (0.38, 0.60)
_STAMP_RIGHT_LIMIT = 0.97

# The stamp goes below the signature, no higher than this share of the page
# and never closer than this many nominal sizes to its foot.
_STAMP_TOP = (0.68, 0.80)
_STAMP_FOOT_MARGIN = 1.0

# The other place an office stamps a letter: the empty top-left corner,
# beside the addressee block, clear of it by this many nominal sizes.
_CORNER_SHARE = 0.3
_CORNER_LEFT = (0.03, 0.08)
_CORNER_TOP = (0.015, 0.05)
_CORNER_GAP = 0.5

# The clerk's hand inside a stamp is at most this many times the stamp's
# type, so it stays in its row, and shrinks no further than this to fit.
_ENTRY_SCALE = 1.15
_MIN_ENTRY_SIZE = 18
_ENTRY_SHRINK = 0.9

#: Blocks for what the clerk writes into the stamp. The stamp's own block
#: carries their text, so these are regions of the page but not read on
#: their own.
_ENTRY_BLOCKS = ("reg_entry_number", "reg_entry_date")


class ArizaGenerator(LetterGenerator):
    """Render a one-page handwritten application letter.

    Args:
        config: Paths to the fonts and background templates to use.
        font_path: Force a specific handwriting font instead of sampling.
        page_size: Page size as ``(width, height)`` in pixels.
        ink_strength: How heavily the pen writes.
    """

    name: ClassVar[str] = "ariza"
    default_title: ClassVar[str] = DEFAULT_TITLE

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "recipient",
        "applicant",
        "body",
        "title",
        "signature_name",
        "date",
        "phone",
        "reg_number",
        "reg_date",
        "page_number",
        REGISTRATION_STAMP_FIELD,
    )

    # An ariza's phone number is written with the signature, and the office
    # registers it in a hand of its own.
    BLOCK_PARTS: ClassVar[Mapping[str, PartKind]] = {
        **LetterGenerator.BLOCK_PARTS,
        "phone": ("signature", "handwritten"),
        "registration": ("other", "handwritten"),
        **{block: ("other", "handwritten") for block in _ENTRY_BLOCKS},
    }

    READING_ORDER: ClassVar[tuple[str, ...]] = (
        "page_number",
        "recipient",
        "applicant",
        "title",
        "body",
        "signature_name",
        "date",
        "phone",
        "registration",
    )

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names an ariza understands."""
        return self.FIELD_NAMES

    @property
    def reading_order(self) -> tuple[str, ...]:
        """Blocks a human reads on the finished letter."""
        return self.READING_ORDER

    def _collect_text(self, fields: FieldValues) -> str:
        """Join what the applicant writes; the stamp is set in type."""
        return super()._collect_text(
            {
                name: value
                for name, value in fields.items()
                if name != REGISTRATION_STAMP_FIELD
            }
        )

    def _second_hand_text(self, fields: FieldValues) -> str:
        """Return the registration marks, which a clerk writes, not the author."""
        return " ".join(
            str(fields[name])
            for name in ("reg_number", "reg_date", "page_number")
            if fields.get(name)
        )

    def _put_foot(
        self,
        page: Page,
        fields: FieldValues,
        second_hand: Hand,
        rng: random.Random,
        style: HandwritingStyle,
        y: int,
    ) -> None:
        """File the letter: stamp it if the office did, and number it."""
        rows = fields.get(REGISTRATION_STAMP_FIELD)
        if rows and self._stamp(
            page, [str(row) for row in rows], fields, second_hand, rng, style, y
        ):
            return
        self._write_registration(page, fields, second_hand, rng, style, y)

    def _write_registration(
        self,
        page: Page,
        fields: FieldValues,
        second_hand: Hand,
        rng: random.Random,
        style: HandwritingStyle,
        y: int,
    ) -> None:
        """Write the office's registration marks in the clerk's own hand."""
        lines = [
            str(fields[name])
            for name in ("reg_number", "reg_date")
            if fields.get(name)
        ]
        if not lines:
            return

        width, height = self.page_size
        font_size = style.font_size
        bottom = max(
            int(height * rng.uniform(0.80, 0.93)), y + int(font_size * 1.6)
        )
        bottom = min(bottom, height - int(font_size * 3.0))
        page.put_lines(
            "registration",
            lines,
            second_hand,
            int(width * rng.uniform(0.68, 0.80)),
            bottom,
            color=rng.choice(INK_PALETTE),
        )

    def _stamp(
        self,
        page: Page,
        rows: list[str],
        fields: FieldValues,
        second_hand: Hand,
        rng: random.Random,
        style: HandwritingStyle,
        y: int,
    ) -> bool:
        """Press the office's incoming stamp and write the filing into it.

        The stamp's block carries everything read off it: its printed rows,
        with the number after the ``№`` and the date in its empty row. A
        stamp without an empty row leaves the date to be written below it.

        Args:
            page: The page being written.
            rows: The stamp's rows, as the record words them.
            fields: The record's field values, for the number and date.
            second_hand: The clerk's hand.
            rng: Random source for the placement.
            style: The page style, for its nominal size.
            y: Baseline the signature block ended on.

        Returns:
            Whether the stamp was pressed. It is not when the page has no
            room left for it below the signature.
        """
        number = str(fields.get("reg_number") or "")
        date = str(fields.get("reg_date") or "")
        font_size = style.font_size
        font_path = find_print_font(self.library)
        type_size = int(font_size * rng.uniform(*_STAMP_TYPE_SCALE))
        blank_width = int(second_hand.measure(number) + font_size)
        stamp_width, stamp_height = box_stamp_size(
            rows, type_size, blank_width, font_path
        )

        placed = self._place_stamp((stamp_width, stamp_height), style, rng, y)
        if placed is None:
            return False
        left, top = placed

        box, blanks = draw_box_stamp(
            page=page.image,
            top_left=(left, top),
            rows=rows,
            type_size=type_size,
            blank_width=blank_width,
            color=rng.choice(SEAL_COLORS),
            font_path=font_path,
            rng=rng,
        )
        ink = rng.choice(INK_PALETTE)
        entries = [value for value in (number, date) if value]
        # Writing into a stamp, a clerk keeps to its rows.
        entry_size = int(type_size * _ENTRY_SCALE)
        entry_hand = (
            second_hand
            if second_hand.size <= entry_size
            else second_hand.resized(entry_size)
        )
        boxes: list[BoundingBox | None] = [box]
        for block, blank, value in zip(_ENTRY_BLOCKS, blanks, entries):
            boxes.append(_write_in(page, block, blank, value, entry_hand, ink))
        page.add_block(
            "stamp",
            _read_stamp(rows, entries[: len(blanks)]),
            BoundingBox.union(boxes),
        )

        if date and len(blanks) < len(entries):
            page.put_lines(
                "registration",
                [date],
                second_hand,
                left + int(font_size * rng.uniform(0.2, 1.5)),
                top + stamp_height + int(second_hand.size * 1.2),
                color=ink,
            )
        return True

    def _place_stamp(
        self,
        size: tuple[int, int],
        style: HandwritingStyle,
        rng: random.Random,
        y: int,
    ) -> tuple[int, int] | None:
        """Choose where the incoming stamp is pressed.

        Below the signature when the letter left room there. Otherwise, or
        now and then by choice, in the empty corner left of the addressee
        block. A letter that fills its page to the foot gets the stamp over
        its last lines, as a busy clerk would press it.

        Args:
            size: The stamp's ``(width, height)``.
            style: The page style, for the header's edge and the hand size.
            rng: Random source for the placement.
            y: Baseline the signature block ended on.

        Returns:
            The stamp's top-left corner, or None when it fits nowhere.
        """
        width, height = self.page_size
        stamp_width, stamp_height = size
        font_size = style.font_size
        lowest = height - int(font_size * _STAMP_FOOT_MARGIN) - stamp_height

        left = int(width * rng.uniform(*_STAMP_LEFT))
        left = min(left, int(width * _STAMP_RIGHT_LIMIT) - stamp_width)
        top = max(int(height * rng.uniform(*_STAMP_TOP)), y + font_size // 2)
        below = (left, top) if left >= 0 and top <= lowest else None

        corner_left = int(width * rng.uniform(*_CORNER_LEFT))
        corner_top = int(height * rng.uniform(*_CORNER_TOP))
        fits_corner = (
            corner_left + stamp_width
            <= width * style.header_x - font_size * _CORNER_GAP
        )
        corner = (corner_left, corner_top) if fits_corner else None

        prefer_corner = rng.random() < _CORNER_SHARE
        for choice in (corner, below) if prefer_corner else (below, corner):
            if choice is not None:
                return choice
        if left < 0 or lowest < 0:
            return None
        return left, lowest


def _write_in(
    page: Page,
    block: str,
    blank: StampBlank,
    value: str,
    hand: Hand,
    ink: tuple[int, int, int],
) -> BoundingBox | None:
    """Write a value into the room a stamp left for it.

    The clerk's hand shrinks until the value fits; ink past the stamp's
    frame would be writing the stamp's reading does not account for.

    Returns:
        The box around what was written.
    """
    room = blank.right - blank.left
    while hand.measure(value) > room and hand.size > _MIN_ENTRY_SIZE:
        hand = hand.resized(int(hand.size * _ENTRY_SHRINK))
    page.put_lines(block, [value], hand, blank.left, blank.baseline, color=ink)
    return page.lines[-1].bbox


def _read_stamp(rows: Sequence[str], entries: Sequence[str]) -> str:
    """Spell a filled-in stamp out, its entries where they were written.

    Args:
        rows: The stamp's printed rows; an empty row is room for an entry,
            as is the space after a row ending in ``№``.
        entries: What was written into the room, in order.

    Returns:
        The stamp's rows, newline separated.
    """
    remaining = list(entries)
    lines = []
    for row in rows:
        if not row:
            if remaining:
                lines.append(remaining.pop(0))
        elif row.rstrip().endswith(NUMBER_SIGN) and remaining:
            lines.append(f"{row} {remaining.pop(0)}")
        else:
            lines.append(row)
    return "\n".join(lines)
