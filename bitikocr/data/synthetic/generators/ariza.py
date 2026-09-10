"""Layout for a handwritten Uzbek "Ariza" (application letter).

The arrangement a letter shares with every other letter — the addressee
block, the title, the body, the signature — lives in
:mod:`bitikocr.data.synthetic.generators.letter`. What is left here is what
belongs to an ariza alone: the marks the receiving office writes on it once
it has been filed.
"""

from __future__ import annotations

import random
from typing import ClassVar

from bitikocr.data.synthetic.generators.base import FieldValues
from bitikocr.data.synthetic.generators.letter import (
    DEFAULT_PAGE_SIZE,
    LetterGenerator,
)
from bitikocr.data.synthetic.hand import Hand
from bitikocr.data.synthetic.layout import Page
from bitikocr.data.synthetic.style import INK_PALETTE, HandwritingStyle

__all__ = ["DEFAULT_PAGE_SIZE", "DEFAULT_TITLE", "ArizaGenerator"]

DEFAULT_TITLE = "Ариза"


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
    )

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
        """Blocks in the order a human reads the finished letter."""
        return self.READING_ORDER

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
