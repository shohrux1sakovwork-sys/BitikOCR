"""Layout for a handwritten Uzbek "Tushuntirish xati" (explanatory letter).

Someone asked to account for something writes one: a citizen telling the
district mayor what they will do with land they are allotted, or an employee
or a pupil explaining a lateness, an absence or a task left undone. Whoever
is answering, the page is a letter — addressee block top-right, title, body,
signature — so it shares
:class:`~bitikocr.data.synthetic.generators.letter.LetterGenerator` with the
ariza and the consent letter.

It has no foot of its own. Nobody registers or seals an explanation: it is
the writer's own account, signed and usually dated, and nothing more. What
sets it apart on the page is the archive's handling — the scanned ones are
numbered at the foot, not the head — and that a hand sometimes runs the
title on as the last words of the sender's block instead of giving it a
line of its own.
"""

from __future__ import annotations

from typing import ClassVar

from bitikocr.data.synthetic.generators.base import FieldValues
from bitikocr.data.synthetic.generators.letter import (
    DEFAULT_PAGE_SIZE,
    LetterGenerator,
)

__all__ = ["DEFAULT_PAGE_SIZE", "DEFAULT_TITLE", "ExplanatoryLetterGenerator"]

DEFAULT_TITLE = "Тушунтириш хати"


class ExplanatoryLetterGenerator(LetterGenerator):
    """Render a one-page handwritten explanatory letter.

    Args:
        config: Paths to the fonts and background templates to use.
        font_path: Force a specific handwriting font instead of sampling.
        page_size: Page size as ``(width, height)`` in pixels.
        ink_strength: How heavily the pen writes.
    """

    name: ClassVar[str] = "explanatory_letter"
    default_title: ClassVar[str] = DEFAULT_TITLE

    # Dated at the foot on the left, with the signature opposite it.
    date_x: ClassVar[tuple[float, float]] = (0.12, 0.26)

    # Every scanned one is numbered in the bottom-right corner.
    page_number_at_foot: ClassVar[bool] = True

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "recipient",
        "applicant",
        "title",
        "body",
        "signature_name",
        "date",
        "page_number",
    )

    # The page number is read last, because it is written last on the page.
    READING_ORDER: ClassVar[tuple[str, ...]] = (
        "recipient",
        "applicant",
        "title",
        "body",
        "signature_name",
        "date",
        "page_number",
    )

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names an explanatory letter understands."""
        return self.FIELD_NAMES

    @property
    def reading_order(self) -> tuple[str, ...]:
        """Blocks in the order a human reads the finished letter."""
        return self.READING_ORDER

    def _second_hand_text(self, fields: FieldValues) -> str:
        """Return the archivist's number, the only thing another hand adds."""
        return str(fields.get("page_number") or "")
