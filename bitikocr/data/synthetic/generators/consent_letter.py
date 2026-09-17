"""Layout for a handwritten Uzbek "Rozilik xati" (consent letter).

A resident writes to the city mayor to say they have no objection to
something a neighbour is doing — a shared courtyard boundary, a
privatisation, a claim over a house. The letter is shaped like an ariza and
shares its engine, but it is not filed with an office: it is *certified*.
Below the author's signature the mahalla chairman writes that he attests to
it, signs, and presses the neighbourhood seal over the whole block.

That foot is the whole of what lives here. Everything above it — the
addressee block, the title, the body, the author's signature — comes from
:mod:`bitikocr.data.synthetic.generators.letter`.
"""

from __future__ import annotations

import random
from typing import ClassVar

from bitikocr.data.synthetic.effects import (
    SEAL_COLORS,
    STAMP_REACH,
    draw_round_stamp,
)
from bitikocr.data.synthetic.generators.base import FieldValues
from bitikocr.data.synthetic.generators.letter import (
    DEFAULT_PAGE_SIZE,
    LetterGenerator,
)
from bitikocr.data.synthetic.hand import Hand
from bitikocr.data.synthetic.layout import Page
from bitikocr.data.synthetic.style import HandwritingStyle
from bitikocr.data.synthetic.system_fonts import find_print_font

__all__ = [
    "CERTIFICATION_SIGNATURE_BLOCK",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_TITLE",
    "ConsentLetterGenerator",
]

DEFAULT_TITLE = "Розилик хати"

#: Block name for the certifying official's own scribble. It is kept apart
#: from the author's so that one page never carries two blocks of the same
#: name, which would make the ground truth ambiguous about whose hand signed
#: where.
CERTIFICATION_SIGNATURE_BLOCK = "certification_signature"

#: Fields carrying the seal's own lettering rather than the letter's content.
SEAL_RING_FIELD = "stamp_ring"
SEAL_CENTRE_FIELD = "stamp_center"

# The certification runs across the page under the signature: the attesting
# phrase on the left, the official's scribble in the middle and their name on
# the right, each placed as a share of the page width.
_NOTE_X = (0.13, 0.24)
_CERTIFIER_SCRIBBLE_X = (0.38, 0.50)
_CERTIFIER_NAME_X = (0.58, 0.70)
_ROLE_X = (0.16, 0.26)

# Gaps between the author's signature, the certification and its second line,
# in nominal handwriting sizes.
_CERTIFICATION_GAP = (1.6, 3.2)
_ROLE_LINE_GAP = (1.0, 1.5)

# The seal is pressed over the certification by hand, so it lands low and
# left of the middle of that block, and its size varies between offices.
_SEAL_RADIUS = (2.4, 3.2)
_SEAL_CENTRE_X = (0.24, 0.44)
_SEAL_CENTRE_Y = (0.1, 0.9)

# Everything at the foot stays this many nominal sizes clear of the paper's
# edge: ink drawn past it is clipped, leaving a box that claims more than the
# page shows.
_EDGE_MARGIN = 1.2


class ConsentLetterGenerator(LetterGenerator):
    """Render a one-page handwritten consent letter.

    Args:
        config: Paths to the fonts and background templates to use.
        font_path: Force a specific handwriting font instead of sampling.
        page_size: Page size as ``(width, height)`` in pixels.
        ink_strength: How heavily the pen writes.
    """

    name: ClassVar[str] = "consent_letter"
    default_title: ClassVar[str] = DEFAULT_TITLE

    # The certification block and its seal sit below the signature, so the
    # body is fitted into a shorter page than an ariza's.
    foot_allowance: ClassVar[float] = 9.5

    # The spec puts the date at the foot on the left, with the signature
    # opposite it on the right.
    date_x: ClassVar[tuple[float, float]] = (0.12, 0.24)

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "recipient",
        "applicant",
        "passport",
        "phone",
        "body",
        "title",
        "signature_name",
        "date",
        "certifier_note",
        "certifier_role",
        "certifier_name",
        "page_number",
        SEAL_RING_FIELD,
        SEAL_CENTRE_FIELD,
    )

    READING_ORDER: ClassVar[tuple[str, ...]] = (
        "page_number",
        "recipient",
        "applicant",
        "passport",
        "phone",
        "title",
        "body",
        "signature_name",
        "date",
        "certifier_note",
        "certifier_role",
        "certifier_name",
    )

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names a consent letter understands."""
        return self.FIELD_NAMES

    @property
    def reading_order(self) -> tuple[str, ...]:
        """Blocks in the order a human reads the finished letter."""
        return self.READING_ORDER

    def _header_extra(self, fields: FieldValues) -> list[tuple[str, str]]:
        """Identify a private citizen by passport and telephone.

        A letter from a citizen has to say who they are well enough to be
        acted on, so the sender's block carries their passport and a number
        to reach them on. An organisation identifies itself by its name and
        carries neither.

        Args:
            fields: The record's field values.

        Returns:
            The passport and telephone lines that are present.
        """
        return [
            (name, str(fields[name]))
            for name in ("passport", "phone")
            if fields.get(name)
        ]

    def _second_hand_text(self, fields: FieldValues) -> str:
        """Return what the certifying official writes, not the author."""
        return " ".join(
            str(fields[name])
            for name in ("certifier_note", "certifier_role", "certifier_name")
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
        """Attest to the letter, sign the attestation and press the seal.

        The two are independent, because who signs and who seals depends on
        who wrote the letter. A citizen's letter is attested to by an
        official, and the seal that follows is that official's. An
        organisation attests to nothing — it signs its own letter and seals
        it, so the page gets a seal and no attestation at all.

        The official's name may share the attesting line or take one of its
        own. Which it is follows the record: a letter that names the office
        the certifier holds has too much to fit on one line, and the scans
        put that form on two.

        Args:
            page: The page being written.
            fields: The record's field values.
            second_hand: The certifying official's hand.
            rng: Random source for the placement.
            style: The page style, for its nominal size.
            y: Baseline the author's signature block ended on.
        """
        note = str(fields.get("certifier_note") or "")
        role = str(fields.get("certifier_role") or "")
        name = str(fields.get("certifier_name") or "")

        width, height = self.page_size
        font_size = style.font_size
        last_baseline = height - int(font_size * _EDGE_MARGIN)

        y = min(
            y + int(font_size * rng.uniform(*_CERTIFICATION_GAP)),
            last_baseline,
        )
        if not (note or name):
            # Nobody attested to this one. Whatever seal it carries is the
            # author's own, and it goes where the attestation would have.
            self._stamp(page, fields, rng, font_size, y)
            return

        if note:
            page.put_lines(
                "certifier_note",
                [note],
                second_hand,
                int(width * rng.uniform(*_NOTE_X)),
                y,
            )

        # With an office named, the name drops to a second line under the
        # attesting phrase; without one it sits beside it.
        name_y = y
        if role:
            name_y = min(
                y + int(font_size * rng.uniform(*_ROLE_LINE_GAP)),
                last_baseline,
            )
            page.put_lines(
                "certifier_role",
                [role],
                second_hand,
                int(width * rng.uniform(*_ROLE_X)),
                name_y,
            )

        if name:
            page.put_lines(
                "certifier_name",
                [name],
                second_hand,
                int(width * rng.uniform(*_CERTIFIER_NAME_X)),
                name_y,
            )

        page.put_scribble(
            int(width * rng.uniform(*_CERTIFIER_SCRIBBLE_X)),
            name_y - int(font_size * 0.9),
            font_size,
            block=CERTIFICATION_SIGNATURE_BLOCK,
        )
        self._stamp(page, fields, rng, font_size, name_y)

    def _stamp(
        self,
        page: Page,
        fields: FieldValues,
        rng: random.Random,
        font_size: int,
        y: int,
    ) -> None:
        """Press the neighbourhood seal over the certification block.

        Args:
            page: The page being written.
            fields: The record's field values, read for the seal's lettering.
            rng: Random source for the seal's size, place and ink.
            font_size: Nominal handwriting size the seal is scaled against.
            y: Baseline the certification was written on.
        """
        ring = str(fields.get(SEAL_RING_FIELD) or "")
        centre_lines = list(fields.get(SEAL_CENTRE_FIELD) or ())
        if not ring:
            return

        width, height = self.page_size
        radius = int(font_size * rng.uniform(*_SEAL_RADIUS))
        centre_x = int(width * rng.uniform(*_SEAL_CENTRE_X))
        centre_y = y + int(radius * rng.uniform(*_SEAL_CENTRE_Y))

        # A seal pressed off the paper would be clipped, and its measured
        # box would then claim ink the page never carried. The ring's curved
        # lettering reaches past the ring, so the whole footprint is what has
        # to fit, not the radius.
        reach = int(radius * STAMP_REACH)
        centre_x = max(reach, min(centre_x, width - reach))
        centre_y = max(reach, min(centre_y, height - reach))

        box = draw_round_stamp(
            page=page.image,
            centre=(centre_x, centre_y),
            radius=radius,
            ring_text=ring,
            centre_lines=centre_lines,
            color=rng.choice(SEAL_COLORS),
            font_path=find_print_font(self.library),
            rng=rng,
        )
        page.add_block("stamp", f"{ring} / {' '.join(centre_lines)}", box)
