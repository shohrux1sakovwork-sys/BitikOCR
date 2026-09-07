"""Layout for a handwritten Uzbek "Ariza" (application letter).

Only the *arrangement* lives here: where the addressee block, the title, the
body, the signature and the clerk's registration marks go. Rendering, fonts,
styles and ground-truth bookkeeping all come from the shared engine.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic.generators.base import (
    DEFAULT_INK_STRENGTH,
    DocumentGenerator,
    FieldValues,
    SyntheticDocument,
)
from bitikocr.data.synthetic.hand import Hand
from bitikocr.data.synthetic.layout import Page, wrap_text
from bitikocr.data.synthetic.style import (
    INK_PALETTE,
    PENCIL_COLOR,
    HandwritingStyle,
)

__all__ = ["ArizaGenerator"]

# A4 at 200 dpi.
DEFAULT_PAGE_SIZE = (1654, 2339)

DEFAULT_TITLE = "Ариза"

# The layout is re-measured at ever smaller sizes until it fits this share of
# the page, or until the handwriting would become unreadably small.
_TARGET_FILL = 0.86

# Nothing is written closer than this many nominal sizes to the foot of the
# page: a clerk runs out of paper before they run out of margin.
_FOOT_MARGIN = 2.0
_MIN_FONT_SIZE = 34
_SHRINK_FACTOR = 0.92


class ArizaGenerator(DocumentGenerator):
    """Render a one-page handwritten application letter.

    Args:
        config: Paths to the fonts and background templates to use.
        font_path: Force a specific handwriting font instead of sampling.
        page_size: Page size as ``(width, height)`` in pixels.
    """

    name: ClassVar[str] = "ariza"

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

    def __init__(
        self,
        config: SyntheticConfig,
        font_path: Path | str | None = None,
        page_size: tuple[int, int] = DEFAULT_PAGE_SIZE,
        ink_strength: float = DEFAULT_INK_STRENGTH,
    ) -> None:
        super().__init__(config, font_path, ink_strength)
        self.page_size = page_size

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names an ariza understands."""
        return self.FIELD_NAMES

    @property
    def reading_order(self) -> tuple[str, ...]:
        """Blocks in the order a human reads the finished letter."""
        return self.READING_ORDER

    def generate(
        self,
        fields: FieldValues,
        seed: int | None = None,
        style_overrides: Mapping[str, Any] | None = None,
    ) -> SyntheticDocument:
        """Render one ariza page. See :meth:`DocumentGenerator.generate`."""
        seed, rng = self._seeded_rng(seed)

        recipient = str(fields.get("recipient", ""))
        applicant = str(fields.get("applicant", ""))
        body = str(fields.get("body", ""))
        title = str(fields.get("title") or DEFAULT_TITLE)
        signature_name = fields.get("signature_name")
        date = fields.get("date")
        phone = fields.get("phone")
        reg_number = fields.get("reg_number")
        reg_date = fields.get("reg_date")
        page_number = fields.get("page_number")

        main_info, style = self._sample_style(
            rng, self._collect_text({**fields, "title": title}), style_overrides
        )
        second_text = " ".join(
            str(value) for value in (reg_number, reg_date, page_number) if value
        )
        second_info = self._second_hand_font(style, second_text)

        width, height = self.page_size
        style, body_hand, head_hand, title_hand = self._fit_to_page(
            style, main_info, rng, recipient, applicant, body
        )
        font_size = style.font_size

        second_style = style.replace(
            slant=rng.uniform(-0.05, 0.35),
            pen=rng.choice(["hard", "gel"]),
            char_rot_jitter=1.5,
            baseline_wobble=0.02,
        )
        second_hand = Hand(
            second_info,
            int(font_size * rng.uniform(0.75, 0.95)),
            second_style,
            rng,
        )

        page = Page(self.page_size, style, rng)

        if page_number:
            self._put_page_number(
                page, str(page_number), second_info, second_style, rng, style
            )

        y = self._put_header(
            page, recipient, applicant, head_hand, rng, style, width, height
        )
        y = self._put_title(page, title, title_hand, rng, style, width, y)
        y = self._put_body(page, body, body_hand, rng, style, width, y)
        y = self._put_signature(
            page,
            signature_name=signature_name,
            date=date,
            phone=phone,
            hand=body_hand,
            rng=rng,
            style=style,
            width=width,
            height=height,
            y=y,
        )

        if reg_number or reg_date:
            self._put_registration(
                page,
                lines=[str(v) for v in (reg_number, reg_date) if v],
                hand=second_hand,
                rng=rng,
                width=width,
                height=height,
                font_size=font_size,
                y=y,
            )

        annotation = page.annotation(
            self.reading_order,
            metadata={
                "document_type": self.name,
                "seed": seed,
                "font": main_info.path.name,
                "style": style.to_dict(),
                "fields": {
                    name: str(fields[name])
                    for name in self.field_names
                    if fields.get(name)
                },
            },
        )
        return SyntheticDocument(image=page.render(), annotation=annotation)

    # -- layout steps ------------------------------------------------------

    def _fit_to_page(
        self,
        style: HandwritingStyle,
        main_info: Any,
        rng: random.Random,
        recipient: str,
        applicant: str,
        body: str,
    ) -> tuple[HandwritingStyle, Hand, Hand, Hand]:
        """Shrink the nominal font size until the estimated layout fits.

        Args:
            style: The sampled style, whose ``font_size`` is the starting
                point.
            main_info: Font the body is written in.
            rng: Random source, passed on to every hand created here.
            recipient: Addressee block text.
            applicant: Applicant block text.
            body: Body text.

        Returns:
            ``(style, body_hand, header_hand, title_hand)`` with the style's
            ``font_size`` updated to the size that fits.
        """
        width, height = self.page_size
        font_size = style.font_size

        while True:
            body_hand = Hand(main_info, font_size, style, rng)
            head_hand = Hand(
                main_info, int(font_size * style.header_scale), style, rng
            )
            title_hand = Hand(
                main_info, int(font_size * style.title_scale), style, rng
            )

            header_lines = len(
                wrap_text(
                    f"{recipient}\n{applicant}",
                    head_hand,
                    int(width * 0.955 - width * style.header_x),
                )
            )
            body_lines = len(
                wrap_text(
                    body,
                    body_hand,
                    int(width * (style.right_margin - style.left_margin)),
                )
            )
            estimated = (
                height * style.top_margin
                + (header_lines + body_lines + 1)
                * font_size
                * style.line_spacing
                + font_size * 6.5
            )
            if (
                estimated <= height * _TARGET_FILL
                or font_size <= _MIN_FONT_SIZE
            ):
                break
            font_size = int(font_size * _SHRINK_FACTOR)

        return (
            style.replace(font_size=font_size),
            body_hand,
            head_hand,
            title_hand,
        )

    def _put_page_number(
        self,
        page: Page,
        page_number: str,
        second_info: Any,
        second_style: HandwritingStyle,
        rng: random.Random,
        style: HandwritingStyle,
    ) -> None:
        """Write the archivist's pencil page number in the top-right corner."""
        width, height = self.page_size
        pencil = Hand(
            second_info,
            int(style.font_size * 0.8),
            second_style.replace(pen="hard", ink_variation=0.3),
            rng,
        )
        page.put_lines(
            "page_number",
            [page_number],
            pencil,
            int(width * rng.uniform(0.86, 0.93)),
            int(height * 0.015) + pencil.baseline,
            color=PENCIL_COLOR,
        )

    def _put_header(
        self,
        page: Page,
        recipient: str,
        applicant: str,
        hand: Hand,
        rng: random.Random,
        style: HandwritingStyle,
        width: int,
        height: int,
    ) -> int:
        """Write the addressee and applicant block in the top-right area."""
        left = int(width * style.header_x)
        max_width = int(width * 0.955) - left

        y = int(height * style.top_margin) + hand.baseline
        y = page.put_lines(
            "recipient", wrap_text(recipient, hand, max_width), hand, left, y
        )
        y += int(hand.size * rng.uniform(0.0, 0.5))
        return page.put_lines(
            "applicant", wrap_text(applicant, hand, max_width), hand, left, y
        )

    def _put_title(
        self,
        page: Page,
        title: str,
        hand: Hand,
        rng: random.Random,
        style: HandwritingStyle,
        width: int,
        y: int,
    ) -> int:
        """Write the centred title."""
        y += int(style.font_size * rng.uniform(0.8, 2.6))
        left = int(width * style.title_x - hand.measure(title) / 2)
        return page.put_lines("title", [title], hand, left, y)

    def _put_body(
        self,
        page: Page,
        body: str,
        hand: Hand,
        rng: random.Random,
        style: HandwritingStyle,
        width: int,
        y: int,
    ) -> int:
        """Write the wrapped body paragraph."""
        y += int(style.font_size * rng.uniform(0.3, 1.2))
        left = int(width * style.left_margin)
        right = int(width * style.right_margin)
        indent = int(style.font_size * style.indent)
        return page.put_lines(
            "body",
            wrap_text(body, hand, right - left, first_indent=indent),
            hand,
            left,
            y,
            indent_first=indent,
        )

    def _put_signature(
        self,
        page: Page,
        signature_name: Any,
        date: Any,
        phone: Any,
        hand: Hand,
        rng: random.Random,
        style: HandwritingStyle,
        width: int,
        height: int,
        y: int,
    ) -> int:
        """Write the signature scribble, the signer's name, the date and phone.

        Everything here follows the body, so on a full page it can run out of
        room. Each baseline is therefore held above the foot of the page:
        ink drawn past the edge is clipped, which would leave a
        transcription the image does not show.
        """
        font_size = style.font_size
        last_baseline = height - int(font_size * _FOOT_MARGIN)
        y = min(y + int(font_size * rng.uniform(0.6, 2.0)), last_baseline)

        # A phone number always takes the left side, pushing the signature right.
        signs_right = True if phone else rng.random() < 0.55

        if style.signature_scribble:
            spread = (
                rng.uniform(0.58, 0.72)
                if signs_right
                else rng.uniform(0.35, 0.5)
            )
            page.put_scribble(
                int(width * spread), y - int(font_size * 0.9), font_size
            )

        if signature_name:
            spread = (
                rng.uniform(0.12, 0.3)
                if signs_right
                else rng.uniform(0.62, 0.72)
            )
            page.put_lines(
                "signature_name",
                [str(signature_name)],
                hand,
                int(width * spread),
                min(y + int(font_size * rng.uniform(0, 0.5)), last_baseline),
            )

        if date:
            y = page.put_lines(
                "date",
                [str(date)],
                hand,
                int(width * rng.uniform(0.55, 0.78)),
                min(y + int(font_size * rng.uniform(1.7, 2.3)), last_baseline),
            )

        if phone:
            phone_y = min(
                (
                    y + int(font_size * 1.5)
                    if signature_name
                    else y + int(font_size * rng.uniform(0, 0.5))
                ),
                last_baseline,
            )
            y = page.put_lines(
                "phone",
                [str(phone)],
                hand,
                int(width * rng.uniform(0.12, 0.25)),
                phone_y,
            )
        return y

    def _put_registration(
        self,
        page: Page,
        lines: list[str],
        hand: Hand,
        rng: random.Random,
        width: int,
        height: int,
        font_size: int,
        y: int,
    ) -> None:
        """Write the office's registration marks in a second hand."""
        bottom = max(
            int(height * rng.uniform(0.80, 0.93)), y + int(font_size * 1.6)
        )
        bottom = min(bottom, height - int(font_size * 3.0))
        page.put_lines(
            "registration",
            lines,
            hand,
            int(width * rng.uniform(0.68, 0.80)),
            bottom,
            color=rng.choice(INK_PALETTE),
        )
