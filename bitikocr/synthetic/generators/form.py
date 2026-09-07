"""Fill a pre-printed Uzbek civil-registry form with handwriting.

This module knows *how* a clerk fills a form: which hand writes the digits,
how a value is shrunk to fit its line, where the seal is pressed and how the
registrar signs. It does not know *where* any of it goes — that comes from a
measured :class:`~bitikocr.synthetic.templates.FormTemplate`.

Every certificate the registry issues is therefore the same generator with a
different template. A new form, or a second variant of an existing one — a
single-language certificate beside the bilingual one — is a background scan
and a layout JSON, not a subclass.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image, ImageDraw

from bitikocr.config import SyntheticConfig
from bitikocr.models.geometry import BoundingBox
from bitikocr.synthetic.effects import draw_round_stamp
from bitikocr.synthetic.generators.base import (
    DEFAULT_INK_STRENGTH,
    DocumentGenerator,
    FieldValues,
    SyntheticDocument,
)
from bitikocr.synthetic.hand import Hand
from bitikocr.synthetic.layout import Page, wrap_text
from bitikocr.synthetic.style import Color, HandwritingStyle
from bitikocr.synthetic.system_fonts import find_monospace_font, find_print_font
from bitikocr.synthetic.templates import FieldGeometry, FormTemplate, MarkArea

__all__ = [
    "REGISTRAR_NAME_FIELD",
    "FormGenerator",
    "FormOptions",
]

logger = logging.getLogger(__name__)

#: Field holding the registrar's surname. Some forms give it a line of its
#: own, in which case it is an ordinary template field; on the rest it shares
#: the signature area and is written there instead.
REGISTRAR_NAME_FIELD = "registrar_name"

#: Fields carrying the seal's own lettering rather than document content.
SEAL_RING_FIELD = "stamp_ring"
SEAL_CENTRE_FIELD = "stamp_center"

DEFAULT_SEAL_RING = (
    "O'ZBEKISTON RESPUBLIKASI * FUQAROLIK HOLATI DALOLATNOMALARINI "
    "YOZISH BO'LIMI *"
)
DEFAULT_SEAL_CENTRE = ("FHDYO",)

_SEAL_COLORS: tuple[Color, ...] = (
    (70, 40, 150),
    (40, 60, 170),
    (60, 30, 130),
)

# The seal is pressed by hand onto a page that is already written on. It
# lands off-centre — usually a little left of the printed circle, and high or
# low enough to catch the lines above or below it — and its size varies a
# little between offices. Offsets are fractions of the seal's printed area,
# so they scale with whatever form is being filled.
_SEAL_RADIUS_JITTER = (0.92, 1.06)
_SEAL_LEFT_BIAS = 0.08
_SEAL_JITTER_X = 0.10
_SEAL_JITTER_Y = 0.15

# Machine-printed text sits just inside the left edge of its area, following
# whatever label the form prints there.
_PRINTED_INSET = 0.04
_PRINTED_HEIGHT_RATIO = 0.8
_PRINTED_COLOR: Color = (25, 22, 28)

# A hand shrinking below this share of its normal size is worth a debug note.
_MIN_FIT_RATIO = 0.6
_FIT_STEP = 0.92

# Nominal handwriting size below which shrinking stops regardless of fit.
_MIN_HAND_SIZE = 10

# Enough shrink passes to take the largest hand down to _MIN_HAND_SIZE.
_MAX_FIT_PASSES = 40

# Share of the vertical gap between two printed rules that the handwriting
# may occupy. Below 1.0 the lines stay clear of each other.
_MULTILINE_GAP_RATIO = 0.95

# How far off centre a value may sit in its cell, as a share of the slack
# around it. 0.0 would centre every value exactly; 0.5 would let one sit
# flush against either end.
_CENTRE_JITTER = 0.15


@dataclass(frozen=True)
class FormOptions:
    """Rendering options that are not document content.

    Args:
        template: Name of the layout to fill; see ``bitikocr synth
            list-templates``. None uses the generator's default variant.
        scale: Multiplies the background resolution. 2.0 turns a 1419x1108
            scan into 2838x2216.
        draw_seal: Whether the office seal is stamped on the page.
    """

    template: str | None = None
    scale: float = 2.0
    draw_seal: bool = True


class FormGenerator(DocumentGenerator):
    """Fill a pre-printed civil-registry form with a clerk's handwriting.

    Subclasses supply only an identity: the document type ``name`` and the
    layout variant to fill by default. Everything else comes from the
    template.

    Args:
        config: Paths to the fonts, backgrounds and layouts to use.
        font_path: Force a specific handwriting font instead of sampling.
        options: Rendering options; see :class:`FormOptions`.

    Raises:
        FileNotFoundError: If the requested layout or its background is
            missing.
        ValueError: If the layout is malformed.
    """

    #: Layout filled when the caller names no variant.
    default_template: ClassVar[str]

    def __init__(
        self,
        config: SyntheticConfig,
        font_path: Path | str | None = None,
        options: FormOptions | None = None,
        ink_strength: float = DEFAULT_INK_STRENGTH,
    ) -> None:
        super().__init__(config, font_path, ink_strength)
        self.options = options or FormOptions()
        name = self.options.template or self.default_template
        self.template = FormTemplate.load(config.layout(name))

    @classmethod
    def with_template(
        cls,
        config: SyntheticConfig,
        font_path: Path | str | None,
        template: str,
        ink_strength: float = DEFAULT_INK_STRENGTH,
    ) -> FormGenerator:
        """Build a generator for one of this document's form variants.

        Args:
            config: Paths to the fonts, backgrounds and layouts to use.
            font_path: Force a specific handwriting font.
            template: Name of the layout to fill.
            ink_strength: How heavily the pen writes.

        Returns:
            A generator bound to that template.
        """
        return cls(
            config, font_path, FormOptions(template=template), ink_strength
        )

    @property
    def registrar_writes_on_signature(self) -> bool:
        """Whether the registrar's name shares the signature area.

        A signature area with a printed rule through it is a name line that
        also gets signed, so the name is written there. An area with no rule
        is a free signature zone, and the form gives the name a field of its
        own elsewhere.
        """
        signature = self.template.signature
        return signature is not None and signature.baseline_y is not None

    @property
    def field_names(self) -> tuple[str, ...]:
        """Every field this template accepts, plus its non-underlined marks."""
        extra: tuple[str, ...] = (SEAL_RING_FIELD, SEAL_CENTRE_FIELD)
        if self.registrar_writes_on_signature:
            extra = (REGISTRAR_NAME_FIELD, *extra)
        return (
            *self.template.field_names,
            *self.template.printed_names,
            *extra,
        )

    @property
    def reading_order(self) -> tuple[str, ...]:
        """Blocks in the order a human reads the finished form."""
        registrar = (
            (REGISTRAR_NAME_FIELD,)
            if self.registrar_writes_on_signature
            else ()
        )
        return (
            *self.template.field_names,
            *registrar,
            *self.template.printed_names,
        )

    def generate(
        self,
        fields: FieldValues,
        seed: int | None = None,
        style_overrides: Mapping[str, Any] | None = None,
    ) -> SyntheticDocument:
        """Render one filled certificate.

        See :meth:`DocumentGenerator.generate`.
        """
        seed, rng = self._seeded_rng(seed)
        template, options = self.template, self.options

        values = {
            name: str(fields[name])
            for name in template.field_names
            if fields.get(name)
        }
        registrar_name = (
            str(fields.get(REGISTRAR_NAME_FIELD) or "")
            if self.registrar_writes_on_signature
            else ""
        )

        background_path = self.config.background(template.background)
        background = Image.open(background_path)
        native_width, native_height = template.native_size
        scale_x = background.width / native_width * options.scale
        scale_y = background.height / native_height * options.scale
        size = (
            int(background.width * options.scale),
            int(background.height * options.scale),
        )

        main_info, style = self._sample_style(
            rng, " ".join(values.values()), style_overrides
        )
        style = self._as_form_style(style, rng, options.scale, style_overrides)
        second_info = self._second_hand_font(style, registrar_name)

        page = Page(size, style, rng, background=background)
        font_size = style.font_size

        hand = Hand(main_info, font_size, style, rng)
        numeric_hand = Hand(
            main_info, int(font_size * rng.uniform(0.95, 1.15)), style, rng
        )
        registrar_style = style.replace(slant=rng.uniform(0.0, 0.5), pen="hard")
        registrar_hand = Hand(
            second_info,
            int(font_size * rng.uniform(0.9, 1.2)),
            registrar_style,
            rng,
        )

        writer = _FieldWriter(
            page=page,
            scale_x=scale_x,
            scale_y=scale_y,
            font_size=font_size,
            rng=rng,
        )
        for name, text in values.items():
            geometry = template.field(name)
            writer.write(
                geometry, text, numeric_hand if geometry.is_numeric else hand
            )

        if template.signature is not None:
            self._sign(
                page=page,
                area=template.signature,
                registrar_name=registrar_name,
                hand=registrar_hand,
                writer=writer,
                style=registrar_style,
                rng=rng,
            )
        if options.draw_seal and template.seal is not None:
            self._stamp(page, template.seal, fields, scale_x, scale_y, rng)

        printed = {}
        for area in template.printed:
            value = fields.get(area.name)
            if not value:
                continue
            self._print_text(page, area, str(value), scale_x, scale_y)
            printed[area.name] = str(value)

        self._check_keep_out(page, template, scale_x, scale_y)

        recorded = dict(values)
        if registrar_name:
            recorded[REGISTRAR_NAME_FIELD] = registrar_name
        recorded.update(printed)

        annotation = page.annotation(
            self.reading_order,
            metadata={
                "document_type": self.name,
                "template": template.name,
                "seed": seed,
                "font": main_info.path.name,
                "style": style.to_dict(),
                "fields": recorded,
                "background": background_path.name,
                "scale": options.scale,
            },
        )
        return SyntheticDocument(image=page.render(), annotation=annotation)

    # -- style -------------------------------------------------------------

    @staticmethod
    def _as_form_style(
        style: HandwritingStyle,
        rng: random.Random,
        scale: float,
        style_overrides: Mapping[str, Any] | None,
    ) -> HandwritingStyle:
        """Tighten a page style into the cramped hand used on printed forms.

        Args:
            style: The freely sampled page style.
            rng: Random source for the form-specific ranges.
            scale: Background resolution multiplier the sizes scale with.
            style_overrides: Caller pins, re-applied so they still win.

        Returns:
            A style with smaller, steadier, mostly-ballpoint handwriting.
        """
        style = style.replace(
            font_size=int(rng.randint(26, 36) * scale),
            pen=rng.choices(["hard", "gel"], weights=[0.7, 0.3])[0],
            stroke_px=rng.uniform(2.2, 3.4) * scale / 2.0,
            ink_variation=rng.uniform(0.03, 0.18),
            slant=rng.uniform(-0.05, 0.4),
            line_slope=rng.uniform(-0.8, 0.8),
            baseline_wobble=rng.uniform(0.0, 0.03),
        )
        return style.replace(**style_overrides) if style_overrides else style

    @staticmethod
    def _check_keep_out(
        page: Page,
        template: FormTemplate,
        scale_x: float,
        scale_y: float,
    ) -> None:
        """Warn when drawn ink lands on a region the blank form reserves.

        A form may already carry a printed QR code or photo window. Writing
        over one both spoils the document and puts ink inside a box no
        annotation claims, so it is worth surfacing rather than shipping.

        Args:
            page: The finished page.
            template: The form being filled.
            scale_x: Native-to-page horizontal scale factor.
            scale_y: Native-to-page vertical scale factor.
        """
        for area in template.keep_out:
            reserved = BoundingBox(
                left=int(area.bbox.left * scale_x),
                top=int(area.bbox.top * scale_y),
                right=int(area.bbox.right * scale_x),
                bottom=int(area.bbox.bottom * scale_y),
            )
            for block in page.blocks:
                if block.bbox is not None and _overlaps(block.bbox, reserved):
                    logger.warning(
                        "%s was drawn over the reserved %s region",
                        block.kind,
                        area.name,
                    )

    # -- marks -------------------------------------------------------------

    @staticmethod
    def _sign(
        page: Page,
        area: MarkArea,
        registrar_name: str,
        hand: Hand,
        writer: _FieldWriter,
        style: HandwritingStyle,
        rng: random.Random,
    ) -> None:
        """Sign the form, following the shape of its signature area.

        A narrow area with a printed rule is a name line: the surname goes
        above it and the signature runs across it. A tall open zone is signed
        freely, roughly in the middle.

        Args:
            page: The page being filled.
            area: The signature region measured on the form.
            registrar_name: The registrar's surname; may be empty.
            hand: The registrar's hand.
            writer: Writer used to place the surname.
            style: The registrar's style, for the signature ink.
            rng: Random source for the scribble's placement.
        """
        if registrar_name:
            writer.write_on(
                block=REGISTRAR_NAME_FIELD,
                text=registrar_name,
                hand=hand,
                baseline_y=area.bbox.top,
                x_start=area.bbox.left,
                x_end=area.bbox.right,
            )

        size = int(writer.font_size * 0.9)
        if area.baseline_y is not None:
            # Start inside the top edge and run down across the printed rule.
            top = area.bbox.top + rng.uniform(-2, 10)
        else:
            # A free zone: sign near the middle, leaving room for the flourish.
            top = area.bbox.top + area.bbox.height * rng.uniform(0.25, 0.45)

        # A signature zone can sit right beside a printed QR code, so the
        # scribble starts a little way in and is clamped to what is left.
        left = area.bbox.left + rng.uniform(0, min(40, area.bbox.width * 0.2))
        available = writer.to_page_x(area.bbox.right) - writer.to_page_x(left)

        page.put_scribble(
            writer.to_page_x(left),
            writer.to_page_y(top),
            size,
            color=style.ink,
            max_width=available,
        )

    def _stamp(
        self,
        page: Page,
        area: MarkArea,
        fields: FieldValues,
        scale_x: float,
        scale_y: float,
        rng: random.Random,
    ) -> None:
        """Press the round office seal onto its printed area.

        Args:
            page: The page being filled.
            area: The seal region measured on the form.
            fields: Field values, read for the seal's own lettering.
            scale_x: Native-to-page horizontal scale factor.
            scale_y: Native-to-page vertical scale factor.
            rng: Random source for the press's offset, size and colour.
        """
        ring = str(fields.get(SEAL_RING_FIELD) or DEFAULT_SEAL_RING)
        centre_lines = list(
            fields.get(SEAL_CENTRE_FIELD) or DEFAULT_SEAL_CENTRE
        )

        native_x, native_y = area.centre
        offset_x = area.bbox.width * (
            -_SEAL_LEFT_BIAS + rng.uniform(-_SEAL_JITTER_X, _SEAL_JITTER_X)
        )
        offset_y = area.bbox.height * rng.uniform(
            -_SEAL_JITTER_Y, _SEAL_JITTER_Y
        )
        centre = (
            int((native_x + offset_x) * scale_x),
            int((native_y + offset_y) * scale_y),
        )
        radius = int(area.radius * rng.uniform(*_SEAL_RADIUS_JITTER) * scale_x)

        box = draw_round_stamp(
            page=page.image,
            centre=centre,
            radius=radius,
            ring_text=ring,
            centre_lines=centre_lines,
            color=rng.choice(_SEAL_COLORS),
            font_path=find_print_font(self.library),
            rng=rng,
        )
        page.add_block("stamp", f"{ring} / {' '.join(centre_lines)}", box)

    @staticmethod
    def _print_text(
        page: Page,
        area: MarkArea,
        text: str,
        scale_x: float,
        scale_y: float,
    ) -> None:
        """Print machine-set text — a serial number, a form series — in place.

        These are typeset rather than handwritten, and the blank form leaves
        their area empty, so nothing has to be covered first.

        Args:
            page: The page being filled.
            area: The printed region measured on the form.
            text: The characters to print.
            scale_x: Native-to-page horizontal scale factor.
            scale_y: Native-to-page vertical scale factor.
        """
        left = int(area.bbox.left * scale_x)
        top = int(area.bbox.top * scale_y)
        right = int(area.bbox.right * scale_x)
        bottom = int(area.bbox.bottom * scale_y)
        height = bottom - top

        size = max(8, int(height * _PRINTED_HEIGHT_RATIO))
        font = find_monospace_font(size)
        width = font.getlength(text)

        # The form prints its label ("I-HR №", "№") right at the area's left
        # edge, so the value follows it instead of sitting in the middle.
        text_left = left + int((right - left) * _PRINTED_INSET)
        text_top = top + int((height - size) / 2)
        ImageDraw.Draw(page.image).text(
            (text_left, text_top),
            text,
            font=font,
            fill=_PRINTED_COLOR + (255,),
        )

        box = BoundingBox(
            left=text_left,
            top=text_top,
            right=int(text_left + width),
            bottom=text_top + size,
        )
        page.add_block(area.name, text, box)
        page.add_line(area.name, text, box)


@dataclass
class _FieldWriter:
    """Write field values onto the printed underlines of a form.

    Args:
        page: The page being filled.
        scale_x: Native-to-page horizontal scale factor.
        scale_y: Native-to-page vertical scale factor.
        font_size: The clerk's nominal handwriting size.
        rng: Random source for placement inside a segment.
    """

    page: Page
    scale_x: float
    scale_y: float
    font_size: int
    rng: random.Random

    def write(self, geometry: FieldGeometry, text: str, hand: Hand) -> None:
        """Write one field, wrapping and shrinking it to fit its segments.

        Args:
            geometry: Where the field goes and whether it is numeric.
            text: The value to write.
            hand: The hand to write it in.
        """
        widths = [
            self.to_page_x(segment.x_end) - self.to_page_x(segment.x_start)
            for segment in geometry.segments
        ]
        hand, lines = self._fit_lines(text, hand, geometry.segments, widths)

        boxes: list[BoundingBox | None] = []
        for line, segment in zip(lines, geometry.segments):
            self._put(
                block=geometry.name,
                line=line,
                hand=hand,
                baseline_y=segment.baseline_y,
                x_start=segment.x_start,
                x_end=segment.x_end,
            )
            boxes.append(self.page.lines[-1].bbox)

        self.page.add_block(
            geometry.name, "\n".join(lines), BoundingBox.union(boxes)
        )

    def write_on(
        self,
        block: str,
        text: str,
        hand: Hand,
        baseline_y: int,
        x_start: int,
        x_end: int,
    ) -> None:
        """Write a single-line value on an explicit rule.

        Used for values that are not template fields, such as the registrar's
        surname in the signature area.

        Args:
            block: Block name to record.
            text: The value to write.
            hand: The hand to write it in.
            baseline_y: The rule to sit on, in template pixels.
            x_start: Left end of the writable span, in template pixels.
            x_end: Right end of the writable span, in template pixels.
        """
        width = self.to_page_x(x_end) - self.to_page_x(x_start)
        hand = self._shrink(text, hand, width)
        self._put(block, text, hand, baseline_y, x_start, x_end)
        self.page.add_block(block, text, self.page.lines[-1].bbox)

    def to_page_x(self, native: float) -> int:
        """Scale a native x coordinate to page pixels."""
        return int(native * self.scale_x)

    def to_page_y(self, native: float) -> int:
        """Scale a native y coordinate to page pixels."""
        return int(native * self.scale_y)

    # -- internals ---------------------------------------------------------

    def _put(
        self,
        block: str,
        line: str,
        hand: Hand,
        baseline_y: int,
        x_start: int,
        x_end: int,
    ) -> None:
        """Place one line in the middle of its span, give or take.

        A clerk aims for the middle of the space a form gives them rather
        than crowding its left end, but never hits it exactly, so the value
        is centred and then nudged either way within its slack.
        """
        left = self.to_page_x(x_start)
        free = max(0.0, (self.to_page_x(x_end) - left) - hand.measure(line))
        share = 0.5 + self.rng.uniform(-_CENTRE_JITTER, _CENTRE_JITTER)
        # Clerks write on the rule or just above it, never below.
        top = self.to_page_y(baseline_y) - int(
            self.font_size * self.rng.uniform(0.05, 0.25)
        )
        self.page.put_lines(
            block,
            [line],
            hand,
            left + int(free * share),
            top,
            record_block=False,
        )

    def _fit_lines(
        self,
        text: str,
        hand: Hand,
        segments: Sequence[Any],
        widths: Sequence[int],
    ) -> tuple[Hand, list[str]]:
        """Fit a value to a field: shrink until it fits, then wrap.

        Both constraints matter. Wrapping alone cannot save a single word
        wider than its cell — many form cells hold one long month name — so
        the hand shrinks until every line fits its own segment as well as
        the field fitting its line count.

        Args:
            text: The value to write.
            hand: The hand it would be written in.
            segments: The underline segments available for this field.
            widths: Page-pixel width of each segment.

        Returns:
            ``(hand, lines)`` with at most one line per segment, each line
            no wider than the segment it goes on.
        """
        hand = self._compress_to_gap(hand, segments)
        lines = wrap_text(text, hand, widths[0])

        for _ in range(_MAX_FIT_PASSES):
            overflow = max(
                (
                    hand.measure(line) - width
                    for line, width in zip(lines, widths)
                ),
                default=0.0,
            )
            fits = len(lines) <= len(segments) and overflow <= 0
            if fits or hand.size <= _MIN_HAND_SIZE:
                break
            hand = hand.resized(max(_MIN_HAND_SIZE, int(hand.size * _FIT_STEP)))
            lines = wrap_text(text, hand, widths[0])

        if len(lines) > len(segments):
            logger.warning(
                "Value %r does not fit its %d line(s); truncating",
                text,
                len(segments),
            )
            lines = lines[: len(segments)]
        return hand, lines

    def _compress_to_gap(self, hand: Hand, segments: Sequence[Any]) -> Hand:
        """Shrink a hand so consecutive lines of a field do not collide.

        Some fields are printed as two rules only ~20px apart. A clerk
        writing across them compresses their hand rather than letting the
        two lines overlap, and so must we: overlapping ink would make both
        lines unreadable while the ground truth still claims two clean ones.

        Args:
            hand: The hand the field would be written in.
            segments: The underline segments available for this field.

        Returns:
            A hand whose lines fit between the rules.
        """
        if len(segments) < 2:
            return hand

        gaps = [
            self.to_page_y(later.baseline_y)
            - self.to_page_y(earlier.baseline_y)
            for earlier, later in itertools.pairwise(segments)
        ]
        allowed = int(min(gaps) * _MULTILINE_GAP_RATIO)
        if allowed >= hand.size:
            return hand
        return hand.resized(max(_MIN_HAND_SIZE, allowed))

    @staticmethod
    def _shrink(text: str, hand: Hand, width: float) -> Hand:
        """Shrink a hand until the text fits the given width.

        Staying inside the printed line matters more than staying legible:
        an overflowing value would put ink where no annotation claims it is.
        Values that need more than :data:`_MIN_FIT_RATIO` of the clerk's
        normal size are therefore still shrunk, down to a hard floor.

        Args:
            text: The value to fit.
            hand: The hand it would be written in.
            width: Available width in pixels.

        Returns:
            A hand small enough to fit, or the smallest allowed one.
        """
        original_size = hand.size
        size = original_size
        while hand.measure(text) > width and size > _MIN_HAND_SIZE:
            size = max(_MIN_HAND_SIZE, int(size * _FIT_STEP))
            hand = hand.resized(size)
        if hand.size < original_size * _MIN_FIT_RATIO:
            logger.debug(
                "Value %r had to shrink below %.0f%% to fit %d px",
                text,
                _MIN_FIT_RATIO * 100,
                width,
            )
        return hand


def _overlaps(first: BoundingBox, second: BoundingBox) -> bool:
    """Return whether two boxes share any area."""
    return (
        first.left < second.right
        and second.left < first.right
        and first.top < second.bottom
        and second.top < first.bottom
    )
