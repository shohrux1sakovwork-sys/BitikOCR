"""Page composition: where ink lands, and what ground truth it produces.

:class:`Page` is the one place that knows both about the image being drawn and
about the annotation being collected. A document generator decides *where*
things go; the page draws them and records what it drew.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from PIL import Image

from bitikocr.models.annotation import (
    BlockAnnotation,
    DocumentAnnotation,
    LineAnnotation,
)
from bitikocr.models.geometry import BoundingBox
from bitikocr.synthetic.effects import draw_scribble
from bitikocr.synthetic.hand import Hand
from bitikocr.synthetic.style import Color, HandwritingStyle
from bitikocr.utils.image_ops import alpha_bounding_box

__all__ = ["Page", "wrap_text"]

logger = logging.getLogger(__name__)


def wrap_text(
    text: str, hand: Hand, max_width: float, first_indent: float = 0.0
) -> list[str]:
    """Greedily wrap text to a width, measured in the hand that will write it.

    Paragraphs are split on newlines; blank paragraphs are dropped.

    Args:
        text: The text to wrap.
        hand: The hand whose metrics decide how wide a word is.
        max_width: Available width in pixels.
        first_indent: Width taken by the first line's indent, in pixels.

    Returns:
        One string per rendered line.
    """
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            continue

        current: list[str] = []
        available = max_width - first_indent if not lines else max_width
        for word in words:
            candidate = " ".join(current + [word])
            if current and hand.measure(candidate) > available:
                lines.append(" ".join(current))
                current = [word]
                available = max_width
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
    return lines


class Page:
    """A canvas that places handwriting and accumulates its ground truth.

    Args:
        size: Page size as ``(width, height)`` in pixels.
        style: The writer style; supplies ink, paper and spacing behaviour.
        rng: Random source for per-line slope and spacing jitter.
        background: A blank form scan to write on. When omitted the page is
            filled with the style's paper colour instead.
    """

    def __init__(
        self,
        size: tuple[int, int],
        style: HandwritingStyle,
        rng: random.Random,
        background: Image.Image | None = None,
    ) -> None:
        self.width, self.height = size
        self.style = style
        self.rng = rng
        self.lines: list[LineAnnotation] = []
        self.blocks: list[BlockAnnotation] = []

        if background is None:
            self.image = Image.new("RGBA", size, style.paper + (255,))
        else:
            self.image = background.convert("RGBA")
            if self.image.size != size:
                self.image = self.image.resize(size, Image.Resampling.LANCZOS)

    # -- per-line randomness -----------------------------------------------

    def line_slope(self) -> float:
        """Sample how far this line runs uphill, in degrees."""
        return self.style.line_slope + self.rng.gauss(
            0, self.style.line_slope_jitter
        )

    def line_step(self, hand: Hand) -> int:
        """Sample the baseline-to-baseline distance for the given hand."""
        jitter = 1 + self.rng.gauss(0, self.style.line_spacing_jitter)
        return int(hand.size * self.style.line_spacing * jitter)

    # -- drawing -----------------------------------------------------------

    def put_lines(
        self,
        block: str,
        lines: list[str],
        hand: Hand,
        x: int,
        y: int,
        color: Color | None = None,
        indent_first: int = 0,
        align_right_edge: int | None = None,
        record_block: bool = True,
    ) -> int:
        """Write lines top-down and record them as one block.

        Args:
            block: Name recorded for these lines.
            lines: The lines to write, already wrapped.
            hand: The hand to write them in.
            x: Left edge of the text, in page pixels.
            y: Baseline of the first line, in page pixels.
            color: Ink colour. Defaults to the page style's ink.
            indent_first: Extra left offset applied to the first line only.
            align_right_edge: Right-align every line to this x instead of
                left-aligning to ``x``.
            record_block: Whether to append a block annotation. Pass False
                when the caller merges several calls into one block itself.

        Returns:
            The baseline y for the line that would follow the last one.
        """
        ink = self.style.ink if color is None else color
        boxes: list[BoundingBox | None] = []

        for index, line in enumerate(lines):
            rendered, pen_offset = hand.render_line(
                line, ink, self.line_slope()
            )
            left = x + (indent_first if index == 0 else 0)
            if align_right_edge is not None:
                left = align_right_edge - int(hand.measure(line)) - pen_offset

            position = (left - pen_offset, y - hand.baseline)
            self.image.alpha_composite(rendered, position)

            box = self._clip(alpha_bounding_box(rendered, position), block)
            self.lines.append(LineAnnotation(block=block, text=line, bbox=box))
            boxes.append(box)
            y += self.line_step(hand)

        if record_block:
            self.add_block(block, "\n".join(lines), BoundingBox.union(boxes))
        return y

    def put_scribble(
        self,
        x: int,
        y: int,
        size: int,
        color: Color | None = None,
        block: str = "signature",
        max_width: int | None = None,
    ) -> BoundingBox | None:
        """Draw a signature scribble and record it as a block.

        Args:
            x: Left edge of the scribble.
            y: Top edge of the scribble.
            size: Nominal handwriting size the scribble is scaled against.
            color: Ink colour. Defaults to the page style's ink.
            block: Name recorded for the scribble.
            max_width: Total footprint the scribble must stay within.

        Returns:
            The box around the scribble, or None if nothing was drawn.
        """
        box = draw_scribble(
            page=self.image,
            x=x,
            y=y,
            size=size,
            color=self.style.ink if color is None else color,
            rng=self.rng,
            pen=self.style.pen,
            max_width=max_width,
        )
        box = self._clip(box, block)
        self.add_block(block, "", box)
        return box

    def _clip(self, box: BoundingBox | None, block: str) -> BoundingBox | None:
        """Trim a box to the page, since ink beyond it is never drawn.

        A box reaching past the canvas would claim ink the page does not
        carry. That is always a layout bug, so it is logged rather than
        quietly trimmed away.

        Args:
            box: The measured box, or None when nothing was drawn.
            block: Name of the block being recorded, for the log.

        Returns:
            The box within the page, or None if none of it landed on it.
        """
        if box is None:
            return None

        clipped = BoundingBox(
            left=max(0, box.left),
            top=max(0, box.top),
            right=min(self.width, box.right),
            bottom=min(self.height, box.bottom),
        )
        if clipped.right <= clipped.left or clipped.bottom <= clipped.top:
            logger.warning("%s was drawn entirely off the page", block)
            return None
        if clipped != box:
            logger.warning("%s was drawn partly off the page", block)
        return clipped

    # -- annotation bookkeeping --------------------------------------------

    def add_block(
        self, kind: str, text: str, bbox: BoundingBox | None
    ) -> BlockAnnotation:
        """Record a block that was drawn by other means, such as a stamp."""
        annotation = BlockAnnotation(kind=kind, text=text, bbox=bbox)
        self.blocks.append(annotation)
        return annotation

    def add_line(
        self, block: str, text: str, bbox: BoundingBox | None
    ) -> LineAnnotation:
        """Record a line that was drawn by other means, such as printed text."""
        annotation = LineAnnotation(block=block, text=text, bbox=bbox)
        self.lines.append(annotation)
        return annotation

    # -- results -----------------------------------------------------------

    def render(self) -> Image.Image:
        """Return the finished page as an RGB image."""
        return self.image.convert("RGB")

    def annotation(
        self,
        reading_order: tuple[str, ...],
        metadata: dict[str, Any] | None = None,
    ) -> DocumentAnnotation:
        """Collect the ground truth for everything drawn so far.

        Args:
            reading_order: Block names in the order a human would read them.
                Blocks whose name is absent contribute no transcription.
            metadata: Generator-specific extras merged into the JSON output.

        Returns:
            The page's full annotation.
        """
        transcription = [
            block.text
            for kind in reading_order
            for block in self.blocks
            if block.kind == kind and block.text
        ]
        return DocumentAnnotation(
            text="\n".join(transcription),
            blocks=list(self.blocks),
            lines=list(self.lines),
            size=(self.width, self.height),
            metadata=dict(metadata or {}),
        )
