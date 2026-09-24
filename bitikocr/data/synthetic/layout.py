"""Page composition: where ink lands, and what ground truth it produces.

:class:`Page` is the one place that knows both about the image being drawn and
about the annotation being collected. A document generator decides *where*
things go; the page draws them and records what it drew.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from PIL import Image

from bitikocr.data.models.geometry import BoundingBox
from bitikocr.data.synthetic.annotation import (
    BlockAnnotation,
    DocumentAnnotation,
    LineAnnotation,
)
from bitikocr.data.synthetic.effects import draw_scribble
from bitikocr.data.synthetic.hand import Hand
from bitikocr.data.synthetic.ink import alpha_bounding_box
from bitikocr.data.synthetic.style import Color, HandwritingStyle
from bitikocr.data.synthetic.transcript import (
    SIGNATURE_MARK,
    Column,
    MarkKind,
    Piece,
    compose,
)

__all__ = ["CLIPPED_KEY", "EDGE_MARGIN", "Page", "wrap_text"]

logger = logging.getLogger(__name__)

#: Metadata key listing the blocks whose ink the page edge cut off. A page
#: with any is one whose transcription claims letters the image lacks.
CLIPPED_KEY = "clipped_blocks"

#: How close to the edge a line that had to be moved back is left, in
#: pixels.
EDGE_MARGIN = 4


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
        #: Blocks whose written text the page edge cut into.
        self.clipped: set[str] = set()

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
        rule_y: int | None = None,
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
            rule_y: The printed rule a single line is written on, in page
                pixels. A clerk writes a little above the rule, but the
                rule is what the value shares with the label printed on it,
                so it is recorded as the line's baseline instead of ``y``.

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

            position = self._inside(
                rendered, (left - pen_offset, y - hand.baseline)
            )
            self.image.alpha_composite(rendered, position)

            box = self._clip(alpha_bounding_box(rendered, position), block)
            self.lines.append(
                LineAnnotation(
                    block=block,
                    text=line,
                    bbox=box,
                    baseline=y if rule_y is None else rule_y,
                )
            )
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
        box = self._clip(box, block, carries_text=False)
        self.add_block(block, "", box)
        return box

    def _inside(
        self, rendered: Image.Image, position: tuple[int, int]
    ) -> tuple[int, int]:
        """Move a rendered line so its ink stays on the page.

        A line is placed by rule, and a long name in a large hand can carry
        it past an edge, where the ink would be cut off while the
        transcription kept every letter. Such a line is moved back inside
        instead; a line that fits is left exactly where it was placed.

        Args:
            rendered: The line's ink.
            position: Where its top-left corner would go.

        Returns:
            Where it goes.
        """
        box = alpha_bounding_box(rendered, position)
        if box is None:
            return position
        x, y = position
        if box.right > self.width:
            x -= box.right - self.width + EDGE_MARGIN
        if box.bottom > self.height:
            y -= box.bottom - self.height + EDGE_MARGIN
        # The left and top edges win: a line wider than the page is kept
        # whole at its start rather than at its end.
        x += max(0, EDGE_MARGIN - (box.left + x - position[0]))
        y += max(0, EDGE_MARGIN - (box.top + y - position[1]))
        return x, y

    def _clip(
        self, box: BoundingBox | None, block: str, carries_text: bool = True
    ) -> BoundingBox | None:
        """Trim a box to the page, since ink beyond it is never drawn.

        A box reaching past the canvas would claim ink the page does not
        carry. That is always a layout bug, so it is logged rather than
        quietly trimmed away.

        Args:
            box: The measured box, or None when nothing was drawn.
            block: Name of the block being recorded, for the log.
            carries_text: Whether the ink is transcribed. A scribble cut by
                the edge is still a scribble; cut text is a wrong label.

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
            if carries_text:
                self.clipped.add(block)
            return None
        if clipped != box:
            logger.warning("%s was drawn partly off the page", block)
            if carries_text:
                self.clipped.add(block)
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
        readable: Collection[str],
        metadata: dict[str, Any] | None = None,
        marks: Mapping[str, MarkKind] | None = None,
        columns: Sequence[Column] = (),
        printed: Collection[str] = (),
    ) -> DocumentAnnotation:
        """Collect the ground truth for everything drawn so far.

        The transcription is read off where things landed, not the order
        they were drawn in: see :func:`~bitikocr.data.synthetic.transcript.compose`.

        Args:
            readable: Names of the blocks whose lines are transcribed.
                Lines of any other block contribute nothing.
            metadata: Generator-specific extras merged into the JSON output.
            marks: Blocks transcribed as a mark rather than as their lines,
                and which mark each is.
            columns: The facing sheets the page is printed as, if more
                than one.
            printed: Names of the blocks that are type rather than
                handwriting, whose lines sit squarely on their row.

        Returns:
            The page's full annotation.
        """
        wanted = set(readable)
        pieces = [
            Piece(
                line.text,
                line.bbox,
                "print" if line.block in printed else "writing",
                line.baseline,
            )
            for line in self.lines
            if line.block in wanted and line.text and line.bbox is not None
        ]
        for block in self.blocks:
            kind = (marks or {}).get(block.kind)
            if kind is None or block.bbox is None:
                continue
            text = block.text if kind == "stamp" else SIGNATURE_MARK
            pieces.append(Piece(text, block.bbox, kind))

        return DocumentAnnotation(
            text=compose(pieces, columns),
            blocks=list(self.blocks),
            lines=list(self.lines),
            size=(self.width, self.height),
            metadata={
                **(metadata or {}),
                **({CLIPPED_KEY: sorted(self.clipped)} if self.clipped else {}),
            },
        )
