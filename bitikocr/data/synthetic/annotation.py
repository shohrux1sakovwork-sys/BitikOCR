"""What a generator drew, in the generator's own terms.

A page is built block by block and line by line, each with the box its ink
actually occupies, and this is the structure that collects them. It is
internal to the generator: :mod:`bitikocr.data.synthetic.export` maps it
onto the corpus schema, which is what gets written beside the image and
what a reader outside this package consumes.

That translation is the point of keeping the two apart. This structure is
free to carry whatever the renderer finds useful — a style, a seed, a font
— while :mod:`bitikocr.models.schema` stays a stable contract that real
scans and human annotators describe themselves in too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bitikocr.models.geometry import BoundingBox

__all__ = [
    "BlockAnnotation",
    "DocumentAnnotation",
    "LineAnnotation",
]


@dataclass
class LineAnnotation:
    """One rendered text line and the box its ink occupies.

    Args:
        block: Name of the block this line belongs to.
        text: Transcription of the line.
        bbox: Tight box around the rendered ink, or None if nothing was drawn.
    """

    block: str
    text: str
    bbox: BoundingBox | None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of this line."""
        return {
            "block": self.block,
            "text": self.text,
            "bbox": self.bbox.to_list() if self.bbox else None,
        }


@dataclass
class BlockAnnotation:
    """A logical region of the document, such as a form field or a paragraph.

    Args:
        kind: Block name, e.g. ``"body"``, ``"surname"`` or ``"stamp"``.
        text: Transcription of the block. Empty for non-textual blocks.
        bbox: Tight box around the rendered ink, or None if nothing was drawn.
    """

    kind: str
    text: str
    bbox: BoundingBox | None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of this block."""
        return {
            "type": self.kind,
            "text": self.text,
            "bbox": self.bbox.to_list() if self.bbox else None,
        }


@dataclass
class DocumentAnnotation:
    """Full ground truth for one generated page.

    Args:
        text: Page transcription, blocks joined in reading order.
        blocks: Every block laid out on the page.
        lines: Every rendered line, in the order it was drawn.
        size: Page size as ``(width, height)`` in pixels.
        metadata: Generator-specific extras (seed, style, font, fields, ...).
    """

    text: str
    blocks: list[BlockAnnotation]
    lines: list[LineAnnotation]
    size: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the annotation.

        Metadata keys are merged into the top level so that consumers can read
        ``gt["seed"]`` directly.
        """
        payload: dict[str, Any] = {
            "text": self.text,
            "blocks": [block.to_dict() for block in self.blocks],
            "lines": [line.to_dict() for line in self.lines],
            "size": list(self.size),
        }
        payload.update(self.metadata)
        return payload
