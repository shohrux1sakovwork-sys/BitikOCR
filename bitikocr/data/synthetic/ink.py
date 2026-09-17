"""Measuring the ink a rendered layer carries.

Ground truth is measured rather than predicted: a box comes from the pixels
that were actually drawn, not from the metrics that were meant to place
them. That is what keeps a box right through per-glyph jitter, slant, line
slope and a scan skew.
"""

from __future__ import annotations

from PIL import Image

from bitikocr.data.models.geometry import BoundingBox

__all__ = ["alpha_bounding_box"]


def alpha_bounding_box(
    image: Image.Image, offset: tuple[int, int] = (0, 0)
) -> BoundingBox | None:
    """Find the tight box around the opaque pixels of an RGBA image.

    Args:
        image: An RGBA image whose alpha channel marks the drawn ink.
        offset: Position the image is pasted at, added to the result.

    Returns:
        The box in page coordinates, or None if the image is fully
        transparent.
    """
    box = image.getchannel("A").getbbox()
    if box is None:
        return None
    return BoundingBox(
        left=box[0] + offset[0],
        top=box[1] + offset[1],
        right=box[2] + offset[0],
        bottom=box[3] + offset[1],
    )
