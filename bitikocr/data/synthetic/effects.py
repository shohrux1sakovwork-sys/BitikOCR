"""Non-textual ink: signature scribbles and round office seals.

These marks carry no transcription but they do occupy space and they teach a
model that a page is not only handwriting, so every one of them still reports
a bounding box.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from bitikocr.data.synthetic.ink import alpha_bounding_box
from bitikocr.data.synthetic.style import Color, PenKind
from bitikocr.models.geometry import BoundingBox

__all__ = ["SCRIBBLE_WIDTH_RANGE", "draw_round_stamp", "draw_scribble"]

_SCRIBBLE_MARGIN = 20

#: A scribble spans this many times its nominal size, before clamping.
SCRIBBLE_WIDTH_RANGE = (2.5, 4.5)


def draw_scribble(
    page: Image.Image,
    x: int,
    y: int,
    size: int,
    color: Color,
    rng: random.Random,
    pen: PenKind,
    max_width: int | None = None,
) -> BoundingBox | None:
    """Draw a signature-like scribble onto a page.

    The stroke is a Lissajous-style curve, which reads as a fluent single-pen
    movement rather than as random noise.

    Args:
        page: The RGBA page to composite onto, modified in place.
        x: Left edge of the scribble.
        y: Top edge of the scribble.
        size: Nominal handwriting size the scribble is scaled against.
        color: Ink colour as RGB.
        rng: Random source for the curve's shape.
        pen: Which pen signed; anything but ``"hard"`` gets a softer edge.
        max_width: Total footprint the scribble must stay within, in pixels.
            Signature areas butt up against printed marks, so a signature
            that ran long would land on them.

    Returns:
        The box around the drawn ink, or None if nothing was drawn.
    """
    width = int(size * rng.uniform(*SCRIBBLE_WIDTH_RANGE))
    if max_width is not None:
        width = max(1, min(width, max_width - 2 * _SCRIBBLE_MARGIN))
    height = int(size * rng.uniform(1.0, 1.8))
    layer = Image.new(
        "L", (width + 2 * _SCRIBBLE_MARGIN, height + 2 * _SCRIBBLE_MARGIN), 0
    )
    draw = ImageDraw.Draw(layer)

    point_count = rng.randint(60, 140)
    slow, fast, cross = (
        rng.uniform(1, 3),
        rng.uniform(3, 7),
        rng.uniform(0.5, 1.5),
    )
    phase = rng.uniform(0, 6.3)
    points = []
    for index in range(point_count):
        t = index / (point_count - 1)
        px = _SCRIBBLE_MARGIN + width * (
            t + 0.08 * math.sin(fast * t * 6.28 + phase)
        )
        py = _SCRIBBLE_MARGIN + height * (
            0.5
            + 0.35
            * math.sin(slow * t * 6.28)
            * math.cos(cross * t * 6.28 + phase)
            + 0.12 * math.sin(fast * t * 6.28)
        )
        points.append((px, py))
    draw.line(points, fill=255, width=max(2, round(size * 0.05)), joint="curve")

    if rng.random() < 0.6:  # Many signatures end in an underline flourish.
        draw.line(
            [
                (
                    _SCRIBBLE_MARGIN + width * 0.1,
                    _SCRIBBLE_MARGIN + height * 0.95,
                ),
                (
                    _SCRIBBLE_MARGIN + width * 1.05,
                    _SCRIBBLE_MARGIN + height * 0.85,
                ),
            ],
            fill=255,
            width=max(2, round(size * 0.04)),
        )

    if pen != "hard":
        layer = layer.filter(ImageFilter.GaussianBlur(0.6))

    scribble = Image.new("RGBA", layer.size, color + (0,))
    scribble.putalpha(layer)
    page.alpha_composite(scribble, (x, y))
    return alpha_bounding_box(scribble, (x, y))


def draw_round_stamp(
    page: Image.Image,
    centre: tuple[int, int],
    radius: int,
    ring_text: str,
    centre_lines: list[str],
    color: Color,
    font_path: Path,
    rng: random.Random,
) -> BoundingBox | None:
    """Draw a round office seal: two rings, curved text and a centred caption.

    Args:
        page: The RGBA page to composite onto, modified in place.
        centre: Where the middle of the seal lands, in page pixels.
        radius: Radius of the outer ring in pixels.
        ring_text: Text laid around the ring; drawn upper-cased.
        centre_lines: Short lines printed in the middle of the seal.
        color: Stamp-pad colour as RGB.
        font_path: A print font able to render the stamp lettering.
        rng: Random source for the tilt and the patchy ink.

    Returns:
        The box around the stamp's ink, or None if nothing was drawn. The
        ink is measured rather than assumed: the layer is padded well beyond
        the ring to leave room for the curved lettering, so its canvas would
        overstate what the seal covers.
    """
    size = int(radius * 2.6)
    layer = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(layer)
    middle = size // 2

    line_width = max(2, int(radius * 0.045))
    draw.ellipse(
        [
            middle - radius,
            middle - radius,
            middle + radius,
            middle + radius,
        ],
        outline=255,
        width=line_width,
    )
    inner = int(radius * 0.80)
    draw.ellipse(
        [middle - inner, middle - inner, middle + inner, middle + inner],
        outline=255,
        width=max(1, line_width // 2),
    )

    _draw_ring_text(layer, ring_text.upper(), middle, radius, font_path)
    _draw_centre_lines(draw, centre_lines, middle, radius, font_path)

    layer = _weather_stamp(layer, size, rng)
    stamp = Image.new("RGBA", layer.size, color + (0,))
    stamp.putalpha(layer)

    left, top = centre[0] - middle, centre[1] - middle
    page.alpha_composite(stamp, (left, top))
    return alpha_bounding_box(stamp, (left, top))


def _draw_ring_text(
    layer: Image.Image,
    text: str,
    middle: int,
    radius: int,
    font_path: Path,
) -> None:
    """Lay text around the seal's ring, one rotated glyph at a time."""
    font = ImageFont.truetype(str(font_path), max(8, int(radius * 0.19)))
    text_radius = int(radius * 0.90)
    count = max(1, len(text))
    step = 2 * math.pi * 0.92 / count
    start = -math.pi / 2 - (count - 1) * step / 2  # Centred on the top.

    for index, char in enumerate(text):
        angle = start + index * step
        glyph_w, glyph_h = font.getbbox(char)[2:]
        glyph = Image.new("L", (int(glyph_w) + 6, int(glyph_h) + 6), 0)
        ImageDraw.Draw(glyph).text((3, 3), char, font=font, fill=255)
        glyph = glyph.rotate(
            -math.degrees(angle) - 90,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
        x = middle + text_radius * math.cos(angle) - glyph.width / 2
        y = middle + text_radius * math.sin(angle) - glyph.height / 2
        layer.paste(glyph, (int(x), int(y)), glyph)


def _draw_centre_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    middle: int,
    radius: int,
    font_path: Path,
) -> None:
    """Print the short caption in the middle of the seal."""
    font = ImageFont.truetype(str(font_path), max(8, int(radius * 0.22)))
    step = font.size * 1.25
    top = middle - len(lines) * step / 2
    for index, line in enumerate(lines):
        width = font.getlength(line)
        draw.text(
            (middle - width / 2, top + index * step), line, font=font, fill=255
        )


def _weather_stamp(
    layer: Image.Image, size: int, rng: random.Random
) -> Image.Image:
    """Tilt the seal and make its ink patchy and semi-transparent."""
    layer = layer.rotate(
        rng.uniform(-25, 25), resample=Image.Resampling.BICUBIC
    )

    noise_rng = np.random.default_rng(rng.randrange(2**31))
    coarse = (noise_rng.random((size // 6 + 1, size // 6 + 1)) * 255).astype(
        np.uint8
    )
    noise = (
        Image.fromarray(coarse)
        .resize((size, size), Image.Resampling.BILINEAR)
        .filter(ImageFilter.GaussianBlur(3))
    )

    patchy = np.asarray(layer).astype(np.float32) * (
        0.35 + 0.65 * np.asarray(noise) / 255.0
    )
    patchy *= rng.uniform(0.55, 0.85)
    return Image.fromarray(np.clip(patchy, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.6)
    )
