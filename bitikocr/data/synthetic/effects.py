"""Marks that are not handwriting: signature scribbles and office stamps.

These marks carry no transcription but they do occupy space and they teach a
model that a page is not only handwriting, so every one of them still reports
a bounding box.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from bitikocr.data.models.geometry import BoundingBox
from bitikocr.data.synthetic.ink import alpha_bounding_box
from bitikocr.data.synthetic.style import Color, PenKind

__all__ = [
    "NUMBER_SIGN",
    "SCRIBBLE_WIDTH_RANGE",
    "SEAL_COLORS",
    "STAMP_REACH",
    "StampBlank",
    "box_stamp_size",
    "draw_box_stamp",
    "draw_round_stamp",
    "draw_scribble",
]

#: How far a seal's ink reaches from its centre, as a multiple of the
#: ring radius. The curved lettering sits outside the ring, so a caller
#: placing a seal has to leave this much room, not just the radius.
STAMP_REACH = 1.3

#: Stamp-pad inks an office seal is pressed in. Uzbek civil seals are
#: violet or blue, never black.
SEAL_COLORS: tuple[Color, ...] = (
    (70, 40, 150),
    (40, 60, 170),
    (60, 30, 130),
)

_SCRIBBLE_MARGIN = 20

#: A scribble spans this many times its nominal size, before clamping.
SCRIBBLE_WIDTH_RANGE = (2.5, 4.5)

#: A round seal is pressed at any angle; a rectangular one is squared up by
#: eye against the page, so it leans only a little.
_ROUND_TILT = 25.0
_BOX_TILT = 3.0

#: A row of a rectangular stamp ending in this sign leaves room after it for
#: a number to be written in.
NUMBER_SIGN = "№"

# A rectangular stamp's rows are spaced this many times its type size, and
# its lettering sits this many type sizes inside the frame.
_BOX_ROW_STEP = 1.35
_BOX_PADDING = 0.7


@dataclass(frozen=True)
class StampBlank:
    """Room a stamp leaves for a clerk to write in.

    Args:
        left: Where the room starts, in page pixels.
        right: Where it ends.
        baseline: The line to write along.
    """

    left: int
    right: int
    baseline: int


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
    size = int(radius * STAMP_REACH * 2)
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

    layer = _weather_stamp(layer, rng, _ROUND_TILT)
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


def draw_box_stamp(
    page: Image.Image,
    top_left: tuple[int, int],
    rows: Sequence[str],
    type_size: int,
    blank_width: int,
    color: Color,
    font_path: Path,
    rng: random.Random,
) -> tuple[BoundingBox | None, list[StampBlank]]:
    """Press a rectangular office stamp that leaves room to be written in.

    This is the stamp an office presses on the letters it receives: the
    office's name, then a row ending in ``№`` for the number it files the
    letter under, then usually an empty row for the date. The clerk writes
    both in by hand, so the stamp only reports where they go.

    Args:
        page: The RGBA page to composite onto, modified in place.
        top_left: Where the stamp's frame starts, in page pixels.
        rows: What each row prints. A row ending in ``№`` leaves room after
            it; an empty row is room across the whole stamp.
        type_size: Size of the stamp's lettering, in pixels.
        blank_width: How much room a ``№`` row leaves after the sign.
        color: Stamp-pad colour as RGB.
        font_path: A print font able to render the lettering.
        rng: Random source for the tilt and the patchy ink.

    Returns:
        ``(box, blanks)``: the box around the stamp's ink, or None if none
        landed, and the room left in it, top to bottom, in page pixels.
    """
    font = ImageFont.truetype(str(font_path), type_size)
    padding = int(type_size * _BOX_PADDING)
    step = int(type_size * _BOX_ROW_STEP)
    widths = _row_widths(rows, font, blank_width)
    width, height = box_stamp_size(rows, type_size, blank_width, font_path)

    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    frame = max(2, type_size // 9)
    draw.rectangle([0, 0, width - 1, height - 1], outline=255, width=frame)
    inset = frame * 3
    draw.rectangle(
        [inset, inset, width - 1 - inset, height - 1 - inset],
        outline=255,
        width=max(1, frame // 2),
    )

    left, top = top_left
    blanks: list[StampBlank] = []
    for index, (row, row_width) in enumerate(zip(rows, widths)):
        row_top = padding + index * step
        baseline = row_top + type_size
        if not row:
            # Handwriting rises higher than type, so an empty row is written
            # along its foot to keep clear of the printed row above.
            blanks.append(
                StampBlank(
                    left + padding, left + width - padding, top + row_top + step
                )
            )
            continue
        x: float
        if _leaves_room(row):
            x = padding
            printed = font.getlength(row)
            blanks.append(
                StampBlank(
                    int(left + x + printed + type_size * 0.3),
                    left + width - padding,
                    top + baseline,
                )
            )
        else:
            x = (width - row_width) / 2
        draw.text((x, baseline), row, font=font, fill=255, anchor="ls")

    layer = _weather_stamp(layer, rng, _BOX_TILT)
    stamp = Image.new("RGBA", layer.size, color + (0,))
    stamp.putalpha(layer)
    page.alpha_composite(stamp, (left, top))
    return alpha_bounding_box(stamp, (left, top)), blanks


def box_stamp_size(
    rows: Sequence[str], type_size: int, blank_width: int, font_path: Path
) -> tuple[int, int]:
    """Return how large a rectangular stamp will be, before pressing it.

    Args:
        rows: What each row prints; see :func:`draw_box_stamp`.
        type_size: Size of the stamp's lettering, in pixels.
        blank_width: How much room a ``№`` row leaves after the sign.
        font_path: The print font the lettering is set in.

    Returns:
        ``(width, height)`` of the stamp's frame, in pixels.
    """
    font = ImageFont.truetype(str(font_path), type_size)
    padding = int(type_size * _BOX_PADDING)
    step = int(type_size * _BOX_ROW_STEP)
    width = int(max(_row_widths(rows, font, blank_width)) + 2 * padding)
    height = int(step * len(rows) + 2 * padding - (step - type_size))
    return width, height


def _row_widths(
    rows: Sequence[str], font: ImageFont.FreeTypeFont, blank_width: int
) -> list[float]:
    """Return how wide each row of a stamp is, the room it leaves included."""
    return [
        font.getlength(row) + (blank_width if _leaves_room(row) else 0)
        for row in rows
    ]


def _leaves_room(row: str) -> bool:
    """Whether a stamp's row ends in a sign that a number is written after."""
    return row.rstrip().endswith(NUMBER_SIGN)


def _weather_stamp(
    layer: Image.Image, rng: random.Random, max_tilt: float
) -> Image.Image:
    """Tilt a stamp and make its ink patchy and semi-transparent."""
    layer = layer.rotate(
        rng.uniform(-max_tilt, max_tilt), resample=Image.Resampling.BICUBIC
    )

    width, height = layer.size
    noise_rng = np.random.default_rng(rng.randrange(2**31))
    coarse = (noise_rng.random((height // 6 + 1, width // 6 + 1)) * 255).astype(
        np.uint8
    )
    noise = (
        Image.fromarray(coarse)
        .resize((width, height), Image.Resampling.BILINEAR)
        .filter(ImageFilter.GaussianBlur(3))
    )

    patchy = np.asarray(layer).astype(np.float32) * (
        0.35 + 0.65 * np.asarray(noise) / 255.0
    )
    patchy *= rng.uniform(0.55, 0.85)
    return Image.fromarray(np.clip(patchy, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.6)
    )
