"""Redrawing a written line as a ballpoint lays it down.

A handwriting font is drawn with a broad nib: its strokes swell and thin,
and every copy of a letter is outlined the same way. A ballpoint does none
of that. It leaves a line of one width, a little darker where the writer
pressed and lighter where the ball ran dry, and now and then a skip.

So a written line is traced back to the path the pen took — its centre line,
one pixel wide — and that path is inked again with a ballpoint. The letters
keep their shapes; they lose the font's calligraphy.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter

__all__ = ["centre_line", "redraw_ballpoint"]

#: Share of the line's darkest ink a pixel must carry to count as part of a
#: stroke when the line is traced. Below it is only the anti-aliased fringe.
#: It is a share rather than a level because a hand shrunk into a narrow
#: cell has strokes too fine for any pixel to reach full ink.
_INK_THRESHOLD = 0.43

#: Size of the pressure field's cells, times the hand's nominal size. A
#: writer presses harder or lighter over a few letters, not pixel by pixel.
_PRESSURE_CELL = 0.9

#: How much ink the lightest touch still leaves. A ballpoint line is dark
#: and saturated; pressure shades it, it does not fade it.
_LIGHTEST_PRESS = 0.75

#: Size of the cells a ball skips over, in pixels, and the share of them it
#: skips. A skip is a short pale gap along the line, not a hole.
_SKIP_CELL = 3
_SKIP_SHARE = 0.06
_SKIP_DEPTH = 0.55

#: Blur that anti-aliases the redrawn line, in pixels. A ballpoint's edge is
#: crisp; this only takes the stair-steps off it.
_EDGE_SOFTNESS = 0.45

#: Widest line, in pixels, that the one-pixel path gives once softened.
#: Anything finer is drawn on the path alone, darkened by up to this gain.
_FINEST_LINE = 2.2
_MAX_LINE_GAIN = 1.8


def centre_line(mask: np.ndarray) -> np.ndarray:
    """Thin a stroke mask down to its one-pixel centre line.

    This is Zhang and Suen's thinning: the stroke's border is peeled away a
    layer at a time, from alternate sides, until only pixels whose removal
    would break the stroke or shorten a line's end remain.

    Args:
        mask: The stroke, as a boolean array.

    Returns:
        The centre line, as a boolean array of the same shape.
    """
    image = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            removable = _removable(image, step)
            if removable.any():
                image[1:-1, 1:-1][removable] = 0
                changed = True
        if not changed:
            return image[1:-1, 1:-1].astype(bool)


def _removable(image: np.ndarray, step: int) -> np.ndarray:
    """Return the border pixels one thinning sub-step may remove."""
    p2 = image[:-2, 1:-1]
    p3 = image[:-2, 2:]
    p4 = image[1:-1, 2:]
    p5 = image[2:, 2:]
    p6 = image[2:, 1:-1]
    p7 = image[2:, :-2]
    p8 = image[1:-1, :-2]
    p9 = image[:-2, :-2]
    ring = (p2, p3, p4, p5, p6, p7, p8, p9, p2)

    neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
    # Count the 0 -> 1 steps walking once round the pixel: exactly one means
    # the pixel sits on a single stretch of border, not at a junction.
    crossings = sum(
        ((ring[index] == 0) & (ring[index + 1] == 1)).astype(np.uint8)
        for index in range(8)
    )
    if step == 0:
        side = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
    else:
        side = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
    return (
        (image[1:-1, 1:-1] == 1)
        & (neighbours >= 2)
        & (neighbours <= 6)
        & (crossings == 1)
        & side
    )


def redraw_ballpoint(
    ink: Image.Image, width: float, size: int, rng: random.Random
) -> Image.Image:
    """Redraw a written line as a ballpoint would have drawn it.

    Args:
        ink: The line's ``L`` mask, as the font drew it.
        width: The ballpoint's line width, in pixels.
        size: The hand's nominal size, which sets how far pressure carries.
        rng: Random source for the pressure and the skips.

    Returns:
        A new ``L`` mask of the same size: the line's centre line, inked at
        ``width`` with the writer's pressure and the ball's skips.
    """
    box = ink.getbbox()
    if box is None:
        return ink
    # Work only where there is ink; a line's canvas is mostly margin.
    left, top, right, bottom = box
    pad = int(width) + 2
    left, top = max(0, left - pad), max(0, top - pad)
    right, bottom = min(ink.width, right + pad), min(ink.height, bottom + pad)
    region = np.asarray(ink.crop((left, top, right, bottom)))

    path = centre_line(region >= _INK_THRESHOLD * int(region.max()))
    if not path.any():
        # Nothing to trace would leave written text with no ink at all, so
        # the font's own stroke is kept instead.
        return ink
    stroke = _ink_path(path, width)
    stroke *= _pressure(stroke.shape, size, rng)
    stroke *= _skips(stroke.shape, rng)

    redrawn = Image.new("L", ink.size, 0)
    redrawn.paste(
        Image.fromarray(np.clip(stroke * 255, 0, 255).astype(np.uint8)),
        (left, top),
    )
    return redrawn


def _ink_path(path: np.ndarray, width: float) -> np.ndarray:
    """Lay a line of the given width along a one-pixel path.

    The path is widened by whole pixels on both sides, so to an odd width
    — one, three, five — then softened so its edge is anti-aliased. A fine
    ball stays on the one-pixel path, which the softening spreads to about
    two; the width a whole pixel cannot give is given as opacity instead.

    Returns:
        Ink coverage in ``[0, 1]``.
    """
    mask = Image.fromarray(path.astype(np.uint8) * 255)
    whole = 1 + 2 * max(0, int((width - _FINEST_LINE) / 2 + 1))
    if whole > 1:
        mask = mask.filter(ImageFilter.MaxFilter(whole))
    mask = mask.filter(ImageFilter.GaussianBlur(_EDGE_SOFTNESS))
    coverage = np.asarray(mask).astype(np.float32) / 255.0
    gain = min(_MAX_LINE_GAIN, 0.35 + width / whole)
    return np.clip(coverage * gain, 0.0, 1.0)


def _pressure(
    shape: tuple[int, ...], size: int, rng: random.Random
) -> np.ndarray:
    """Return how hard the writer pressed at each point, as ink from 0 to 1."""
    height, width = shape[:2]
    cell = max(4, int(size * _PRESSURE_CELL))
    noise_rng = np.random.default_rng(rng.randrange(2**31))
    coarse = noise_rng.random((height // cell + 2, width // cell + 2))
    field = Image.fromarray((coarse * 255).astype(np.uint8)).resize(
        (width + cell, height + cell), Image.Resampling.BICUBIC
    )
    values = np.asarray(field).astype(np.float32)[:height, :width] / 255.0
    return _LIGHTEST_PRESS + (1.0 - _LIGHTEST_PRESS) * np.clip(values, 0.0, 1.0)


def _skips(shape: tuple[int, ...], rng: random.Random) -> np.ndarray:
    """Return where the ball ran dry for a moment: pale specks along a line."""
    height, width = shape[:2]
    noise_rng = np.random.default_rng(rng.randrange(2**31))
    coarse = noise_rng.random(
        (height // _SKIP_CELL + 1, width // _SKIP_CELL + 1)
    )
    dry = (coarse < _SKIP_SHARE).astype(np.uint8) * 255
    field = (
        Image.fromarray(dry)
        .resize((width, height), Image.Resampling.BILINEAR)
        .filter(ImageFilter.GaussianBlur(0.8))
    )
    return 1.0 - _SKIP_DEPTH * np.asarray(field).astype(np.float32) / 255.0
