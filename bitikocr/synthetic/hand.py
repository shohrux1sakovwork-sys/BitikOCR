"""The handwriting renderer: one font plus one style equals one "hand".

A :class:`Hand` turns a string into an RGBA image of handwritten ink. It owns
glyph synthesis (including the Uzbek letters most fonts lack), per-character
jitter, pen behaviour and the ink texture. It knows nothing about documents,
fields or page layout.
"""

from __future__ import annotations

import logging
import math
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from bitikocr.synthetic.fonts import (
    FALLBACK_BASE,
    SUBSTITUTE,
    FontInfo,
    open_font,
    pixel_size_for,
)
from bitikocr.synthetic.style import Color, HandwritingStyle

__all__ = ["Hand"]

logger = logging.getLogger(__name__)

# One Min/MaxFilter(3) pass changes stroke width by roughly two pixels.
_PIXELS_PER_MORPHOLOGY_PASS = 2.0

# Never erode away more than this share of a stroke or thin fonts fall apart.
_MAX_EROSION_SHARE = 0.55

_MISSING_GLYPH = "?"

# Ceiling on the opacity gain that stands in for sub-pixel stroke width.
# Beyond this a thin hand stops looking like ink and starts looking printed.
_MAX_INK_GAIN = 2.0


class Hand:
    """Render text lines in a single handwriting style.

    Args:
        info: The font this hand writes with.
        size: Nominal layout size in pixels. The font itself is opened at
            whatever pixel size gives every font the same x-height.
        style: The writer's style knobs.
        rng: Random source for per-character jitter and ink texture.
    """

    def __init__(
        self,
        info: FontInfo,
        size: int,
        style: HandwritingStyle,
        rng: random.Random,
    ) -> None:
        self.info = info
        self.size = size
        self.style = style
        self.rng = rng

        self.pixel_size = pixel_size_for(info, size)
        self.font: ImageFont.FreeTypeFont = open_font(
            info, self.pixel_size, rng
        )
        self._glyph_cache: dict[str, tuple[Image.Image, float]] = {}

        # Every glyph shares this canvas geometry so baselines line up.
        self.padding = int(size * 0.6)
        self.canvas_height = int(size * 3.2)
        self.baseline = int(size * 2.0)

        self.stroke_width = max(2, round(self.pixel_size * 0.045))
        self.space_advance = self.font.getlength(" ") * style.word_spacing
        self.letter_gap = style.letter_spacing * size

    def resized(self, size: int) -> Hand:
        """Return the same hand writing at a different nominal size.

        Args:
            size: The new nominal layout size in pixels.

        Returns:
            A new hand; this one is left untouched.
        """
        return Hand(self.info, size, self.style, self.rng)

    # -- glyphs ------------------------------------------------------------

    def _draw_uzbek_mark(
        self, draw: ImageDraw.ImageDraw, char: str, base: str
    ) -> None:
        """Draw the diacritic or descender the font is missing.

        Args:
            draw: Drawing context over the glyph canvas.
            char: The Uzbek character being synthesised.
            base: The Cyrillic glyph already drawn on the canvas.
        """
        left, top, right, bottom = self.font.getbbox(base, anchor="ls")
        x0, x1 = self.padding + left, self.padding + right
        y_top, y_bottom = self.baseline + top, self.baseline + bottom
        width = max(1, x1 - x0)
        size = self.pixel_size
        pen = self.stroke_width
        is_lower = char.islower()

        if char in "ҳҲ":
            # A small descender hooking down-left off the bottom-right leg.
            x = x1 - width * (0.22 if is_lower else 0.18)
            draw.line(
                [
                    (x, y_bottom - size * 0.02),
                    (x + width * 0.10, y_bottom + size * 0.14),
                    (x - width * 0.18, y_bottom + size * 0.22),
                ],
                fill=255,
                width=pen,
                joint="curve",
            )
        elif char in "қҚ":
            x = x1 - width * (0.10 if is_lower else 0.08)
            draw.line(
                [
                    (x, y_bottom - size * 0.02),
                    (x + width * 0.06, y_bottom + size * 0.15),
                    (x - width * 0.20, y_bottom + size * 0.21),
                ],
                fill=255,
                width=pen,
                joint="curve",
            )
        elif char in "ғҒ":
            y = y_top + (y_bottom - y_top) * (0.50 if is_lower else 0.45)
            draw.line(
                [
                    (x0 - width * 0.15, y),
                    (x1 + width * 0.15, y - size * 0.01),
                ],
                fill=255,
                width=pen,
            )
        elif char in "ўЎ":
            centre_x = (x0 + x1) / 2
            radius_x, radius_y = width * 0.32, size * 0.11
            draw.arc(
                [
                    centre_x - radius_x,
                    y_top - radius_y * 2.3,
                    centre_x + radius_x,
                    y_top - radius_y * 0.3,
                ],
                start=20,
                end=160,
                fill=255,
                width=pen,
            )

    def glyph(self, char: str) -> tuple[Image.Image, float]:
        """Render one character on the shared glyph canvas.

        Args:
            char: The character to draw.

        Returns:
            ``(mask, advance)`` where mask is an ``L`` image on the common
            canvas and advance is the pen movement in pixels.
        """
        cached = self._glyph_cache.get(char)
        if cached is not None:
            return cached

        base, extra = char, None
        if not self.info.has_glyph(char):
            if char in FALLBACK_BASE:
                base, extra = FALLBACK_BASE[char], char
            else:
                base = SUBSTITUTE.get(char, _MISSING_GLYPH)

        advance = self.font.getlength(base)
        canvas_width = int(advance + self.padding * 2 + self.size)
        mask = Image.new("L", (canvas_width, self.canvas_height), 0)
        draw = ImageDraw.Draw(mask)
        draw.text(
            (self.padding, self.baseline),
            base,
            font=self.font,
            fill=255,
            anchor="ls",
        )
        if extra is not None:
            self._draw_uzbek_mark(draw, extra, base)

        self._glyph_cache[char] = (mask, advance)
        return mask, advance

    # -- measuring ---------------------------------------------------------

    def measure(self, text: str) -> float:
        """Return how wide ``text`` will be in pixels, in this hand."""
        width = 0.0
        for char in text:
            if char == " ":
                width += self.space_advance
            else:
                width += self.glyph(char)[1] + self.letter_gap
        return width

    # -- line rendering ----------------------------------------------------

    def render_line(
        self, text: str, color: Color, slope_deg: float = 0.0
    ) -> tuple[Image.Image, int]:
        """Render one line of handwriting.

        Args:
            text: The line to write.
            color: Ink colour as RGB.
            slope_deg: Rotation applied to the finished line, in degrees.

        Returns:
            ``(image, pen_offset)``. Place the image's top-left corner at
            ``(x - pen_offset, y - self.baseline)`` to put the first pen
            stroke at ``(x, y)`` on the page.
        """
        style, rng = self.style, self.rng
        width = int(self.measure(text) + self.padding * 2 + self.size * 2)
        ink = Image.new("L", (width, self.canvas_height), 0)

        # A smooth two-frequency wave gives the baseline a natural drift.
        amplitude = style.baseline_wobble * self.size
        freq_a, freq_b = rng.uniform(0.004, 0.012), rng.uniform(0.015, 0.03)
        phase_a, phase_b = rng.uniform(0, 6.3), rng.uniform(0, 6.3)

        x = float(self.padding)
        for char in text:
            if char == " ":
                x += self.space_advance
                continue

            glyph, advance = self.glyph(char)
            scale = 1.0 + rng.gauss(0, style.char_scale_jitter)
            rotation = rng.gauss(0, style.char_rot_jitter)
            offset_y = amplitude * (
                math.sin(x * freq_a + phase_a)
                + 0.5 * math.sin(x * freq_b + phase_b)
            )
            offset_y += rng.gauss(0, style.char_y_jitter * self.size)
            fade = style.ink_variation / max(0.1, style.ink_strength)
            alpha = 1.0 - abs(rng.gauss(0, fade))
            alpha = max(0.35, min(1.0, alpha))

            if abs(scale - 1.0) > 0.005:
                glyph = glyph.resize(
                    (
                        max(1, int(glyph.width * scale)),
                        max(1, int(glyph.height * scale)),
                    ),
                    Image.Resampling.BILINEAR,
                )
            if abs(rotation) > 0.2:
                glyph = glyph.rotate(
                    rotation, resample=Image.Resampling.BILINEAR, expand=False
                )
            if alpha < 0.995:
                glyph = glyph.point(lambda v, a=alpha: int(v * a))

            # Keep the baseline and the left origin fixed under scaling.
            paste_x = round(x - self.padding * scale)
            paste_y = round(self.baseline - self.baseline * scale + offset_y)
            # Compose with "lighter" so overlapping strokes do not erase
            # each other where cursive letters join.
            region = ink.crop(
                (
                    paste_x,
                    paste_y,
                    paste_x + glyph.width,
                    paste_y + glyph.height,
                )
            )
            ink.paste(ImageChops.lighter(region, glyph), (paste_x, paste_y))

            x += advance * scale + self.letter_gap

        ink = self._apply_pen(ink)
        ink = self._apply_ink_texture(ink)
        ink, slant_margin = self._apply_slant(ink)

        rendered = Image.new("RGBA", ink.size, color + (0,))
        rendered.putalpha(ink)
        if abs(slope_deg) > 0.05:
            rendered = rendered.rotate(
                slope_deg, resample=Image.Resampling.BICUBIC, expand=True
            )
        # Slanting widens the canvas on the left, moving the pen origin with
        # it; the caller subtracts this offset to put the first stroke back
        # where it asked for.
        return rendered, self.padding + slant_margin

    # -- post effects ------------------------------------------------------

    def _apply_pen(self, ink: Image.Image) -> Image.Image:
        """Bring the font's natural stroke to the target width, then texture it.

        Widening happens in whole erosion or dilation passes, each worth
        about two pixels. Most of the handwriting fonts are thinner than the
        pen asks for by less than that, so the whole adjustment used to round
        away and they wrote far too faintly. Whatever a whole pass cannot
        cover is applied as opacity instead, which darkens the anti-aliased
        edge and reads as a heavier nib.

        Args:
            ink: The ``L`` mask of the rendered line.

        Returns:
            The mask with the pen's stroke weight and edge quality applied.
        """
        natural = self.info.stroke_ratio * self.pixel_size
        wanted = self.style.stroke_px * self.style.ink_strength
        difference = wanted - natural
        passes = round(abs(difference) / _PIXELS_PER_MORPHOLOGY_PASS)

        if difference < 0:
            passes = min(
                passes,
                int(natural * _MAX_EROSION_SHARE / _PIXELS_PER_MORPHOLOGY_PASS),
            )
            ink = self._erode(ink, passes)
        else:
            for _ in range(passes):
                ink = ink.filter(ImageFilter.MaxFilter(3))
            ink = self._deepen(
                ink, difference - passes * _PIXELS_PER_MORPHOLOGY_PASS, natural
            )

        if self.style.pen == "soft":  # Felt or worn ballpoint: fuzzy edge.
            ink = ink.filter(ImageFilter.GaussianBlur(0.8))
        elif (
            self.style.pen == "hard"
        ):  # Hard ballpoint: crisp, a touch lighter.
            ink = ink.point(lambda v: int(v * 0.94))
        return ink

    @staticmethod
    def _deepen(
        ink: Image.Image, residual_px: float, natural: float
    ) -> Image.Image:
        """Darken a stroke by the fraction of a pixel dilation cannot add.

        Args:
            ink: The ``L`` mask of the rendered line.
            residual_px: Stroke width still wanted, under one dilation pass.
            natural: The font's own stroke width in pixels.

        Returns:
            The mask, with its soft edge pulled towards opaque.
        """
        if residual_px <= 0:
            return ink
        gain = 1.0 + min(_MAX_INK_GAIN - 1.0, residual_px / max(1.0, natural))
        return ink.point(lambda v: min(255, int(v * gain)))

    def _erode(self, ink: Image.Image, passes: int) -> Image.Image:
        """Thin the stroke, backing off before the line disappears.

        The pass count comes from the font's *measured* stroke width, which
        can overshoot. Since a blank line would still be reported as written
        text in the ground truth, erosion stops as soon as the mask empties
        and the last surviving state is kept.

        Args:
            ink: The ``L`` mask of the rendered line.
            passes: How many erosion passes to apply at most.

        Returns:
            The thinned mask, never an empty one.
        """
        if passes <= 0:
            return ink

        applied = 0
        for _ in range(passes):
            thinner = ink.filter(ImageFilter.MinFilter(3))
            if thinner.getbbox() is None:
                logger.debug(
                    "Stopped eroding %s after %d of %d passes to keep the "
                    "stroke visible",
                    self.info.name,
                    applied,
                    passes,
                )
                break
            ink, applied = thinner, applied + 1

        if applied:
            # Erosion is aliased; blur restores the soft edge and the point()
            # call keeps the stroke core solid.
            ink = ink.filter(ImageFilter.GaussianBlur(0.5))
            ink = ink.point(lambda v: min(255, int(v * 1.7)))
        return ink

    def _apply_ink_texture(self, ink: Image.Image) -> Image.Image:
        """Modulate the ink with smooth noise so the line is not evenly dark."""
        variation = self.style.ink_variation / max(0.1, self.style.ink_strength)
        if variation <= 0.02:
            return ink

        noise_rng = np.random.default_rng(self.rng.randrange(2**31))
        coarse = noise_rng.random(
            (ink.height // 4 + 1, ink.width // 4 + 1)
        ).astype(np.float32)
        noise = Image.fromarray((coarse * 255).astype(np.uint8))
        noise = noise.resize(ink.size, Image.Resampling.BILINEAR).filter(
            ImageFilter.GaussianBlur(2)
        )

        factor = 1.0 - variation * 1.4 * (
            np.asarray(noise).astype(np.float32) / 255.0
        )
        textured = np.asarray(ink).astype(np.float32) * factor
        return Image.fromarray(np.clip(textured, 0, 255).astype(np.uint8))

    def _apply_slant(self, ink: Image.Image) -> tuple[Image.Image, int]:
        """Shear the line about its baseline to lean the writing.

        Args:
            ink: The ``L`` mask of the rendered line.

        Returns:
            ``(mask, margin)`` where margin is how far the pen origin moved
            right to make room for the lean. Callers must add it to the pen
            offset they report, or the line lands that far off its mark.
        """
        shear = self.style.slant
        if abs(shear) < 0.01:
            return ink, 0

        width, height = ink.size
        margin = int(abs(shear) * height) + 2
        padded = Image.new("L", (width + 2 * margin, height), 0)
        padded.paste(ink, (margin, 0))
        sheared = padded.transform(
            padded.size,
            Image.Transform.AFFINE,
            (1, shear, -shear * self.baseline, 0, 1, 0),
            resample=Image.Resampling.BILINEAR,
        )
        return sheared, margin
