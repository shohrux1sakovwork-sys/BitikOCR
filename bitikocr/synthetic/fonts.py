"""Handwriting font discovery, coverage checks and size equalisation.

Cyrillic handwriting fonts differ wildly in x-height, letter width and stroke
weight, and most of them lack the Uzbek letters ``ў қ ғ ҳ``. This module hides
both problems behind :class:`FontInfo` and :class:`FontLibrary`, so the
renderer can treat every font as if it had the same proportions.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

__all__ = [
    "FALLBACK_BASE",
    "SUBSTITUTE",
    "FontInfo",
    "FontLibrary",
    "open_font",
    "pixel_size_for",
]

logger = logging.getLogger(__name__)

# Uzbek letters many Cyrillic fonts lack, mapped to the base glyph we build
# them from by drawing the missing diacritic or descender by hand.
FALLBACK_BASE = {
    "ҳ": "х",
    "Ҳ": "Х",
    "қ": "к",
    "Қ": "К",
    "ғ": "г",
    "Ғ": "Г",
    "ў": "у",
    "Ў": "У",
}

# Characters drawn as a look-alike when the font lacks them (no extra strokes).
SUBSTITUTE = {
    "ʻ": "'",
    "ʼ": "'",
    "’": "'",
    "‘": "'",
    "`": "'",
    "ʹ": "'",
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "–": "-",
    "—": "-",
    "№": "N",
}

# Caveat-like proportions every font is rescaled towards.
TARGET_XHEIGHT = 0.42
TARGET_WIDTH = 0.46

_METRIC_SIZE = 100
_MAX_EROSION_PASSES = 30

_DEFAULT_METRICS = (0.45, 0.46, 0.08)

# Reference glyphs the measurements are taken on, per script. A font is
# measured with the first set it actually covers: measuring a Latin-only font
# on Cyrillic letters would read the notdef box instead of a letter and report
# a wildly wrong stroke width.
_REFERENCE_SETS: tuple[tuple[str, str, str], ...] = (
    ("хопс", "мамлакатнинг", "лпшн"),
    ("xons", "handwriting", "hnlu"),
)


@dataclass(frozen=True)
class _ReferenceGlyphs:
    """Sample text a font's proportions are measured on.

    Args:
        xheight_chars: Flat-topped lowercase letters, for the x-height.
        width_word: A representative word, for the average advance.
        stroke_chars: Letters with plain vertical strokes, for the stroke.
    """

    xheight_chars: str
    width_word: str
    stroke_chars: str


def _reference_glyphs(codepoints: frozenset[int]) -> _ReferenceGlyphs | None:
    """Pick the reference glyphs a font can actually draw.

    Args:
        codepoints: The code points the font covers.

    Returns:
        The first fully covered reference set, or None if the font covers
        none of them.
    """
    for xheight, word, stroke in _REFERENCE_SETS:
        sample = f"{xheight}{word}{stroke}"
        if all(ord(char) in codepoints for char in sample):
            return _ReferenceGlyphs(xheight, word, stroke)
    return None


def _stroke_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    """Estimate a font's stroke width in pixels.

    The glyph mask is eroded one pass at a time until almost no ink is left;
    each pass removes roughly two pixels of width.

    Args:
        font: An already opened font.
        text: Sample letters with representative vertical strokes.

    Returns:
        The estimated stroke width in pixels.
    """
    mask = Image.new("L", (int(font.getlength(text)) + 40, 200), 0)
    ImageDraw.Draw(mask).text((20, 150), text, font=font, fill=255, anchor="ls")
    mask = mask.point(lambda value: 255 if value > 128 else 0)
    total_ink = np.asarray(mask).sum()

    passes = 0
    while passes < _MAX_EROSION_PASSES:
        mask = mask.filter(ImageFilter.MinFilter(3))
        passes += 1
        if np.asarray(mask).sum() < total_ink * 0.05:
            break
    return 2.0 * passes


def _measure_font(
    path: Path, codepoints: frozenset[int]
) -> tuple[float, float, float]:
    """Measure a font's proportions relative to its nominal size.

    Args:
        path: Path to a ``.ttf`` or ``.otf`` file.
        codepoints: The code points the font covers, used to choose glyphs
            it can actually draw.

    Returns:
        ``(x-height ratio, average letter width ratio, stroke ratio)``.
        Conservative defaults are returned when the font cannot be measured.
    """
    reference = _reference_glyphs(codepoints)
    if reference is None:
        logger.warning(
            "Font %s covers no reference alphabet; using default metrics",
            path.name,
        )
        return _DEFAULT_METRICS

    try:
        font = ImageFont.truetype(str(path), _METRIC_SIZE)
        heights = [
            -font.getbbox(char, anchor="ls")[1]
            for char in reference.xheight_chars
        ]
        xheight = sorted(heights)[len(heights) // 2] / _METRIC_SIZE
        width = (
            font.getlength(reference.width_word)
            / len(reference.width_word)
            / _METRIC_SIZE
        )
        stroke = _stroke_width(font, reference.stroke_chars) / _METRIC_SIZE
    except OSError:
        logger.warning("Could not measure font %s; using defaults", path.name)
        return _DEFAULT_METRICS
    return (
        max(0.2, min(0.9, xheight)),
        max(0.2, min(0.9, width)),
        max(0.02, min(0.3, stroke)),
    )


@dataclass(frozen=True)
class FontInfo:
    """A handwriting font together with its coverage and proportions.

    Args:
        path: Location of the font file.
        name: Display name, taken from the file stem.
        codepoints: Unicode code points the font can draw directly.
        variable_axes: Variation axis tag mapped to ``(min, default, max)``.
        xheight_ratio: x-height divided by nominal point size.
        width_ratio: Average lowercase advance divided by nominal point size.
        stroke_ratio: Natural stroke width divided by nominal point size.
    """

    path: Path
    name: str
    codepoints: frozenset[int]
    variable_axes: dict[str, tuple[float, float, float]] = field(
        default_factory=dict
    )
    xheight_ratio: float = 0.45
    width_ratio: float = 0.46
    stroke_ratio: float = 0.08

    def has_glyph(self, char: str) -> bool:
        """Return whether the font can draw a character without a fallback."""
        return ord(char) in self.codepoints

    def can_render(self, text: str) -> bool:
        """Return whether every character is drawable, directly or via fallback.

        Args:
            text: The text to check.

        Returns:
            True when no character would be lost.
        """
        for char in text:
            if char.isspace() or self.has_glyph(char):
                continue
            base = FALLBACK_BASE.get(char) or SUBSTITUTE.get(char)
            if base is None or not self.has_glyph(base):
                return False
        return True

    @classmethod
    def from_path(cls, path: Path) -> FontInfo:
        """Read coverage, variation axes and proportions from a font file.

        Args:
            path: Path to a ``.ttf`` or ``.otf`` file.

        Returns:
            The parsed font description.

        Raises:
            ValueError: If the file cannot be parsed as a font.
        """
        try:
            ttf = TTFont(path)
            codepoints = frozenset(ttf.getBestCmap().keys())
            axes = {
                axis.axisTag: (axis.minValue, axis.defaultValue, axis.maxValue)
                for axis in (ttf["fvar"].axes if "fvar" in ttf else [])
            }
        except Exception as error:  # fontTools raises many unrelated types.
            raise ValueError(f"Unreadable font file: {path}") from error

        xheight, width, stroke = _measure_font(path, codepoints)
        return cls(
            path=path,
            name=path.stem,
            codepoints=codepoints,
            variable_axes=axes,
            xheight_ratio=xheight,
            width_ratio=width,
            stroke_ratio=stroke,
        )


def pixel_size_for(info: FontInfo, nominal_size: int) -> int:
    """Convert a layout size into the pixel size that equalises fonts.

    Args:
        info: The font being opened.
        nominal_size: Size used by the layout code, in the Caveat-like scale.

    Returns:
        The pixel size to open this font at so that it visually matches
        ``nominal_size``.
    """
    correction = math.sqrt(
        (TARGET_XHEIGHT / info.xheight_ratio)
        * (TARGET_WIDTH / info.width_ratio)
    )
    return max(8, round(nominal_size * correction))


def open_font(
    info: FontInfo, size: int, rng: random.Random
) -> ImageFont.FreeTypeFont:
    """Open a font at a pixel size, randomising the weight axis if present.

    Args:
        info: The font to open.
        size: Pixel size to open the font at.
        rng: Random source used to pick a weight on variable fonts.

    Returns:
        The opened Pillow font.
    """
    font = ImageFont.truetype(str(info.path), size)
    if "wght" not in info.variable_axes:
        return font

    low, _, high = info.variable_axes["wght"]
    try:
        values: list[float] = []
        for axis in font.get_variation_axes():
            raw_name = axis.get("name", b"")
            name = (
                raw_name.decode(errors="ignore")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            if "eight" in name or name == "wght":
                values.append(rng.uniform(low, low + (high - low) * 0.3))
            else:
                values.append(float(axis.get("default") or 0.0))
        font.set_variation_by_axes(values)
    except OSError:
        logger.debug("Font %s has no usable weight axis", info.name)
    return font


class FontLibrary:
    """The set of handwriting fonts available to a generator.

    Args:
        fonts: The fonts to expose. Must not be empty.

    Raises:
        ValueError: If ``fonts`` is empty.
    """

    def __init__(self, fonts: Sequence[FontInfo]) -> None:
        if not fonts:
            raise ValueError("A font library needs at least one font")
        self._fonts = tuple(fonts)

    def __iter__(self) -> Iterator[FontInfo]:
        return iter(self._fonts)

    def __len__(self) -> int:
        return len(self._fonts)

    @property
    def fonts(self) -> tuple[FontInfo, ...]:
        """Every font in the library, in a stable order."""
        return self._fonts

    @classmethod
    def from_directory(cls, fonts_dir: Path | str) -> FontLibrary:
        """Scan a directory for usable fonts.

        Results are memoised per directory because measuring a font renders
        and erodes glyph masks, which is slow.

        Args:
            fonts_dir: Directory containing ``.ttf`` / ``.otf`` files.

        Returns:
            A library holding every readable font in the directory.

        Raises:
            FileNotFoundError: If the directory holds no usable font.
        """
        return cls(_scan_directory(Path(fonts_dir).resolve()))

    def by_name(self, name: str) -> FontInfo | None:
        """Return the font with this name, or None when it is absent."""
        return next((font for font in self._fonts if font.name == name), None)

    def pick(
        self,
        text: str,
        rng: random.Random,
        font_path: Path | str | None = None,
    ) -> FontInfo:
        """Choose a font able to render ``text``.

        Args:
            text: Every character that will be drawn with this font.
            rng: Random source used when sampling from the library.
            font_path: Force a specific font file instead of sampling.

        Returns:
            A font whose coverage, including fallbacks, spans ``text``.

        Raises:
            ValueError: If the requested font, or every available font, cannot
                render ``text``.
        """
        if font_path is not None:
            info = FontInfo.from_path(Path(font_path))
            if not info.can_render(text):
                raise ValueError(
                    f"Font {info.name} cannot render this text even with "
                    "fallbacks"
                )
            return info

        usable = [font for font in self._fonts if font.can_render(text)]
        if not usable:
            raise ValueError("No available font can render the given text")
        return rng.choice(usable)


@lru_cache(maxsize=8)
def _scan_directory(fonts_dir: Path) -> tuple[FontInfo, ...]:
    """Read every font in a directory; memoised by FontLibrary.from_directory."""
    infos: list[FontInfo] = []
    for path in sorted(fonts_dir.glob("*.[to]tf")):
        try:
            infos.append(FontInfo.from_path(path))
        except ValueError:
            logger.warning("Skipping unreadable font %s", path.name)

    if not infos:
        raise FileNotFoundError(f"No usable fonts found in {fonts_dir}")
    return tuple(infos)
