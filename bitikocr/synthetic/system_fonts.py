"""Printed (non-handwriting) fonts used for stamps and typewritten numbers.

Pre-printed marks on a form are not handwriting, so they must not come from
the handwriting library. Pillow resolves bare file names against the system
font directories, which is why the candidate lists below mix Windows and
Linux names.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import ImageFont

from bitikocr.synthetic.fonts import FontLibrary

__all__ = ["PrintFont", "find_monospace_font", "find_print_font"]

#: Either flavour of Pillow font; both expose getlength() and work with text().
PrintFont = ImageFont.FreeTypeFont | ImageFont.ImageFont

logger = logging.getLogger(__name__)

_SANS_CANDIDATES = (
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf",
    "arialbd.ttf",
    "seguisb.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

_MONO_CANDIDATES = (
    "DejaVuSansMono.ttf",
    "LiberationMono-Regular.ttf",
    "cour.ttf",
    "consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)

# Handwriting fonts to fall back on, least cursive first.
_LEAST_CURSIVE = ("Neucha", "ShantellSans", "AmaticSC", "Caveat")


def find_print_font(library: FontLibrary) -> Path:
    """Find a plain sans font able to render stamp lettering.

    Args:
        library: Handwriting fonts to fall back on when the system has no
            suitable print font installed.

    Returns:
        A path to a usable font file.
    """
    for candidate in _SANS_CANDIDATES:
        path = _resolve(candidate)
        if path is not None:
            return path

    for prefix in _LEAST_CURSIVE:
        for font in library:
            if font.name.startswith(prefix):
                logger.debug("No print font installed; using %s", font.name)
                return font.path
    return library.fonts[0].path


def find_monospace_font(size: int) -> PrintFont:
    """Open a typewriter-like font for printed serial numbers.

    Args:
        size: Pixel size to open the font at.

    Returns:
        The opened font, or Pillow's bundled default if none is installed.
    """
    for candidate in _MONO_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue

    logger.warning("No monospace font found; falling back to Pillow's default")
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1 takes no size argument.
        return ImageFont.load_default()


def _resolve(candidate: str) -> Path | None:
    """Return the path Pillow would open for a font name, or None."""
    try:
        font = ImageFont.truetype(candidate, 12)
    except OSError:
        return None
    path = getattr(font, "path", None)
    return Path(path) if path else None
