"""The "writer style" that drives every randomised aspect of a page.

One :class:`HandwritingStyle` describes a single imaginary writer and the
sheet of paper they used: their pen, ink, slant, spacing habits and how
steadily they hold a baseline. Every knob the renderer reads lives here, so
callers can reproduce or override any of them.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Any, Literal, cast

from bitikocr.data.synthetic.fonts import FontInfo, FontLibrary

__all__ = [
    "INK_PALETTE",
    "PAPER_PALETTE",
    "PENCIL_COLOR",
    "Color",
    "HandwritingStyle",
    "PenKind",
    "sample_style",
]

Color = tuple[int, int, int]
PenKind = Literal["hard", "soft", "gel"]

INK_PALETTE: tuple[Color, ...] = (
    (24, 38, 150),  # Classic blue ballpoint.
    (35, 60, 175),  # Lighter blue.
    (60, 40, 150),  # Violet.
    (75, 30, 130),  # Deep violet.
    (20, 25, 70),  # Blue-black.
    (15, 15, 25),  # Black.
)

PAPER_PALETTE: tuple[Color, ...] = (
    (252, 252, 250),
    (247, 245, 238),
    (243, 238, 224),
    (235, 226, 205),
    (225, 208, 178),
)

PENCIL_COLOR: Color = (105, 105, 110)

# Target ink line width in pixels at 200 dpi, per pen kind. A real ballpoint
# lays down roughly 2-4 px at that resolution.
_STROKE_RANGE: dict[PenKind, tuple[float, float]] = {
    "hard": (2.0, 3.2),
    "gel": (2.8, 4.0),
    "soft": (3.2, 4.8),
}


@dataclass(frozen=True)
class HandwritingStyle:
    """Every randomised parameter of one page, in one immutable record.

    Args:
        font: Name of the font the main hand writes with.
        second_font: Name of the font used by a second hand, e.g. a clerk
            adding registration marks.
        font_size: Nominal size the layout is measured in, in pixels.
        pen: Which pen the writer used.
        stroke_px: Target ink line width in pixels.
        ink: Ink colour as RGB.
        paper: Paper colour as RGB, used when no background scan is given.
        slant: Shear factor; positive leans right.
        letter_spacing: Extra gap between letters, times ``font_size``.
        word_spacing: Multiplier on the font's own space advance.
        line_spacing: Baseline step, times ``font_size``.
        line_spacing_jitter: Relative random variation of ``line_spacing``.
        line_slope: Per-page tendency of lines to run uphill, in degrees.
        line_slope_jitter: Random variation of ``line_slope`` per line.
        baseline_wobble: Baseline wave amplitude, times ``font_size``.
        char_scale_jitter: Relative random variation of each glyph's size.
        char_rot_jitter: Random rotation of each glyph, in degrees.
        char_y_jitter: Random vertical offset per glyph, times ``font_size``.
        ink_variation: How unevenly the ink is laid down; 0 is perfectly even.
        ink_strength: Calibration for how heavily the pen writes. 1.0 draws
            the stroke the font and pen ask for; above that widens and
            darkens it, and steadies the fading. Fonts differ enough in
            stroke weight that the thinnest need help to stay legible once a
            page has been aged.
        indent: First-line indent of the body, times ``font_size``.
        left_margin: Left text edge, as a fraction of the page width.
        right_margin: Right text edge, as a fraction of the page width.
        header_x: Where the addressee block starts, fraction of page width.
        top_margin: Top text edge, as a fraction of the page height.
        title_x: Title centre, as a fraction of the page width.
        title_scale: Title size, times ``font_size``.
        header_scale: Header size, times ``font_size``.
        signature_scribble: Whether the writer signs with a scribble.
    """

    font: str
    second_font: str
    font_size: int
    pen: PenKind
    stroke_px: float
    ink: Color
    paper: Color
    slant: float
    letter_spacing: float
    word_spacing: float
    line_spacing: float
    line_spacing_jitter: float
    line_slope: float
    line_slope_jitter: float
    baseline_wobble: float
    char_scale_jitter: float
    char_rot_jitter: float
    char_y_jitter: float
    ink_variation: float
    ink_strength: float
    indent: float
    left_margin: float
    right_margin: float
    header_x: float
    top_margin: float
    title_x: float
    title_scale: float
    header_scale: float
    signature_scribble: bool

    def replace(self, **changes: Any) -> HandwritingStyle:
        """Return a copy of this style with some fields changed.

        Args:
            **changes: Field names mapped to their new values.

        Returns:
            A new style; this one is left untouched.

        Raises:
            ValueError: If a name does not match any style field.
        """
        unknown = set(changes) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ValueError(
                f"Unknown style fields: {', '.join(sorted(unknown))}"
            )
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the style."""
        payload = dataclasses.asdict(self)
        payload["ink"] = list(self.ink)
        payload["paper"] = list(self.paper)
        return payload


def _jitter_color(color: Color, rng: random.Random, spread: int = 15) -> Color:
    """Nudge each channel of a colour to make two pages never match exactly."""
    return (
        max(0, min(255, color[0] + rng.randint(-spread, spread))),
        max(0, min(255, color[1] + rng.randint(-spread, spread))),
        max(0, min(255, color[2] + rng.randint(-spread, spread))),
    )


def _pick_second_font(
    library: FontLibrary, main: FontInfo, rng: random.Random
) -> FontInfo:
    """Pick a font for the second hand, preferring one that is not the main."""
    others = [font for font in library if font is not main]
    return rng.choice(others or list(library))


def sample_style(rng: random.Random, library: FontLibrary) -> HandwritingStyle:
    """Sample one writer and one sheet of paper.

    Args:
        rng: Random source. The same seed always yields the same style.
        library: Fonts the writer may choose from.

    Returns:
        A fully populated style.
    """
    pen = cast(
        PenKind,
        rng.choices(["hard", "soft", "gel"], weights=[0.6, 0.15, 0.25])[0],
    )
    main_font = rng.choice(list(library))

    return HandwritingStyle(
        font=main_font.name,
        second_font=_pick_second_font(library, main_font, rng).name,
        font_size=rng.randint(62, 90),
        pen=pen,
        stroke_px=rng.uniform(*_STROKE_RANGE[pen]),
        ink=_jitter_color(rng.choice(INK_PALETTE), rng),
        paper=rng.choice(PAPER_PALETTE),
        slant=rng.uniform(-0.05, 0.45),
        letter_spacing=rng.uniform(-0.09, 0.10),
        word_spacing=rng.uniform(0.9, 1.6),
        line_spacing=rng.uniform(1.35, 1.85),
        line_spacing_jitter=rng.uniform(0.0, 0.12),
        line_slope=rng.uniform(-1.4, 1.4),
        line_slope_jitter=rng.uniform(0.0, 0.8),
        baseline_wobble=rng.uniform(0.0, 0.06),
        char_scale_jitter=rng.uniform(0.0, 0.07),
        char_rot_jitter=rng.uniform(0.0, 2.5),
        char_y_jitter=rng.uniform(0.0, 0.05),
        ink_variation=rng.uniform(0.05, 0.25),
        ink_strength=1.0,
        indent=rng.uniform(0.8, 3.0),
        left_margin=rng.uniform(0.08, 0.15),
        right_margin=rng.uniform(0.86, 0.96),
        header_x=rng.uniform(0.50, 0.62),
        top_margin=rng.uniform(0.025, 0.08),
        title_x=rng.uniform(0.40, 0.56),
        title_scale=rng.uniform(1.1, 1.4),
        header_scale=rng.uniform(0.9, 1.05),
        signature_scribble=rng.random() < 0.9,
    )
