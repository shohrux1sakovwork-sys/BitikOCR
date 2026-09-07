"""The contract every synthetic document generator implements.

A generator receives a mapping of field names to text and returns a rendered
page plus its ground truth. Keeping the interface uniform is what lets the
dataset builder and the CLI treat all document types the same way.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from bitikocr.config import SyntheticConfig
from bitikocr.models.annotation import DocumentAnnotation
from bitikocr.synthetic.fonts import FontInfo, FontLibrary
from bitikocr.synthetic.style import HandwritingStyle, sample_style

__all__ = [
    "DEFAULT_INK_STRENGTH",
    "DocumentGenerator",
    "FieldValues",
    "SyntheticDocument",
]

FieldValues = Mapping[str, Any]

_MAX_SEED = 2**31

#: How heavily a hand writes by default. The shipped fonts have thin
#: strokes, and a page that is then aged and re-compressed loses more,
#: so the pen is calibrated a little above what the font asks for.
DEFAULT_INK_STRENGTH = 1.4


@dataclass(frozen=True)
class SyntheticDocument:
    """One generated page and everything known about it.

    Args:
        image: The rendered page as an RGB image.
        annotation: Transcription, blocks, lines and generator metadata.
    """

    image: Image.Image
    annotation: DocumentAnnotation

    @property
    def seed(self) -> int | None:
        """The seed this page was generated from, if the generator set one."""
        return self.annotation.metadata.get("seed")


class DocumentGenerator(ABC):
    """Render one kind of Uzbek document as synthetic handwriting.

    Args:
        config: Paths to the fonts and background templates to use.
        font_path: Force every page to use this handwriting font instead of
            sampling one. Useful for per-font visual comparison.
        ink_strength: How heavily the pen writes; see
            :attr:`~bitikocr.synthetic.style.HandwritingStyle.ink_strength`.

    Raises:
        FileNotFoundError: If the configured fonts directory holds no font.
    """

    #: Name used on the command line and in the generator registry.
    name: ClassVar[str]

    def __init__(
        self,
        config: SyntheticConfig,
        font_path: Path | str | None = None,
        ink_strength: float = DEFAULT_INK_STRENGTH,
    ) -> None:
        self.config = config
        self.font_path = Path(font_path) if font_path else None
        self.ink_strength = ink_strength
        self.library = FontLibrary.from_directory(config.fonts_dir)

    @classmethod
    def with_template(
        cls,
        config: SyntheticConfig,
        font_path: Path | str | None,
        template: str,
        ink_strength: float = DEFAULT_INK_STRENGTH,
    ) -> DocumentGenerator:
        """Build this generator for a named form template.

        Only generators that fill a measured printed form support this;
        the rest lay their content out from the page size alone.

        Args:
            config: Paths to the fonts, backgrounds and layouts to use.
            font_path: Force a specific handwriting font.
            template: Name of the layout to fill.
            ink_strength: How heavily the pen writes.

        Returns:
            A generator bound to that template.

        Raises:
            ValueError: If this document type is not template-driven.
        """
        raise ValueError(
            f"Document type {cls.name!r} does not use form templates"
        )

    @property
    @abstractmethod
    def field_names(self) -> tuple[str, ...]:
        """Field names this generator understands.

        This is a property rather than a constant because a form-filling
        generator takes its fields from whichever template it was built with.
        """

    @property
    @abstractmethod
    def reading_order(self) -> tuple[str, ...]:
        """Block names in the order a human reads the finished page."""

    @abstractmethod
    def generate(
        self,
        fields: FieldValues,
        seed: int | None = None,
        style_overrides: Mapping[str, Any] | None = None,
    ) -> SyntheticDocument:
        """Render one page.

        Args:
            fields: Field name to text. Unknown names are ignored; missing
                ones are left blank.
            seed: Makes the page reproducible. A random seed is drawn and
                recorded in the annotation when omitted.
            style_overrides: Style fields to pin instead of sampling them.

        Returns:
            The rendered page and its ground truth.

        Raises:
            ValueError: If no available font can render the given text, or if
                ``style_overrides`` names an unknown style field.
        """

    # -- shared helpers ----------------------------------------------------

    @staticmethod
    def _seeded_rng(seed: int | None) -> tuple[int, random.Random]:
        """Return the effective seed and the random source built from it."""
        effective = random.randrange(_MAX_SEED) if seed is None else seed
        return effective, random.Random(effective)

    def _sample_style(
        self,
        rng: random.Random,
        text: str,
        style_overrides: Mapping[str, Any] | None,
    ) -> tuple[FontInfo, HandwritingStyle]:
        """Pick the main hand's font and sample the page style around it.

        Args:
            rng: Random source for the whole page.
            text: Every character the main hand will draw, used to reject
                fonts with insufficient coverage.
            style_overrides: Style fields to pin after sampling.

        Returns:
            ``(font, style)`` with ``style.font`` already set to the font's
            name.
        """
        info = self.library.pick(text, rng, self.font_path)
        style = sample_style(rng, self.library).replace(
            font=info.name, ink_strength=self.ink_strength
        )
        if style_overrides:
            style = style.replace(**style_overrides)
        return info, style

    def _second_hand_font(self, style: HandwritingStyle, text: str) -> FontInfo:
        """Resolve the second hand's font, falling back to the main hand's.

        Args:
            style: The page style naming both fonts.
            text: Everything the second hand will draw.

        Returns:
            The second font when it covers ``text``, otherwise the main font.
        """
        main = self.library.by_name(style.font) or self.library.fonts[0]
        second = self.library.by_name(style.second_font)
        if second is None or (text and not second.can_render(text)):
            return main
        return second

    def _collect_text(self, fields: FieldValues) -> str:
        """Join every known field's text, for font coverage checks."""
        parts = [
            str(fields[name]) for name in self.field_names if fields.get(name)
        ]
        return " ".join(parts)
