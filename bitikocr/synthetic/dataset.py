"""Turn a generator plus a pool of field values into a dataset on disk.

Every sample is written as ``<stem>.png`` next to ``<stem>.json``. The JSON
is the full :class:`~bitikocr.models.annotation.DocumentAnnotation`, so an
image and its ground truth are always found by the same stem.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from bitikocr.models.annotation import DocumentAnnotation
from bitikocr.synthetic.generators.base import (
    DocumentGenerator,
    SyntheticDocument,
)

__all__ = [
    "DatasetSummary",
    "SampleFiles",
    "draw_annotations",
    "generate_dataset",
]

logger = logging.getLogger(__name__)

_MAX_SEED = 2**31

_BLOCK_OUTLINE = (30, 160, 30)
_LINE_OUTLINE = (220, 30, 30)


@dataclass(frozen=True)
class SampleFiles:
    """Where one generated sample was written.

    Args:
        image: Path to the rendered page.
        annotation: Path to the ground-truth JSON.
        preview: Path to the box overlay, when one was requested.
    """

    image: Path
    annotation: Path
    preview: Path | None = None


@dataclass(frozen=True)
class DatasetSummary:
    """What a dataset run produced.

    Args:
        document_type: The generated document type.
        output_dir: Directory the samples were written to.
        samples: One entry per generated sample, in generation order.
    """

    document_type: str
    output_dir: Path
    samples: tuple[SampleFiles, ...]

    def __len__(self) -> int:
        return len(self.samples)


def generate_dataset(
    generator: DocumentGenerator,
    field_sets: Sequence[dict[str, Any]],
    count: int,
    output_dir: Path,
    seed: int | None = None,
    draw_boxes: bool = False,
) -> DatasetSummary:
    """Generate ``count`` samples and write them to ``output_dir``.

    Field sets are drawn at random from ``field_sets``, so a small pool still
    produces a varied dataset: the handwriting, layout and paper differ on
    every page even when the text repeats.

    Args:
        generator: The document generator to render with.
        field_sets: Pool of field values to draw from. Must not be empty.
        count: How many samples to generate. Must be positive.
        output_dir: Directory to write into; created if it does not exist.
        seed: Makes the whole run reproducible, including which field set
            and which per-page seed each sample gets.
        draw_boxes: Also write a ``_boxes.png`` overlay per sample.

    Returns:
        A summary listing every file written.

    Raises:
        ValueError: If ``count`` is not positive or ``field_sets`` is empty.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if not field_sets:
        raise ValueError("At least one field set is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    written: list[SampleFiles] = []
    for index, document in enumerate(
        _render_samples(generator, field_sets, count, rng)
    ):
        stem = f"{generator.name}_{index:04d}_seed{document.seed}"
        written.append(write_sample(document, output_dir, stem, draw_boxes))
        logger.info(
            "Generated %s (font=%s, lines=%d)",
            stem,
            document.annotation.metadata.get("font"),
            len(document.annotation.lines),
        )

    return DatasetSummary(
        document_type=generator.name,
        output_dir=output_dir,
        samples=tuple(written),
    )


def write_sample(
    document: SyntheticDocument,
    output_dir: Path,
    stem: str,
    draw_boxes: bool = False,
) -> SampleFiles:
    """Write one sample's image and annotation to disk.

    Args:
        document: The generated page.
        output_dir: Directory to write into; must already exist.
        stem: File name without extension, shared by every written file.
        draw_boxes: Also write a ``_boxes.png`` overlay.

    Returns:
        The paths that were written.
    """
    image_path = output_dir / f"{stem}.png"
    annotation_path = output_dir / f"{stem}.json"

    document.image.save(image_path)
    annotation_path.write_text(
        json.dumps(document.annotation.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_path: Path | None = None
    if draw_boxes:
        preview_path = output_dir / f"{stem}_boxes.png"
        draw_annotations(document.image, document.annotation).save(preview_path)

    return SampleFiles(
        image=image_path, annotation=annotation_path, preview=preview_path
    )


def draw_annotations(
    image: Image.Image, annotation: DocumentAnnotation
) -> Image.Image:
    """Draw block and line boxes over a page, for visual inspection.

    Args:
        image: The rendered page.
        annotation: The ground truth to overlay.

    Returns:
        A new image; the input is left untouched.
    """
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for block in annotation.blocks:
        if block.bbox:
            draw.rectangle(
                block.bbox.to_list(), outline=_BLOCK_OUTLINE, width=2
            )
    for line in annotation.lines:
        if line.bbox:
            draw.rectangle(line.bbox.to_list(), outline=_LINE_OUTLINE, width=3)
    return preview


def _render_samples(
    generator: DocumentGenerator,
    field_sets: Sequence[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> Iterator[SyntheticDocument]:
    """Yield ``count`` rendered pages, skipping ones the fonts cannot draw."""
    for _ in range(count):
        fields = rng.choice(field_sets)
        page_seed = rng.randrange(_MAX_SEED)
        try:
            yield generator.generate(fields, seed=page_seed)
        except ValueError as error:
            logger.warning("Skipping sample with seed %d: %s", page_seed, error)
