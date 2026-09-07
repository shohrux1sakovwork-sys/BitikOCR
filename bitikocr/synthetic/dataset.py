"""Turning records into a dataset on disk.

Generation is two steps, and they are deliberately separable:

1. **Sample records** — the field values a batch of documents will carry.
   They are written to ``metadata.jsonl``, one JSON object per line, and can
   be read, edited or replaced before anything is rendered.
2. **Render records** — draw each one, spoil it like a scan, and write the
   page beside its ground truth.

Keeping them apart means a batch can be re-rendered with different fonts or
augmentation without resampling the text, and the text can be reviewed
before the expensive step runs.

Layout on disk::

    <output>/
      metadata.jsonl        the records, one per line
      index.jsonl           one line per rendered page, for a data loader
      images/<stem>.png
      labels/<stem>.json    the DocumentAnnotation
      previews/<stem>.png   box overlays, only with draw_boxes
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from bitikocr.models.annotation import DocumentAnnotation
from bitikocr.synthetic.augment import AugmentationProfile, augment_page
from bitikocr.synthetic.generators.base import (
    DocumentGenerator,
    SyntheticDocument,
)
from bitikocr.synthetic.records import DocumentRecord

__all__ = [
    "DatasetLayout",
    "DatasetSummary",
    "SampleFiles",
    "draw_annotations",
    "read_records",
    "render_records",
    "write_records",
]

logger = logging.getLogger(__name__)

_BLOCK_OUTLINE = (30, 160, 30)
_LINE_OUTLINE = (220, 30, 30)

METADATA_NAME = "metadata.jsonl"
INDEX_NAME = "index.jsonl"


@dataclass(frozen=True)
class DatasetLayout:
    """Where the pieces of one dataset live.

    Args:
        root: The dataset's directory.
    """

    root: Path

    @property
    def metadata(self) -> Path:
        """The sampled records, one JSON object per line."""
        return self.root / METADATA_NAME

    @property
    def index(self) -> Path:
        """One line per rendered page, for a data loader to read."""
        return self.root / INDEX_NAME

    @property
    def images(self) -> Path:
        """The rendered pages."""
        return self.root / "images"

    @property
    def labels(self) -> Path:
        """The ground truth, one JSON per page."""
        return self.root / "labels"

    @property
    def previews(self) -> Path:
        """Box overlays, for looking at rather than training on."""
        return self.root / "previews"

    def create(self, with_previews: bool = False) -> None:
        """Create the directories a render will write into."""
        self.images.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)
        if with_previews:
            self.previews.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SampleFiles:
    """Where one rendered sample was written.

    Args:
        stem: The name every file for this sample shares.
        image: Path to the rendered page.
        label: Path to the ground-truth JSON.
        preview: Path to the box overlay, when one was requested.
    """

    stem: str
    image: Path
    label: Path
    preview: Path | None = None


@dataclass(frozen=True)
class DatasetSummary:
    """What a render produced.

    Args:
        document_type: The generated document type.
        layout: Where everything was written.
        samples: One entry per rendered page, in render order.
        skipped: Records that could not be rendered, with the reason.
    """

    document_type: str
    layout: DatasetLayout
    samples: tuple[SampleFiles, ...]
    skipped: tuple[tuple[int, str], ...] = ()

    def __len__(self) -> int:
        return len(self.samples)


# -- records on disk -------------------------------------------------------


def write_records(records: Iterable[DocumentRecord], path: Path) -> Path:
    """Write records to a JSON Lines file.

    Args:
        records: The records to write.
        path: File to write to; parent directories are created.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
            )
    return path


def read_records(path: Path) -> list[DocumentRecord]:
    """Read records back from a JSON Lines file.

    Args:
        path: The metadata file to read.

    Returns:
        The records, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is not a well-formed record.
    """
    records: list[DocumentRecord] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{number} is not valid JSON"
                ) from error
            records.append(DocumentRecord.from_dict(payload))
    return records


# -- rendering -------------------------------------------------------------


def render_records(
    generator: DocumentGenerator,
    records: Sequence[DocumentRecord],
    output_dir: Path,
    augmentation: AugmentationProfile | None = None,
    draw_boxes: bool = False,
) -> DatasetSummary:
    """Render records to images and ground truth.

    Args:
        generator: The generator matching the records' document type.
        records: The records to render.
        output_dir: Dataset directory to write into.
        augmentation: How hard to spoil each page; no augmentation when
            omitted.
        draw_boxes: Also write a box overlay per page.

    Returns:
        A summary listing every file written and every record skipped.

    Raises:
        ValueError: If ``records`` is empty.
    """
    if not records:
        raise ValueError("At least one record is required")

    layout = DatasetLayout(output_dir)
    layout.create(with_previews=draw_boxes)

    written: list[SampleFiles] = []
    skipped: list[tuple[int, str]] = []
    index_lines: list[str] = []

    for position, record in enumerate(records):
        try:
            document = generator.generate(record.fields, seed=record.seed)
        except ValueError as error:
            logger.warning(
                "Skipping record %d (seed %d): %s", position, record.seed, error
            )
            skipped.append((position, str(error)))
            continue

        image, annotation = _finish(document, record, augmentation)
        stem = f"{record.document_type}_{position:05d}_seed{record.seed}"
        files = _write_sample(layout, stem, image, annotation, draw_boxes)
        written.append(files)
        index_lines.append(
            json.dumps(
                _index_entry(record, files, annotation, layout),
                ensure_ascii=False,
            )
        )
        logger.info(
            "Rendered %s (font=%s, script=%s, lines=%d)",
            stem,
            annotation.metadata.get("font"),
            record.script,
            len(annotation.lines),
        )

    layout.index.write_text(
        "\n".join(index_lines) + ("\n" if index_lines else ""),
        encoding="utf-8",
    )
    return DatasetSummary(
        document_type=generator.name,
        layout=layout,
        samples=tuple(written),
        skipped=tuple(skipped),
    )


def _finish(
    document: SyntheticDocument,
    record: DocumentRecord,
    augmentation: AugmentationProfile | None,
) -> tuple[Image.Image, DocumentAnnotation]:
    """Spoil a rendered page and record what the record contributed."""
    image, annotation = document.image, document.annotation
    if augmentation is not None:
        image, annotation = augment_page(
            image, annotation, random.Random(record.seed), augmentation
        )
    annotation.metadata.setdefault("script", record.script)
    return image, annotation


def _write_sample(
    layout: DatasetLayout,
    stem: str,
    image: Image.Image,
    annotation: DocumentAnnotation,
    draw_boxes: bool,
) -> SampleFiles:
    """Write one page, its ground truth and optionally its box overlay."""
    image_path = layout.images / f"{stem}.png"
    label_path = layout.labels / f"{stem}.json"

    image.save(image_path)
    label_path.write_text(
        json.dumps(annotation.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_path: Path | None = None
    if draw_boxes:
        preview_path = layout.previews / f"{stem}.png"
        draw_annotations(image, annotation).save(preview_path)

    return SampleFiles(
        stem=stem,
        image=image_path,
        label=label_path,
        preview=preview_path,
    )


def _index_entry(
    record: DocumentRecord,
    files: SampleFiles,
    annotation: DocumentAnnotation,
    layout: DatasetLayout,
) -> dict[str, Any]:
    """Build one line of the data loader's index."""
    return {
        "stem": files.stem,
        "image": files.image.relative_to(layout.root).as_posix(),
        "label": files.label.relative_to(layout.root).as_posix(),
        "document_type": record.document_type,
        "script": record.script,
        "seed": record.seed,
        "font": annotation.metadata.get("font"),
        "template": annotation.metadata.get("template"),
        "size": list(annotation.size),
        "line_count": len(annotation.lines),
        "text": annotation.text,
    }


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


def iter_index(path: Path) -> Iterator[dict[str, Any]]:
    """Read a dataset index back, one entry at a time.

    Args:
        path: The ``index.jsonl`` to read.

    Yields:
        One decoded entry per rendered page.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
