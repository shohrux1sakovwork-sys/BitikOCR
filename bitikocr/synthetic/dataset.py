"""Turning records into a dataset on disk.

Generation is two steps, and they are deliberately separable:

1. **Sample facts** — what each document will say. They are written one JSON
   per document, and can be read, edited or replaced before anything is
   drawn.
2. **Render them** — draw each page, spoil it like a scan, and write it
   beside its annotation.

Keeping them apart means a batch can be re-rendered with different fonts or
augmentation without resampling the text, and the text can be reviewed
before the expensive step runs.

Layout on disk::

    <output>/
      facts/<id>.json        the structured values the page carries
      images/<id>.png        the rendered page
      annotations/<id>.json  the transcription record: text, parts, capture
      previews/<id>.png      box overlays, only with draw_boxes
      index.jsonl            one line per page, tying the three together

Both JSON records follow the corpus schema in
:mod:`bitikocr.models.schema`, and every file for one document shares its
id, so a fine-tuning pipeline can pair them without consulting the index.
The index is there to iterate the set in order and to filter it by era,
script, font or document type.

The facts written before a render are the generator's own record — what a
page will be told to say. Rendering replaces them with the schema's facts
record, which is what a reader would take off the finished page.
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
from bitikocr.synthetic.augment import (
    AugmentationProfile,
    AugmentationReport,
    augment_page,
)
from bitikocr.synthetic.export import (
    build_facts_record,
    build_transcription_record,
    document_id,
)
from bitikocr.synthetic.generators.base import (
    DocumentGenerator,
    SyntheticDocument,
)
from bitikocr.synthetic.records import DocumentRecord

__all__ = [
    "DEFAULT_ID_PREFIX",
    "INDEX_NAME",
    "DatasetLayout",
    "DatasetSummary",
    "SampleFiles",
    "draw_annotations",
    "iter_index",
    "read_records",
    "render_records",
    "write_records",
]

logger = logging.getLogger(__name__)

_BLOCK_OUTLINE = (30, 160, 30)
_LINE_OUTLINE = (220, 30, 30)

INDEX_NAME = "index.jsonl"

#: What documents are called when the caller does not say. Namespace it per
#: document type if several sets will be merged into one corpus.
DEFAULT_ID_PREFIX = "doc"


@dataclass(frozen=True)
class DatasetLayout:
    """Where the pieces of one dataset live.

    Args:
        root: The dataset's directory.
    """

    root: Path

    @property
    def facts(self) -> Path:
        """What each document says, one JSON per document."""
        return self.root / "facts"

    @property
    def images(self) -> Path:
        """The rendered pages."""
        return self.root / "images"

    @property
    def annotations(self) -> Path:
        """The ground truth, one JSON per page."""
        return self.root / "annotations"

    @property
    def previews(self) -> Path:
        """Box overlays, for looking at rather than training on."""
        return self.root / "previews"

    @property
    def index(self) -> Path:
        """One line per rendered page, for a data loader to read."""
        return self.root / INDEX_NAME

    def create(self, with_previews: bool = False) -> None:
        """Create the directories a render will write into."""
        self.facts.mkdir(parents=True, exist_ok=True)
        self.images.mkdir(parents=True, exist_ok=True)
        self.annotations.mkdir(parents=True, exist_ok=True)
        if with_previews:
            self.previews.mkdir(parents=True, exist_ok=True)

    def document_id(self, position: int, prefix: str) -> str:
        """Return the id every file for one document shares."""
        return document_id(prefix, position)


@dataclass(frozen=True)
class SampleFiles:
    """Where one sample's files were written.

    Args:
        id: The identifier every file for this sample shares.
        facts: Path to the structured values the page carries.
        image: Path to the rendered page.
        annotation: Path to the transcription record.
        preview: Path to the box overlay, when one was requested.
    """

    id: str
    facts: Path
    image: Path
    annotation: Path
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


# -- facts on disk ---------------------------------------------------------


def write_records(
    records: Sequence[DocumentRecord],
    layout: DatasetLayout,
    prefix: str = DEFAULT_ID_PREFIX,
) -> list[Path]:
    """Write one file per record, before anything has been drawn.

    These hold the generator's own record — what a page will be told to say
    — so they can be read and corrected. Rendering replaces each with the
    schema's facts record for the finished page.

    Args:
        records: The records to write, in order.
        layout: The dataset to write them into.
        prefix: What to call documents in this dataset.

    Returns:
        The paths written, in the same order.
    """
    layout.facts.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for position, record in enumerate(records):
        path = layout.facts / f"{layout.document_id(position, prefix)}.json"
        path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(path)
    return written


def read_records(layout: DatasetLayout) -> list[DocumentRecord]:
    """Read every facts file in a dataset, in file-name order.

    Args:
        layout: The dataset to read from.

    Returns:
        The records.

    Raises:
        FileNotFoundError: If the dataset has no facts directory.
        ValueError: If a file is not a well-formed record.
    """
    if not layout.facts.is_dir():
        raise FileNotFoundError(f"No facts directory in {layout.root}")

    records: list[DocumentRecord] = []
    for path in sorted(layout.facts.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} is not valid JSON") from error
        if "facts" in payload and "fields" not in payload:
            raise ValueError(
                f"{path} is a rendered facts record, not a generator record; "
                "re-sample the batch to render it again"
            )
        records.append(DocumentRecord.from_dict(payload))
    return records


# -- rendering -------------------------------------------------------------


def render_records(
    generator: DocumentGenerator,
    records: Sequence[DocumentRecord],
    output_dir: Path,
    augmentation: AugmentationProfile | None = None,
    draw_boxes: bool = False,
    prefix: str = DEFAULT_ID_PREFIX,
    collection: str | None = None,
) -> DatasetSummary:
    """Render records to pages, transcriptions, facts and an index.

    Args:
        generator: The generator matching the records' document type.
        records: The records to render.
        output_dir: Dataset directory to write into.
        augmentation: How hard to spoil each page; no augmentation when
            omitted.
        draw_boxes: Also write a box overlay per page.
        prefix: What to call documents in this dataset.
        collection: The batch name recorded on every page; the dataset
            directory's name when omitted.

    Returns:
        A summary listing every file written and every record skipped.

    Raises:
        ValueError: If ``records`` is empty.
    """
    if not records:
        raise ValueError("At least one record is required")

    layout = DatasetLayout(output_dir)
    layout.create(with_previews=draw_boxes)
    collection = collection or output_dir.name

    written: list[SampleFiles] = []
    skipped: list[tuple[int, str]] = []
    index_lines: list[str] = []

    for position, record in enumerate(records):
        try:
            document = generator.generate(record.fields, seed=record.seed)
        except ValueError as error:
            logger.warning(
                "Skipping record %d (seed %d): %s",
                position,
                record.seed,
                error,
            )
            skipped.append((position, str(error)))
            continue

        image, annotation, quality = _finish(document, record, augmentation)
        identifier = layout.document_id(position, prefix)
        files = _write_sample(
            layout=layout,
            identifier=identifier,
            record=record,
            generator=generator,
            image=image,
            annotation=annotation,
            quality=quality,
            collection=collection,
            draw_boxes=draw_boxes,
        )
        written.append(files)
        index_lines.append(
            json.dumps(
                _index_entry(record, files, annotation, layout),
                ensure_ascii=False,
            )
        )
        logger.info(
            "Rendered %s (font=%s, script=%s, era=%s, lines=%d)",
            identifier,
            annotation.metadata.get("font"),
            record.script,
            record.era,
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
) -> tuple[Image.Image, DocumentAnnotation, AugmentationReport]:
    """Spoil a rendered page and note how it ended up being captured."""
    image, annotation = document.image, document.annotation
    quality = AugmentationReport()
    if augmentation is not None:
        image, annotation, quality = augment_page(
            image, annotation, random.Random(record.seed), augmentation
        )
    annotation.metadata.setdefault("script", record.script)
    return image, annotation, quality


def _write_sample(
    layout: DatasetLayout,
    identifier: str,
    record: DocumentRecord,
    generator: DocumentGenerator,
    image: Image.Image,
    annotation: DocumentAnnotation,
    quality: AugmentationReport,
    collection: str,
    draw_boxes: bool,
) -> SampleFiles:
    """Write one page and the two schema records describing it."""
    facts_path = layout.facts / f"{identifier}.json"
    image_path = layout.images / f"{identifier}.png"
    annotation_path = layout.annotations / f"{identifier}.json"

    image.save(image_path)
    relative_image = image_path.relative_to(layout.root).as_posix()

    transcription = build_transcription_record(
        record=record,
        annotation=annotation,
        generator=generator,
        document_id=identifier,
        image_path=relative_image,
        collection=collection,
        quality=quality,
    )
    annotation_path.write_text(
        json.dumps(transcription.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    facts = build_facts_record(record, identifier, relative_image)
    facts_path.write_text(
        json.dumps(facts.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_path: Path | None = None
    if draw_boxes:
        preview_path = layout.previews / f"{identifier}.png"
        draw_annotations(image, annotation).save(preview_path)

    return SampleFiles(
        id=identifier,
        facts=facts_path,
        image=image_path,
        annotation=annotation_path,
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
        "id": files.id,
        "image": files.image.relative_to(layout.root).as_posix(),
        "annotation": files.annotation.relative_to(layout.root).as_posix(),
        "facts": files.facts.relative_to(layout.root).as_posix(),
        "document_type": record.document_type,
        "script": record.script,
        "era": record.era,
        "year_approx": record.year,
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
