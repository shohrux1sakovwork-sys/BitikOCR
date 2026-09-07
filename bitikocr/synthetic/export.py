"""Describing a generated page in the corpus schema.

The generator works in its own terms — blocks, lines, styles, seeds — and
the corpus is assembled in another: origin, era, text mode, capture quality,
parts, facts. This module is the one place that maps between them, so the
generator stays free to change and the schema stays a stable contract.

Everything here is derived, never guessed. Whether a page has a stamp is
read from what was drawn on it; how blurred it is comes from the
augmentation's own report; which era it belongs to comes from the year its
record is dated.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from bitikocr import __version__
from bitikocr.models.annotation import DocumentAnnotation
from bitikocr.models.schema import (
    AnnotationInfo,
    AnnotationStatus,
    DocumentMetadata,
    Era,
    Fact,
    FactsRecord,
    NoiseLevel,
    Part,
    QualityInfo,
    SourceInfo,
    TextMode,
    TranscriptionRecord,
)
from bitikocr.synthetic.augment import AugmentationReport
from bitikocr.synthetic.facts import build_facts
from bitikocr.synthetic.generators.base import DocumentGenerator
from bitikocr.synthetic.records import DocumentRecord

__all__ = ["build_facts_record", "build_transcription_record"]

#: Blocks that are marks on the page rather than regions of its text.
_MARK_BLOCKS = frozenset({"stamp", "signature"})

#: What a synthetic page's transcription is worth: it was not read off the
#: page, it was written onto it, so it is exact by construction.
_SYNTHETIC_ANNOTATION: AnnotationStatus = "gold"


def build_transcription_record(
    record: DocumentRecord,
    annotation: DocumentAnnotation,
    generator: DocumentGenerator,
    document_id: str,
    image_path: str,
    collection: str,
    quality: AugmentationReport | None = None,
) -> TranscriptionRecord:
    """Describe one generated page in the corpus schema.

    Args:
        record: What the page was told to say.
        annotation: What was actually drawn, with boxes.
        generator: The generator that drew it, for its page traits.
        document_id: The identifier this page shares with its facts record.
        image_path: Path to the page, relative to the corpus root.
        collection: The batch this page belongs to.
        quality: What the augmentation did, when a page was aged.

    Returns:
        The transcription record.
    """
    quality = quality or AugmentationReport()
    blocks = {block.kind for block in annotation.blocks if block.bbox}

    return TranscriptionRecord(
        id=document_id,
        image=image_path,
        source=SourceInfo(
            origin="synthetic",
            collection=collection,
            era=cast(Era, record.era),
            year_approx=record.year,
            seed=record.seed,
            generator=f"bitikocr@{__version__}",
            font=annotation.metadata.get("font"),
        ),
        metadata=DocumentMetadata(
            language=["uz"],
            scripts=_scripts(record, generator),
            primary_script=record.script,
            document_type=record.document_type,
            text_mode=cast(TextMode, generator.text_mode),
            layout=generator.layout,
            quality=QualityInfo(
                blur=quality.blur,
                rotation=round(quality.rotation, 3),
                skew=quality.skew,
                noise=cast(NoiseLevel, quality.noise),
                capture="scanner",
            ),
            has_handwriting=True,
            has_printed_text=generator.has_printed_text,
            has_stamp="stamp" in blocks,
            has_signature="signature" in blocks,
        ),
        text=annotation.text,
        parts=_parts(annotation),
        annotation=AnnotationInfo(
            status=_SYNTHETIC_ANNOTATION,
            pre_annotator=f"generator:bitikocr@{__version__}",
            revision=1,
        ),
    )


def build_facts_record(
    record: DocumentRecord, document_id: str, image_path: str
) -> FactsRecord:
    """Read the structured facts off one generated page.

    Args:
        record: What the page says.
        document_id: The identifier this record shares with the
            transcription.
        image_path: Path to the page, relative to the corpus root.

    Returns:
        The facts record.
    """
    facts: tuple[Fact, ...] = tuple(
        build_facts(record.document_type, record.fields, record.dates)
    )
    return FactsRecord(id=document_id, image=image_path, facts=facts)


def _scripts(record: DocumentRecord, generator: DocumentGenerator) -> list[str]:
    """Return every alphabet on the page, written and printed.

    A bilingual form carries both alphabets in its own printing, whichever
    one the clerk filled it in with.
    """
    seen: list[str] = [record.script]
    template = getattr(generator, "template", None)
    for script in getattr(template, "printed_scripts", ()):
        if script not in seen:
            seen.append(script)
    return seen


def _parts(annotation: DocumentAnnotation) -> tuple[Part, ...]:
    """Turn the drawn blocks into the schema's regions.

    Marks — the seal, the signature — are not regions of text and carry no
    transcription, so they are left out of the parts list; the metadata
    already records that the page has them.
    """
    lines_by_block: dict[str, list[tuple[str, object]]] = {}
    for line in annotation.lines:
        lines_by_block.setdefault(line.block, []).append((line.text, line.bbox))

    return tuple(
        Part(
            role=block.kind,
            text=block.text,
            bbox=block.bbox,
            lines=tuple(lines_by_block.get(block.kind, ())),  # type: ignore[arg-type]
        )
        for block in annotation.blocks
        if block.kind not in _MARK_BLOCKS and block.text
    )


def document_id(prefix: str, position: int) -> str:
    """Return the identifier a document's records share.

    Args:
        prefix: What to call documents in this corpus, e.g. ``doc``.
        position: The document's place in the batch.

    Returns:
        An identifier such as ``doc_000002``.
    """
    return f"{prefix}_{position:06d}"


def relative_path(path: Path, root: Path) -> str:
    """Return a path relative to the corpus root, in POSIX form."""
    return path.relative_to(root).as_posix()
