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

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from bitikocr import __version__
from bitikocr.data.models.schema import (
    AnnotationInfo,
    AnnotationStatus,
    DocumentMetadata,
    Era,
    Fact,
    FactsRecord,
    Language,
    Layout,
    NoiseLevel,
    Part,
    QualityInfo,
    SourceInfo,
    TextMode,
    TranscriptionRecord,
)
from bitikocr.data.synthetic.annotation import DocumentAnnotation
from bitikocr.data.synthetic.augment import AugmentationReport
from bitikocr.data.synthetic.facts import build_facts
from bitikocr.data.synthetic.generators.base import DocumentGenerator
from bitikocr.data.synthetic.records import DocumentRecord

__all__ = ["build_facts_record", "build_transcription_record"]

#: The schema's language for text a clerk wrote in each alphabet. Every
#: synthetic document is written in Uzbek.
_WRITTEN_LANGUAGE: dict[str, Language] = {
    "latin": "uz-latin",
    "cyrillic": "uz-cyrillic",
}

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
    image_size: tuple[int, int] | None = None,
    original_file: str | None = None,
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
        image_size: The saved page's ``(width, height)``; the annotation's
            own size when omitted.
        original_file: What the page was called before it joined the
            corpus. A synthetic page is born under its own name, so the
            dataset passes that.

    Returns:
        The transcription record.
    """
    quality = quality or AugmentationReport()
    parts = _parts(annotation, generator)
    roles = {part.role for part in parts}
    # A signature is a scribble, not a written name, and it may be anyone's:
    # a consent letter can carry only its certifier's.
    signed = any(part.role == "signature" and not part.text for part in parts)

    return TranscriptionRecord(
        id=document_id,
        image=image_path,
        image_size=image_size or annotation.size,
        source=SourceInfo(
            origin="synthetic",
            collection=collection,
            era=cast(Era, record.era),
            year_approx=record.year,
            original_file=original_file,
        ),
        metadata=DocumentMetadata(
            language=_languages(record, generator),
            document_type=record.document_type,
            text_mode=cast(TextMode, generator.text_mode),
            layout=cast(Layout, generator.layout),
            quality=QualityInfo(
                blur=quality.blur,
                rotation=round(quality.rotation, 3),
                skew=quality.skew,
                noise=cast(NoiseLevel, quality.noise),
                capture="scanner",
            ),
            has_handwriting=True,
            has_printed_text=generator.has_printed_text,
            has_stamp="stamp" in roles,
            has_signature=signed,
        ),
        text=annotation.text,
        parts=parts,
        annotation=AnnotationInfo(
            status=_SYNTHETIC_ANNOTATION,
            pre_annotator=f"generator:bitikocr@{__version__}",
            revision=1,
        ),
    )


def build_facts_record(
    record: DocumentRecord,
    document_id: str,
    image_path: str,
    annotation: DocumentAnnotation,
) -> FactsRecord:
    """Read the structured facts off one generated page.

    Args:
        record: What the page was told to say.
        document_id: The identifier this record shares with the
            transcription.
        image_path: Path to the page, relative to the corpus root.
        annotation: What was actually drawn. Only the fields that reached
            the page carry facts: a record holds every spelling its form
            variants use, and one variant may have no cell for some of them.

    Returns:
        The facts record.
    """
    facts: tuple[Fact, ...] = tuple(
        build_facts(
            record.document_type,
            _fields_on_page(record, annotation),
            record.dates,
        )
    )
    return FactsRecord(id=document_id, image=image_path, facts=facts)


def _fields_on_page(
    record: DocumentRecord, annotation: DocumentAnnotation
) -> Mapping[str, Any]:
    """Return the record's fields that the generator actually drew.

    Generators list what they wrote under ``fields`` in the annotation's
    metadata. A generator that records nothing there is taken to have
    written the whole record.
    """
    drawn = annotation.metadata.get("fields")
    if not isinstance(drawn, Mapping):
        return record.fields
    return {
        name: value for name, value in record.fields.items() if name in drawn
    }


def _languages(
    record: DocumentRecord, generator: DocumentGenerator
) -> list[Language]:
    """Return every language on the page, written first, then printed.

    A bilingual form is printed in Uzbek and Russian whichever alphabet the
    clerk filled it in with, so its printing adds to what was written.
    """
    seen: list[Language] = [_WRITTEN_LANGUAGE[record.script]]
    template = getattr(generator, "template", None)
    for language in getattr(template, "printed_languages", ()):
        if language not in seen:
            seen.append(cast(Language, language))
    return seen


def _parts(
    annotation: DocumentAnnotation, generator: DocumentGenerator
) -> tuple[Part, ...]:
    """Turn the drawn blocks into the schema's regions.

    Every block that left ink on the page becomes one part, in the order it
    was drawn — the marks too, since the schema has regions for a stamp and
    a signature. The generator says which region each block is and how it
    was written, and the block's outline, tilted if the page was skewed,
    becomes the part's polygon. A block that drew nothing has no outline and
    is left out.
    """
    parts = []
    for block in annotation.blocks:
        outline = block.outline
        if outline is None:
            continue
        role, hand = generator.part_of(block.kind)
        parts.append(
            Part(role=role, polygon=outline, text=block.text, hand=hand)
        )
    return tuple(parts)


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
