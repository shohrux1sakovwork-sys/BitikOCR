"""The corpus interchange schema: two records per document.

A document is described by a **transcription record** — what is written on
it, where, and under what conditions it was captured — and a **facts
record**, the structured values a reader would extract from it. They share
an ``id`` and point at the same image.

This is the format the training corpus is assembled in, so it is deliberately
independent of how a document was produced: a synthetic page, a real archive
scan and an augmented copy of one all describe themselves the same way, and
``source.origin`` is what tells them apart.

Boxes are ``[x, y, width, height]`` here, not the two-corner spelling the
renderer works in; see :meth:`~bitikocr.models.geometry.BoundingBox.to_xywh`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from bitikocr.models.geometry import BoundingBox

__all__ = [
    "AnnotationInfo",
    "AnnotationStatus",
    "CaptureKind",
    "DocumentMetadata",
    "Era",
    "Fact",
    "FactsRecord",
    "NoiseLevel",
    "Origin",
    "Part",
    "QualityInfo",
    "SourceInfo",
    "TextMode",
    "TranscriptionRecord",
]

#: How a document came to exist. The three-stage split depends on this.
Origin = Literal["synthetic", "real", "augmented"]

#: Which period a document belongs to. The main axis the corpus balances.
Era = Literal["old", "modern"]

#: Whether the text on a page is written, printed, or both.
TextMode = Literal["handwritten", "printed", "mixed"]

#: How far an annotation has been through review.
AnnotationStatus = Literal["auto", "reviewed", "gold", "unclear", "rejected"]

#: How much sensor noise the capture carries.
NoiseLevel = Literal["low", "medium", "high"]

#: What the page was captured with.
CaptureKind = Literal["scanner", "camera", "screenshot", "born_digital"]


@dataclass(frozen=True)
class SourceInfo:
    """Where a document came from.

    Args:
        origin: Synthetic, real, or an augmented copy of a real one.
        collection: The batch it belongs to, for slicing and provenance.
        era: Which period it is from.
        year_approx: The year it is dated, as best known.
        seed: For a synthetic page, the seed that reproduces it exactly.
        generator: For a synthetic page, what drew it.
        font: For a synthetic page, the handwriting font used.
    """

    origin: Origin
    collection: str
    era: Era
    year_approx: int | None = None
    seed: int | None = None
    generator: str | None = None
    font: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form, dropping unknown provenance."""
        payload: dict[str, Any] = {
            "origin": self.origin,
            "collection": self.collection,
            "era": self.era,
            "year_approx": self.year_approx,
        }
        for name in ("seed", "generator", "font"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


@dataclass(frozen=True)
class QualityInfo:
    """How cleanly the page was captured.

    Args:
        blur: Whether the image is out of focus.
        rotation: Skew in degrees, positive anticlockwise.
        skew: Whether the page is off square at all.
        noise: How much sensor grain it carries.
        capture: What captured it.
    """

    blur: bool = False
    rotation: float = 0.0
    skew: bool = False
    noise: NoiseLevel = "low"
    capture: CaptureKind = "scanner"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the capture quality."""
        return {
            "blur": self.blur,
            "rotation": self.rotation,
            "skew": self.skew,
            "noise": self.noise,
            "capture": self.capture,
        }


@dataclass(frozen=True)
class DocumentMetadata:
    """What kind of document this is and what it contains.

    Args:
        language: Languages present, as ISO codes.
        scripts: Every alphabet appearing on the page, printed or written.
        primary_script: The alphabet the document is mainly in.
        document_type: Which kind of document it is.
        text_mode: Whether its text is written, printed, or both.
        has_table: Whether it contains a table.
        has_formula: Whether it contains a formula.
        has_diagram: Whether it contains a diagram.
        has_handwriting: Whether anything on it is handwritten.
        has_printed_text: Whether anything on it is printed.
        has_stamp: Whether it carries an office seal.
        has_signature: Whether it carries a signature.
        layout: How the page is arranged.
        quality: How cleanly it was captured.
    """

    language: list[str]
    scripts: list[str]
    primary_script: str
    document_type: str
    text_mode: TextMode
    layout: str
    quality: QualityInfo
    has_table: bool = False
    has_formula: bool = False
    has_diagram: bool = False
    has_handwriting: bool = True
    has_printed_text: bool = False
    has_stamp: bool = False
    has_signature: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the document's metadata."""
        return {
            "language": list(self.language),
            "scripts": list(self.scripts),
            "primary_script": self.primary_script,
            "document_type": self.document_type,
            "text_mode": self.text_mode,
            "has_table": self.has_table,
            "has_formula": self.has_formula,
            "has_diagram": self.has_diagram,
            "has_handwriting": self.has_handwriting,
            "has_printed_text": self.has_printed_text,
            "has_stamp": self.has_stamp,
            "has_signature": self.has_signature,
            "layout": self.layout,
            "quality": self.quality.to_dict(),
        }


@dataclass(frozen=True)
class Part:
    """One region of the page, with what it says and where it is.

    Args:
        role: What the region is, e.g. ``title``, ``body``, ``recipient``.
        text: Its transcription.
        bbox: Where it sits, as ``[x, y, width, height]``.
        lines: The individual lines inside it, each with its own box. This
            extends the schema: it is what line-level HTR training needs, and
            a reader that only knows ``role``/``text``/``bbox`` can ignore it.
    """

    role: str
    text: str
    bbox: BoundingBox | None = None
    lines: tuple[tuple[str, BoundingBox | None], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the region."""
        payload: dict[str, Any] = {
            "role": self.role,
            "text": self.text,
            "bbox": self.bbox.to_xywh() if self.bbox else None,
        }
        if self.lines:
            payload["lines"] = [
                {"text": text, "bbox": box.to_xywh() if box else None}
                for text, box in self.lines
            ]
        return payload


@dataclass(frozen=True)
class AnnotationInfo:
    """How far the transcription has been through review.

    Args:
        status: Where it sits in the review process.
        pre_annotator: What produced the first draft.
        reviewer: Who checked it, if anyone has.
        reviewed_at: When it was checked, as ``YYYY-MM-DD``.
        revision: How many times it has been revised.
        unclear_reason: Why it is marked unclear, when it is.
    """

    status: AnnotationStatus = "auto"
    pre_annotator: str | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None
    revision: int = 1
    unclear_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the review state."""
        return {
            "status": self.status,
            "pre_annotator": self.pre_annotator,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "revision": self.revision,
            "unclear_reason": self.unclear_reason,
        }


@dataclass(frozen=True)
class TranscriptionRecord:
    """Everything written on one document, and how it was captured.

    Args:
        id: The document's identifier, shared with its facts record.
        image: Path to the page, relative to the corpus root.
        source: Where the document came from.
        metadata: What kind of document it is.
        text: The whole transcription in reading order, one line per
            physical line.
        parts: The regions the text is divided into.
        annotation: How far the transcription has been reviewed.
    """

    id: str
    image: str
    source: SourceInfo
    metadata: DocumentMetadata
    text: str
    parts: tuple[Part, ...] = ()
    annotation: AnnotationInfo = field(default_factory=AnnotationInfo)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the transcription record."""
        return {
            "id": self.id,
            "image": self.image,
            "source": self.source.to_dict(),
            "metadata": self.metadata.to_dict(),
            "target": {
                "text": self.text,
                "parts": [part.to_dict() for part in self.parts],
            },
            "annotation": self.annotation.to_dict(),
        }


@dataclass(frozen=True)
class Fact:
    """One structured value a reader would take off the page.

    Args:
        category: What kind of value it is; see
            :data:`~bitikocr.data.synthetic.facts.FACT_CATEGORIES`.
        value: The value, normalised where there is a normal form.
        evidence_text: The surface form as it appears on the page, so a
            wrong fact can be traced back to the transcription.
        fuzzy: True when the value is only partly legible, or inferred.
        subtype: A finer kind, for categories that have one — a ``number``
            may be an ``id``, ``phone``, ``amount`` or ``reference``.
        field: Which field of the document it came from, when known.
    """

    category: str
    value: str
    evidence_text: str
    fuzzy: bool = False
    subtype: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the fact."""
        payload: dict[str, Any] = {
            "category": self.category,
            "value": self.value,
            "fuzzy": self.fuzzy,
            "evidence_text": self.evidence_text,
        }
        if self.subtype is not None:
            payload["subtype"] = self.subtype
        if self.field is not None:
            payload["field"] = self.field
        return payload


@dataclass(frozen=True)
class FactsRecord:
    """The structured values one document carries.

    Args:
        id: The document's identifier, shared with its transcription.
        image: Path to the page, relative to the corpus root.
        facts: The values, in the order they are read off the page.
    """

    id: str
    image: str
    facts: tuple[Fact, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the facts record."""
        return {
            "id": self.id,
            "image": self.image,
            "facts": [fact.to_dict() for fact in self.facts],
        }
