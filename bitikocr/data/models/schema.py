"""The corpus interchange schema: two records per document.

A document is described by a **transcription record** — what is written on
it, where, and under what conditions it was captured — and a **facts
record**, the structured values a reader would extract from it. They share
an ``id`` and point at the same image.

This is the format the training corpus is assembled in, so it is deliberately
independent of how a document was produced: a synthetic page, a real archive
scan and an augmented copy of one all describe themselves the same way, and
``source.origin`` is what tells them apart.

A region is outlined by a polygon of ``[x, y]`` points, and its box —
``[x, y, width, height]`` here, not the two-corner spelling the renderer
works in — is the one enclosing that outline. Fields a real scan may leave
unknown (``document_type``, ``text_mode``, ``era``, ``year_approx``, a
part's ``hand``) are nullable, and ``parts`` may be empty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from bitikocr.data.models.geometry import Polygon, polygon_bounds

__all__ = [
    "AnnotationInfo",
    "AnnotationStatus",
    "CaptureKind",
    "DocumentMetadata",
    "Era",
    "Fact",
    "FactsRecord",
    "Hand",
    "Language",
    "Layout",
    "NoiseLevel",
    "Origin",
    "Part",
    "QualityInfo",
    "Role",
    "SourceInfo",
    "TextMode",
    "TranscriptionRecord",
    "UncertainSpan",
]

#: How a document came to exist. The three-stage split depends on this.
Origin = Literal["synthetic", "real", "augmented"]

#: Which period a document belongs to. The main axis the corpus balances.
Era = Literal["old", "modern"]

#: A language on the page. Uzbek is split by alphabet, because to a
#: recogniser the two are different scripts to read.
Language = Literal["uz-cyrillic", "uz-latin", "ru"]

#: Whether the text on a page, or in one region of it, is written, printed,
#: or both.
TextMode = Literal["handwritten", "printed", "mixed"]

#: How a region's text was put there. The same values as :data:`TextMode`,
#: named for what it describes.
Hand = TextMode

#: What a region of the page is.
Role = Literal["header", "title", "body", "signature", "stamp", "other"]

#: How the page is arranged.
Layout = Literal["single_column", "two_column"]

#: How far an annotation has been through review.
AnnotationStatus = Literal["auto", "reviewed", "gold", "unclear", "rejected"]

#: How much sensor noise the capture carries.
NoiseLevel = Literal["low", "medium", "high"]

#: What the page was captured with.
CaptureKind = Literal["scanner", "camera", "screenshot", "born_digital"]


def _check(value: object, allowed: object, what: str) -> None:
    """Refuse a value outside a Literal's set, naming what it was for."""
    choices = get_args(allowed)
    if value not in choices:
        raise ValueError(f"{what} must be one of {choices}, got {value!r}")


@dataclass(frozen=True)
class SourceInfo:
    """Where a document came from.

    Args:
        origin: Synthetic, real, or an augmented copy of a real one.
        collection: The batch it belongs to, for slicing and provenance.
        era: Which period it is from, or None when that is not known.
        year_approx: The year it is dated, as best known.
        original_file: The file's name before it was ingested, with
            ``#page=N`` for a page taken out of a PDF.
    """

    origin: Origin
    collection: str
    era: Era | None = None
    year_approx: int | None = None
    original_file: str | None = None

    def __post_init__(self) -> None:
        _check(self.origin, Origin, "source.origin")
        if self.era is not None:
            _check(self.era, Era, "source.era")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the provenance."""
        return {
            "origin": self.origin,
            "collection": self.collection,
            "era": self.era,
            "year_approx": self.year_approx,
            "original_file": self.original_file,
        }


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
        language: Every language on the page, written or printed; see
            :data:`Language`.
        document_type: Which kind of document it is, or None.
        text_mode: Whether its text is written, printed, both, or None.
        layout: How the page is arranged.
        quality: How cleanly it was captured.
        has_table: Whether it contains a table.
        has_formula: Whether it contains a formula.
        has_diagram: Whether it contains a diagram.
        has_handwriting: Whether anything on it is handwritten.
        has_printed_text: Whether anything on it is printed.
        has_stamp: Whether it carries an office seal.
        has_signature: Whether it carries a signature.
    """

    language: list[Language]
    document_type: str | None
    text_mode: TextMode | None
    layout: Layout
    quality: QualityInfo
    has_table: bool = False
    has_formula: bool = False
    has_diagram: bool = False
    has_handwriting: bool = True
    has_printed_text: bool = False
    has_stamp: bool = False
    has_signature: bool = False

    def __post_init__(self) -> None:
        for language in self.language:
            _check(language, Language, "metadata.language")
        if self.text_mode is not None:
            _check(self.text_mode, TextMode, "metadata.text_mode")
        _check(self.layout, Layout, "metadata.layout")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the document's metadata."""
        return {
            "language": list(self.language),
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

    Its outline is the polygon and its box is derived from that, so the two
    can never disagree and the box always encloses the outline.

    Args:
        role: What the region is; see :data:`Role`.
        polygon: The region's outline, three or more ``(x, y)`` points.
        text: Its transcription. Empty for a region with no text, such as a
            signature scribble.
        hand: How its text was put there, or None when that is not known.

    Raises:
        ValueError: If the role or hand is not one the schema allows, or
            the outline has fewer than three points.
    """

    role: Role
    polygon: Polygon
    text: str = ""
    hand: Hand | None = None

    def __post_init__(self) -> None:
        _check(self.role, Role, "part.role")
        if self.hand is not None:
            _check(self.hand, Hand, "part.hand")
        polygon_bounds(self.polygon)  # Refuses fewer than three points.

    @property
    def bbox(self) -> list[int]:
        """The box enclosing the outline, as ``[x, y, width, height]``."""
        return polygon_bounds(self.polygon).to_xywh()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the region."""
        return {
            "role": self.role,
            "polygon": [[x, y] for x, y in self.polygon],
            "bbox": self.bbox,
            "text": self.text,
            "hand": self.hand,
        }


@dataclass(frozen=True)
class UncertainSpan:
    """A stretch of the transcription the annotator is unsure of.

    Args:
        text: The stretch, exactly as it appears in the transcription.
        note: Why it is uncertain.
    """

    text: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the span."""
        return {"text": self.text, "note": self.note}


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

    def __post_init__(self) -> None:
        _check(self.status, AnnotationStatus, "annotation.status")

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
        image_size: The page's ``(width, height)`` in pixels.
        source: Where the document came from.
        metadata: What kind of document it is.
        text: The whole transcription in reading order, one line per
            physical line.
        parts: The regions the page is divided into. Optional: a record may
            carry none.
        annotation: How far the transcription has been reviewed.
        uncertain_spans: Stretches of ``text`` the annotator is unsure of.
            The key is left out of the record when there are none.

    Raises:
        ValueError: If a part reaches off the page, or an uncertain span is
            not a stretch of the transcription.
    """

    id: str
    image: str
    image_size: tuple[int, int]
    source: SourceInfo
    metadata: DocumentMetadata
    text: str
    parts: Sequence[Part] = ()
    annotation: AnnotationInfo = field(default_factory=AnnotationInfo)
    uncertain_spans: Sequence[UncertainSpan] = ()

    def __post_init__(self) -> None:
        width, height = self.image_size
        for part in self.parts:
            for x, y in part.polygon:
                if not (0 <= x <= width and 0 <= y <= height):
                    raise ValueError(
                        f"{self.id}: {part.role} point ({x}, {y}) lies off "
                        f"the {width}x{height} page"
                    )
        for span in self.uncertain_spans:
            if span.text not in self.text:
                raise ValueError(
                    f"{self.id}: uncertain span {span.text!r} is not in "
                    "the transcription"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the transcription record."""
        payload: dict[str, Any] = {
            "id": self.id,
            "image": self.image,
            "image_size": list(self.image_size),
            "source": self.source.to_dict(),
            "metadata": self.metadata.to_dict(),
            "target": {
                "text": self.text,
                "parts": [part.to_dict() for part in self.parts],
            },
            "annotation": self.annotation.to_dict(),
        }
        if self.uncertain_spans:
            payload["uncertain_spans"] = [
                span.to_dict() for span in self.uncertain_spans
            ]
        return payload


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
