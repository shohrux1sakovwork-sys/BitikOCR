"""Checking a finished dataset page by page, from its files alone.

A fine-tuning run trusts every annotation it is given, so a built corpus is
checked the way a consumer would read it: from the JSON and the image on
disk, without the generator that made them. Each page must

- follow the corpus schema exactly, key for key and value for value;
- describe the image it names, at the size it really is;
- outline every region inside the page, with a box that encloses the
  outline exactly;
- carry a facts record whose every piece of evidence can be found in the
  page's text, or on a seal, whose lettering is a region of its own rather
  than part of the reading-order transcription.

Across the corpus, no two pages may share their text or their image.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from PIL import Image

from bitikocr.data.models.schema import (
    AnnotationStatus,
    CaptureKind,
    Era,
    Hand,
    Language,
    Layout,
    NoiseLevel,
    Origin,
    Role,
    TextMode,
)
from bitikocr.data.synthetic.dataset import DatasetLayout
from bitikocr.data.synthetic.facts import FACT_CATEGORIES

__all__ = ["CorpusReport", "check_corpus", "check_page"]

_TOP_KEYS = ("id", "image", "image_size", "source", "metadata", "target")
_SOURCE_KEYS = {"origin", "collection", "era", "year_approx", "original_file"}
_METADATA_KEYS = {
    "language",
    "document_type",
    "text_mode",
    "has_table",
    "has_formula",
    "has_diagram",
    "has_handwriting",
    "has_printed_text",
    "has_stamp",
    "has_signature",
    "layout",
    "quality",
}
_QUALITY_KEYS = {"blur", "rotation", "skew", "noise", "capture"}
_PART_KEYS = ["role", "polygon", "bbox", "text", "hand"]
_ANNOTATION_KEYS = {
    "status",
    "pre_annotator",
    "reviewer",
    "reviewed_at",
    "revision",
    "unclear_reason",
}
_FLAGS = (
    "has_table",
    "has_formula",
    "has_diagram",
    "has_handwriting",
    "has_printed_text",
    "has_stamp",
    "has_signature",
)


@dataclass
class CorpusReport:
    """What a corpus check found.

    Args:
        pages: How many pages were checked.
        problems: ``(page id, problem)`` for everything wrong.
        counts: Axis name mapped to how the pages spread over it.
    """

    pages: int = 0
    problems: list[tuple[str, str]] = field(default_factory=list)
    counts: dict[str, Counter[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether nothing was wrong."""
        return not self.problems

    def count(self, axis: str, value: object) -> None:
        """Tally one page under an axis."""
        self.counts.setdefault(axis, Counter())[str(value)] += 1


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _one_of(value: object, literal: object) -> bool:
    return value in get_args(literal)


def check_page(
    root: Path, annotation_path: Path
) -> tuple[list[str], dict[str, Any]]:
    """Check one page's annotation, image and facts.

    Args:
        root: The dataset directory.
        annotation_path: The page's transcription record.

    Returns:
        ``(problems, record)``: everything wrong with the page, and the
        decoded transcription for corpus-wide checks.
    """
    problems: list[str] = []
    try:
        record = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable annotation: {error}"], {}

    def need(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    keys = list(record)
    need(
        keys[: len(_TOP_KEYS)] == list(_TOP_KEYS)
        and set(keys) - set(_TOP_KEYS) <= {"annotation", "uncertain_spans"}
        and "annotation" in keys,
        f"top-level keys {keys}",
    )
    need(record.get("id") == annotation_path.stem, "id differs from file")

    image_path = root / str(record.get("image", ""))
    size = record.get("image_size")
    try:
        with Image.open(image_path) as image:
            need(size == list(image.size), f"image_size {size} != {image.size}")
            need(image.mode == "RGB", f"image mode {image.mode}")
    except OSError as error:
        return [*problems, f"unreadable image: {error}"], record
    width, height = (
        size if isinstance(size, list) and len(size) == 2 else (0, 0)
    )

    _check_source(record.get("source", {}), image_path, need)
    _check_metadata(record.get("metadata", {}), need)
    text = _check_target(record.get("target", {}), width, height, need)
    _check_flags(record, need)
    _check_annotation(record.get("annotation", {}), need)
    _check_facts(root, record, text, need)
    return problems, record


def _check_source(source: Mapping[str, Any], image: Path, need: Any) -> None:
    need(set(source) == _SOURCE_KEYS, f"source keys {sorted(source)}")
    need(_one_of(source.get("origin"), Origin), "source.origin")
    need(source.get("era") is None or _one_of(source["era"], Era), "era")
    year = source.get("year_approx")
    need(year is None or _is_int(year), "year_approx")
    need(source.get("original_file") == image.name, "original_file")
    if year is not None and source.get("era") is not None:
        need(
            (source["era"] == "modern") == (year >= 2000),
            f"era {source['era']} disagrees with year {year}",
        )


def _check_metadata(metadata: Mapping[str, Any], need: Any) -> None:
    need(set(metadata) == _METADATA_KEYS, f"metadata keys {sorted(metadata)}")
    languages = metadata.get("language")
    need(
        isinstance(languages, list)
        and languages
        and len(set(languages)) == len(languages)
        and all(_one_of(lang, Language) for lang in languages),
        f"language {languages}",
    )
    mode = metadata.get("text_mode")
    need(mode is None or _one_of(mode, TextMode), "text_mode")
    need(_one_of(metadata.get("layout"), Layout), "layout")
    for flag in _FLAGS:
        need(isinstance(metadata.get(flag), bool), f"{flag} is not a boolean")
    quality = metadata.get("quality", {})
    need(set(quality) == _QUALITY_KEYS, f"quality keys {sorted(quality)}")
    need(isinstance(quality.get("blur"), bool), "quality.blur")
    need(isinstance(quality.get("skew"), bool), "quality.skew")
    need(isinstance(quality.get("rotation"), (int, float)), "quality.rotation")
    need(_one_of(quality.get("noise"), NoiseLevel), "quality.noise")
    need(_one_of(quality.get("capture"), CaptureKind), "quality.capture")


def _check_target(
    target: Mapping[str, Any], width: int, height: int, need: Any
) -> str:
    need(set(target) == {"text", "parts"}, f"target keys {sorted(target)}")
    text = target.get("text")
    need(isinstance(text, str) and text.strip(), "target.text is empty")
    text = text if isinstance(text, str) else ""
    parts = target.get("parts")
    need(isinstance(parts, list), "target.parts is not a list")

    for number, part in enumerate(parts or []):
        where = f"part {number}"
        need(list(part) == _PART_KEYS, f"{where} keys {list(part)}")
        need(_one_of(part.get("role"), Role), f"{where} role")
        hand = part.get("hand")
        need(hand is None or _one_of(hand, Hand), f"{where} hand")
        polygon = part.get("polygon") or []
        need(len(polygon) >= 3, f"{where} polygon has {len(polygon)} points")
        if not polygon:
            continue
        need(
            all(
                len(point) == 2 and all(_is_int(v) for v in point)
                for point in polygon
            ),
            f"{where} polygon is not integer points",
        )
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        need(
            all(0 <= x <= width for x in xs)
            and all(0 <= y <= height for y in ys),
            f"{where} ({part.get('role')}) leaves the page",
        )
        expected = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        need(part.get("bbox") == expected, f"{where} bbox != polygon bounds")
        need(expected[2] > 0 and expected[3] > 0, f"{where} has no area")
        if part.get("role") == "stamp":
            # A seal's lettering runs round a ring; it is a region of the
            # page but not part of the reading-order transcription.
            continue
        for line in str(part.get("text") or "").splitlines():
            need(
                line.strip() in text,
                f"{where} text {line.strip()[:30]!r} not in target.text",
            )
    return text


def _check_flags(record: Mapping[str, Any], need: Any) -> None:
    metadata = record.get("metadata", {})
    parts = record.get("target", {}).get("parts") or []
    roles = {part.get("role") for part in parts}
    hands = {part.get("hand") for part in parts}
    need(
        metadata.get("has_stamp") == ("stamp" in roles),
        "has_stamp disagrees with the parts",
    )
    if metadata.get("has_signature"):
        need("signature" in roles, "has_signature without a signature part")
    if metadata.get("has_handwriting"):
        need("handwritten" in hands, "has_handwriting without written part")


def _check_annotation(annotation: Mapping[str, Any], need: Any) -> None:
    need(
        set(annotation) == _ANNOTATION_KEYS,
        f"annotation keys {sorted(annotation)}",
    )
    need(_one_of(annotation.get("status"), AnnotationStatus), "status")
    need(_is_int(annotation.get("revision")), "revision")


def _check_facts(
    root: Path, record: Mapping[str, Any], text: str, need: Any
) -> None:
    layout = DatasetLayout(root)
    path = layout.facts / f"{record.get('id')}.json"
    try:
        facts = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        need(False, f"unreadable facts: {error}")
        return
    need(list(facts) == ["id", "image", "facts"], f"facts keys {list(facts)}")
    need(facts.get("id") == record.get("id"), "facts id")
    need(facts.get("image") == record.get("image"), "facts image")
    # Evidence may come from a seal, whose lettering only its part carries.
    parts = record.get("target", {}).get("parts") or []
    readable = "\n".join([text, *(str(p.get("text") or "") for p in parts)])
    words = set(readable.split())
    for number, fact in enumerate(facts.get("facts", [])):
        where = f"fact {number}"
        category = fact.get("category")
        need(category in FACT_CATEGORIES, f"{where} category {category}")
        subtype = fact.get("subtype")
        need(
            subtype is None or subtype in FACT_CATEGORIES.get(category, ()),
            f"{where} subtype {subtype}",
        )
        need(isinstance(fact.get("fuzzy"), bool), f"{where} fuzzy")
        need(bool(str(fact.get("value", "")).strip()), f"{where} has no value")
        evidence = str(fact.get("evidence_text", ""))
        missing = [
            token
            for token in evidence.split()
            if token not in words and token not in readable
        ]
        need(
            not missing,
            f"{where} ({category}) evidence not on the page: {missing[:3]}",
        )


def check_corpus(root: Path, ids: Iterable[str] | None = None) -> CorpusReport:
    """Check every page of a dataset, and the dataset as a whole.

    Args:
        root: The dataset directory.
        ids: Check only these pages; all of them when omitted.

    Returns:
        The report.
    """
    layout = DatasetLayout(root)
    report = CorpusReport()
    paths = (
        [layout.annotations / f"{i}.json" for i in ids]
        if ids is not None
        else sorted(layout.annotations.glob("*.json"))
    )
    texts: dict[str, str] = {}
    images: dict[str, str] = {}
    for path in paths:
        report.pages += 1
        problems, record = check_page(root, path)
        report.problems.extend((path.stem, problem) for problem in problems)
        if not record:
            continue
        metadata = record.get("metadata", {})
        report.count("type", metadata.get("document_type"))
        report.count("language", metadata.get("language", ["?"])[0])
        report.count("capture", metadata.get("quality", {}).get("capture"))
        report.count("era", record.get("source", {}).get("era"))
        for part in record.get("target", {}).get("parts", []):
            report.count("role", part.get("role"))

        text_key = hashlib.sha1(
            record.get("target", {}).get("text", "").encode("utf-8")
        ).hexdigest()
        if text_key in texts:
            report.problems.append(
                (path.stem, f"same text as {texts[text_key]}")
            )
        texts[text_key] = path.stem
        image_path = root / str(record.get("image", ""))
        if image_path.exists():
            image_key = hashlib.sha1(image_path.read_bytes()).hexdigest()
            if image_key in images:
                report.problems.append(
                    (path.stem, f"same image as {images[image_key]}")
                )
            images[image_key] = path.stem

    if layout.index.exists():
        indexed = {
            json.loads(line)["id"]
            for line in layout.index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        for path in paths:
            if path.stem not in indexed:
                report.problems.append((path.stem, "missing from the index"))
    return report
