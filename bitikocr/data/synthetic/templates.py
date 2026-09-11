"""Where handwriting goes on a pre-printed form.

A :class:`FormTemplate` is the measured geometry of one blank form: which
underline each field is written on, where the seal is stamped, where the
registrar signs. It is loaded from a layout JSON produced by measuring a scan
of the blank form, so the measurements stay the single source of truth
instead of being retyped into Python.

Supporting a new form — another death certificate variant, a birth
certificate, a passport page — means adding a background image and a layout
JSON. No rendering code changes.

Layout JSON schema::

    {
      "template_name": "death_certificate_bilingual",
      "image": "death_certificate_bilingual.png",
      "width": 1419, "height": 1108,
      "fields": [
        {"id": "surname", "baseline_y": 331, "bbox_xyxy": [94, 293, 607, 335],
         "text_type": "uzbek_latin_word"},
        ...
      ]
    }

Measuring tools spell the same facts differently, so both dialects are
accepted: a box as ``bbox_xyxy: [x1, y1, x2, y2]`` or ``bbox: {x1, y1, x2,
y2}``, and a rule as ``underline_y`` (preferred, the printed line itself) or
``baseline_y``.

``text_type`` is free text describing the entry, and decides what it *is* —
see :data:`ROLE_KEYWORDS`. Entries whose id ends in ``_line1``, ``_line2``,
... are one logical field written across several printed lines, and are
merged under their shared base name.

A machine-printed entry may carry a ``prefix``, the label the value is
typeset behind. It belongs to the forms whose blank does not print that
label itself — one certificate prints "I-HR №" and leaves the digits to the
registry, another prints nothing and gets "№ 0024695" whole.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bitikocr.data.models.geometry import BoundingBox

__all__ = [
    "MARK_ROLES",
    "ROLE_KEYWORDS",
    "FieldGeometry",
    "FormTemplate",
    "LineSegment",
    "MarkArea",
]

#: Role a layout entry plays, matched against its ``text_type`` in this
#: order. The first role whose keywords appear wins, so "digits_7
#: (typographic)" is printed rather than handwritten. Anything unmatched is a
#: handwritten text field.
ROLE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("keep_out", ("qr", "barcode", "photo")),
    ("seal", ("stamp", "seal")),
    ("signature", ("signature",)),
    ("printed", ("printed_digits", "typographic", "typewriter", "series")),
    ("numeric_field", ("digit",)),
)

#: Roles that reserve a region rather than describing a written field.
MARK_ROLES = frozenset({"keep_out", "seal", "signature", "printed"})

_LINE_SUFFIX = re.compile(r"^(?P<base>.+)_line(?P<index>\d+)$")


@dataclass(frozen=True)
class LineSegment:
    """One printed underline a field may be written on.

    Args:
        baseline_y: Y of the printed rule, in template pixels.
        x_start: Left end of the rule.
        x_end: Right end of the rule.
    """

    baseline_y: int
    x_start: int
    x_end: int

    @property
    def width(self) -> int:
        """Writable width of the segment in template pixels."""
        return self.x_end - self.x_start


@dataclass(frozen=True)
class FieldGeometry:
    """A field and the printed line, or lines, it is written on.

    Args:
        name: Field name, used as the annotation block name.
        segments: The underlines available, in writing order.
        is_numeric: Whether the value is digits, which a clerk writes in a
            slightly different hand.
    """

    name: str
    segments: tuple[LineSegment, ...]
    is_numeric: bool = False

    @property
    def total_width(self) -> int:
        """Combined width of every segment, in template pixels."""
        return sum(segment.width for segment in self.segments)


@dataclass(frozen=True)
class MarkArea:
    """A region reserved for something other than a written field.

    Args:
        name: The area's name in the layout.
        bbox: The region, in template pixels.
        baseline_y: The printed rule inside the region, when there is one.
        prefix: Label typeset in front of a printed value. Forms that print
            their own label — "I-HR №" beside the serial — leave this
            empty; forms whose blank carries nothing there give the label
            here, so it is typeset along with the value.
    """

    name: str
    bbox: BoundingBox
    baseline_y: int | None = None
    prefix: str = ""

    @property
    def centre(self) -> tuple[int, int]:
        """Middle of the region, in template pixels."""
        return (
            (self.bbox.left + self.bbox.right) // 2,
            (self.bbox.top + self.bbox.bottom) // 2,
        )

    @property
    def radius(self) -> int:
        """Largest radius fitting inside the region."""
        return min(self.bbox.width, self.bbox.height) // 2


@dataclass(frozen=True)
class FormTemplate:
    """The measured geometry of one blank form.

    Args:
        name: Template name, used to select it on the command line.
        background: File name of the blank form scan.
        native_size: Size the coordinates were measured at, ``(width,
            height)`` in pixels.
        fields: Every handwritten field, in reading order.
        seal: Where the round office seal is stamped, if the form has one.
        printed: Areas holding machine-printed text — serial numbers, form
            series — each named after the field that fills it.
        signature: Where the registrar signs, if the form has one.
        keep_out: Regions already occupied by the blank form, such as a
            printed QR code. Nothing may be drawn over them.
        printed_scripts: Alphabets the blank form itself is printed in. A
            bilingual certificate carries both whatever the clerk writes in.
    """

    name: str
    background: str
    native_size: tuple[int, int]
    fields: tuple[FieldGeometry, ...]
    seal: MarkArea | None = None
    printed: tuple[MarkArea, ...] = ()
    signature: MarkArea | None = None
    keep_out: tuple[MarkArea, ...] = ()
    printed_scripts: tuple[str, ...] = ()

    @property
    def field_names(self) -> tuple[str, ...]:
        """Every handwritten field name, in reading order."""
        return tuple(field.name for field in self.fields)

    @property
    def printed_names(self) -> tuple[str, ...]:
        """Every machine-printed area's name, in layout order."""
        return tuple(area.name for area in self.printed)

    def field(self, name: str) -> FieldGeometry:
        """Return one field's geometry.

        Args:
            name: The field name.

        Returns:
            Where that field is written.

        Raises:
            KeyError: If the template has no such field.
        """
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Template {self.name!r} has no field {name!r}")

    @classmethod
    def load(cls, path: Path) -> FormTemplate:
        """Read a template from a layout JSON file.

        Args:
            path: Path to the layout JSON.

        Returns:
            The parsed template.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the layout is malformed.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed layout JSON: {path}") from error
        return cls.from_layout(payload, default_name=path.stem)

    @classmethod
    def from_layout(
        cls, payload: dict[str, Any], default_name: str = "unnamed"
    ) -> FormTemplate:
        """Build a template from a decoded layout JSON document.

        Args:
            payload: The decoded layout.
            default_name: Name to use when the layout does not carry one.

        Returns:
            The parsed template.

        Raises:
            ValueError: If required keys are missing or no field is defined.
        """
        try:
            width = int(payload["width"])
            height = int(payload["height"])
            background = str(payload["image"])
            entries = payload["fields"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "A layout needs 'width', 'height', 'image' and 'fields'"
            ) from error

        segments: dict[str, list[tuple[int, LineSegment]]] = {}
        numeric: set[str] = set()
        marks: dict[str, list[MarkArea]] = {}

        for entry in entries:
            _read_entry(entry, segments, numeric, marks)

        if not segments:
            raise ValueError("A layout needs at least one writable field")

        fields = tuple(
            FieldGeometry(
                name=name,
                segments=tuple(
                    segment for _, segment in sorted(ordered, key=_by_index)
                ),
                is_numeric=name in numeric,
            )
            for name, ordered in segments.items()
        )
        return cls(
            name=str(payload.get("template_name") or default_name),
            background=background,
            native_size=(width, height),
            fields=fields,
            seal=_only(marks.get("seal")),
            printed=tuple(marks.get("printed", ())),
            signature=_only(marks.get("signature")),
            keep_out=tuple(marks.get("keep_out", ())),
            printed_scripts=tuple(payload.get("printed_scripts", ())),
        )


def _by_index(item: tuple[int, LineSegment]) -> int:
    """Sort key putting ``_line1`` before ``_line2``."""
    return item[0]


def _only(areas: list[MarkArea] | None) -> MarkArea | None:
    """Return the single area for a role a form can only have one of."""
    return areas[0] if areas else None


def _role_of(text_type: str) -> str:
    """Classify a layout entry by its ``text_type``.

    Args:
        text_type: The layout's free-form type description.

    Returns:
        One of the roles in :data:`ROLE_KEYWORDS`, or ``"text_field"``.
    """
    lowered = text_type.lower()
    for role, keywords in ROLE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return role
    return "text_field"


def _read_bbox(entry: dict[str, Any]) -> BoundingBox:
    """Read an entry's box, in either layout dialect.

    Args:
        entry: One item of the layout's ``fields`` list.

    Returns:
        The box in template pixels.

    Raises:
        ValueError: If the entry carries no readable box.
    """
    corners = entry.get("bbox_xyxy")
    if corners is None:
        box = entry.get("bbox")
        if isinstance(box, dict):
            corners = [box.get(key) for key in ("x1", "y1", "x2", "y2")]

    try:
        return BoundingBox.from_iterable(corners)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Layout entry needs 'bbox_xyxy' or 'bbox': {entry}"
        ) from error


def _read_rule(entry: dict[str, Any]) -> int | None:
    """Read the printed rule an entry is written on, in either dialect.

    ``underline_y`` is the printed line itself and wins when both are given;
    ``baseline_y`` sits a few pixels above it and is the only rule some
    layouts record.

    Args:
        entry: One item of the layout's ``fields`` list.

    Returns:
        The rule's y in template pixels, or None when the entry has none.
    """
    for key in ("underline_y", "baseline_y"):
        value = entry.get(key)
        if value is not None:
            return int(value)
    return None


def _read_entry(
    entry: dict[str, Any],
    segments: dict[str, list[tuple[int, LineSegment]]],
    numeric: set[str],
    marks: dict[str, list[MarkArea]],
) -> None:
    """Fold one layout entry into the template being built.

    Args:
        entry: One item of the layout's ``fields`` list.
        segments: Field name mapped to its ``(line index, segment)`` pairs.
        numeric: Names of the fields whose values are digits.
        marks: Role mapped to the regions reserved for it.

    Raises:
        ValueError: If the entry is missing an id or a bounding box.
    """
    try:
        entry_id = str(entry["id"])
    except KeyError as error:
        raise ValueError(f"Layout entry needs an 'id': {entry}") from error

    bbox = _read_bbox(entry)
    baseline = _read_rule(entry)
    role = _role_of(str(entry.get("text_type", "")))

    if role in MARK_ROLES:
        marks.setdefault(role, []).append(
            MarkArea(
                name=entry_id,
                bbox=bbox,
                baseline_y=baseline,
                prefix=str(entry.get("prefix", "")),
            )
        )
        return

    if baseline is None:
        raise ValueError(f"Writable field {entry_id!r} has no baseline_y")

    match = _LINE_SUFFIX.match(entry_id)
    name = match.group("base") if match else entry_id
    index = int(match.group("index")) if match else 1

    segments.setdefault(name, []).append(
        (index, LineSegment(int(baseline), bbox.left, bbox.right))
    )
    if role == "numeric_field":
        numeric.add(name)
