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

``text_type`` decides what a field *is*; see :data:`TEXT_TYPE_ROLES`. Fields
whose id ends in ``_line1``, ``_line2``, ... are one logical field written
across several printed lines, and are merged under their shared base name.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bitikocr.models.geometry import BoundingBox

__all__ = [
    "TEXT_TYPE_ROLES",
    "FieldGeometry",
    "FormTemplate",
    "LineSegment",
    "MarkArea",
]

#: ``text_type`` prefix mapped to the role the entry plays on the form.
#: Anything unmatched is a handwritten text field.
TEXT_TYPE_ROLES = {
    "round_stamp": "seal",
    "printed_digits": "serial",
    "signature": "signature",
    "digits": "numeric_field",
}

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
    """

    name: str
    bbox: BoundingBox
    baseline_y: int | None = None

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
        serial: Where the printed serial number goes, if the form has one.
        signature: Where the registrar signs, if the form has one.
    """

    name: str
    background: str
    native_size: tuple[int, int]
    fields: tuple[FieldGeometry, ...]
    seal: MarkArea | None = None
    serial: MarkArea | None = None
    signature: MarkArea | None = None

    @property
    def field_names(self) -> tuple[str, ...]:
        """Every handwritten field name, in reading order."""
        return tuple(field.name for field in self.fields)

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
        marks: dict[str, MarkArea] = {}

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
            seal=marks.get("seal"),
            serial=marks.get("serial"),
            signature=marks.get("signature"),
        )


def _by_index(item: tuple[int, LineSegment]) -> int:
    """Sort key putting ``_line1`` before ``_line2``."""
    return item[0]


def _role_of(text_type: str) -> str:
    """Classify a layout entry by its ``text_type``.

    Args:
        text_type: The layout's free-form type description.

    Returns:
        One of the roles in :data:`TEXT_TYPE_ROLES`, or ``"text_field"``.
    """
    for prefix, role in TEXT_TYPE_ROLES.items():
        if text_type.startswith(prefix):
            return role
    return "text_field"


def _read_entry(
    entry: dict[str, Any],
    segments: dict[str, list[tuple[int, LineSegment]]],
    numeric: set[str],
    marks: dict[str, MarkArea],
) -> None:
    """Fold one layout entry into the template being built.

    Args:
        entry: One item of the layout's ``fields`` list.
        segments: Field name mapped to its ``(line index, segment)`` pairs.
        numeric: Names of the fields whose values are digits.
        marks: Role mapped to the region reserved for it.

    Raises:
        ValueError: If the entry is missing an id or a bounding box.
    """
    try:
        entry_id = str(entry["id"])
        left, top, right, bottom = entry["bbox_xyxy"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Layout entry needs 'id' and 'bbox_xyxy': {entry}"
        ) from error

    baseline = entry.get("baseline_y")
    role = _role_of(str(entry.get("text_type", "")))
    bbox = BoundingBox.from_iterable([left, top, right, bottom])

    if role in ("seal", "serial", "signature"):
        marks[role] = MarkArea(
            name=entry_id,
            bbox=bbox,
            baseline_y=None if baseline is None else int(baseline),
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
