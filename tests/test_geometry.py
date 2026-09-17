"""Tests for bitikocr.data.models.geometry."""

from __future__ import annotations

import pytest

from bitikocr.data.models.geometry import BoundingBox


def test_dimensions_are_derived_from_the_edges() -> None:
    box = BoundingBox(10, 20, 40, 60)
    assert (box.width, box.height) == (30, 40)


def test_to_list_round_trips_through_from_iterable() -> None:
    box = BoundingBox(1, 2, 3, 4)
    assert BoundingBox.from_iterable(box.to_list()) == box


def test_from_iterable_rounds_floats() -> None:
    assert BoundingBox.from_iterable([1.4, 2.6, 3.5, 4.0]) == BoundingBox(
        1, 3, 4, 4
    )


def test_from_iterable_rejects_the_wrong_number_of_coordinates() -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        BoundingBox.from_iterable([1, 2, 3])


def test_union_encloses_every_box() -> None:
    boxes = [BoundingBox(10, 10, 20, 20), BoundingBox(5, 30, 15, 40)]
    assert BoundingBox.union(boxes) == BoundingBox(5, 10, 20, 40)


def test_union_ignores_missing_boxes() -> None:
    box = BoundingBox(1, 2, 3, 4)
    assert BoundingBox.union([None, box, None]) == box


def test_union_of_nothing_is_none() -> None:
    assert BoundingBox.union([None, None]) is None


def test_a_box_converts_between_both_spellings() -> None:
    box = BoundingBox(10, 20, 40, 60)
    assert box.to_list() == [10, 20, 40, 60]
    assert box.to_xywh() == [10, 20, 30, 40]
