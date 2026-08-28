"""Geometric primitives shared by every annotation type."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["BoundingBox"]


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned rectangle in image pixel coordinates.

    Args:
        left: X coordinate of the left edge, inclusive.
        top: Y coordinate of the top edge, inclusive.
        right: X coordinate of the right edge, exclusive.
        bottom: Y coordinate of the bottom edge, exclusive.
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        """Width of the box in pixels."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Height of the box in pixels."""
        return self.bottom - self.top

    def to_list(self) -> list[int]:
        """Return the box as ``[left, top, right, bottom]``."""
        return [self.left, self.top, self.right, self.bottom]

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> BoundingBox:
        """Build a box from four numbers in ``left, top, right, bottom`` order.

        Args:
            values: Any iterable yielding exactly four numbers.

        Returns:
            The corresponding box, with coordinates rounded to integers.

        Raises:
            ValueError: If ``values`` does not contain exactly four numbers.
        """
        coords = [round(v) for v in values]
        if len(coords) != 4:
            raise ValueError(
                f"A bounding box needs exactly 4 coordinates, got {len(coords)}"
            )
        return cls(*coords)

    @classmethod
    def union(cls, boxes: Iterable[BoundingBox | None]) -> BoundingBox | None:
        """Return the smallest box containing every given box.

        Args:
            boxes: Boxes to merge. ``None`` entries are ignored.

        Returns:
            The enclosing box, or None if no box was given.
        """
        present = [box for box in boxes if box is not None]
        if not present:
            return None
        return cls(
            left=min(box.left for box in present),
            top=min(box.top for box in present),
            right=max(box.right for box in present),
            bottom=max(box.bottom for box in present),
        )
