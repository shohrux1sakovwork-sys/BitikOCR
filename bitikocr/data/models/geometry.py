"""Geometric primitives shared by every annotation type."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = ["BoundingBox", "Point", "Polygon", "polygon_bounds"]

#: One ``(x, y)`` position in image pixels.
Point = tuple[int, int]

#: A closed outline, its points in drawing order. Three or more points.
Polygon = tuple[Point, ...]


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

    def to_polygon(self) -> Polygon:
        """Return the box as its four corners, clockwise from top-left."""
        return (
            (self.left, self.top),
            (self.right, self.top),
            (self.right, self.bottom),
            (self.left, self.bottom),
        )

    def to_xywh(self) -> list[int]:
        """Return the box as ``[x, y, width, height]``.

        This is the corner-and-size spelling the dataset schema uses, as
        opposed to the two-corner spelling the renderer works in.
        """
        return [self.left, self.top, self.width, self.height]

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


def polygon_bounds(polygon: Sequence[Sequence[float]]) -> BoundingBox:
    """Return the smallest box enclosing a polygon.

    Args:
        polygon: The outline's points, each ``(x, y)``.

    Returns:
        The enclosing box, with the extreme points on its edges.

    Raises:
        ValueError: If the outline has fewer than three points.
    """
    if len(polygon) < 3:
        raise ValueError(
            f"A polygon needs at least 3 points, got {len(polygon)}"
        )
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return BoundingBox.from_iterable((min(xs), min(ys), max(xs), max(ys)))
