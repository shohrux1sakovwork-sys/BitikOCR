"""Data classes, schemas and type definitions shared across the project."""

from bitikocr.models.annotation import (
    BlockAnnotation,
    DocumentAnnotation,
    LineAnnotation,
)
from bitikocr.models.geometry import BoundingBox

__all__ = [
    "BlockAnnotation",
    "BoundingBox",
    "DocumentAnnotation",
    "LineAnnotation",
]
