"""Tests for bitikocr.data.synthetic.augment."""

from __future__ import annotations

import random

import pytest
from PIL import Image

from bitikocr.data.synthetic.augment import (
    AugmentationProfile,
    augment_page,
    rotate_page,
)
from bitikocr.models.annotation import (
    BlockAnnotation,
    DocumentAnnotation,
    LineAnnotation,
)
from bitikocr.models.geometry import BoundingBox


@pytest.fixture()
def page() -> tuple[Image.Image, DocumentAnnotation]:
    """A plain page with one marked line."""
    image = Image.new("RGB", (400, 300), (245, 242, 232))
    box = BoundingBox(100, 120, 300, 160)
    annotation = DocumentAnnotation(
        text="hello",
        blocks=[BlockAnnotation("body", "hello", box)],
        lines=[LineAnnotation("body", "hello", box)],
        size=(400, 300),
        metadata={"seed": 1},
    )
    return image, annotation


def test_augmentation_keeps_the_page_size(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    image, annotation = page
    spoiled, _, _ = augment_page(image, annotation, random.Random(1))
    assert spoiled.size == image.size


def test_augmentation_changes_the_pixels(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    image, annotation = page
    spoiled, _, _ = augment_page(image, annotation, random.Random(1))
    assert spoiled.tobytes() != image.tobytes()


def test_the_same_seed_spoils_a_page_the_same_way(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    image, annotation = page
    first, _, _ = augment_page(image, annotation, random.Random(2))
    second, _, _ = augment_page(image, annotation, random.Random(2))
    assert first.tobytes() == second.tobytes()


def test_different_seeds_spoil_a_page_differently(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    image, annotation = page
    first, _, _ = augment_page(image, annotation, random.Random(2))
    second, _, _ = augment_page(image, annotation, random.Random(3))
    assert first.tobytes() != second.tobytes()


def test_the_empty_profile_leaves_the_page_alone(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    image, annotation = page
    spoiled, kept, report = augment_page(
        image, annotation, random.Random(1), AugmentationProfile.none()
    )
    assert spoiled is image
    assert kept is annotation
    assert not report.blur and not report.skew


def test_photometric_augmentation_leaves_the_boxes_alone(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    image, annotation = page
    profile = AugmentationProfile(max_rotation=0.0)
    _, kept, _ = augment_page(image, annotation, random.Random(1), profile)
    assert kept.lines[0].bbox == annotation.lines[0].bbox


# -- rotation --------------------------------------------------------------


def test_rotation_moves_the_boxes_with_the_ink(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    image, annotation = page
    _, turned = rotate_page(image, annotation, 5.0)
    assert turned.lines[0].bbox != annotation.lines[0].bbox


def test_rotation_records_the_angle_it_used(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    image, annotation = page
    _, turned = rotate_page(image, annotation, 5.0)
    assert turned.metadata["rotation"] == 5.0


def test_a_tiny_rotation_is_not_worth_doing(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    image, annotation = page
    rotated, turned = rotate_page(image, annotation, 0.001)
    assert rotated is image
    assert turned is annotation


def test_rotated_boxes_stay_on_the_page(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    image, annotation = page
    width, height = image.size
    for degrees in (-8.0, -1.0, 1.0, 8.0):
        _, turned = rotate_page(image, annotation, degrees)
        box = turned.lines[0].bbox
        assert box is not None
        assert 0 <= box.left < box.right <= width, degrees
        assert 0 <= box.top < box.bottom <= height, degrees


def test_a_rotated_box_still_contains_its_ink() -> None:
    """The box must follow the ink, not stay where the ink used to be."""
    image = Image.new("RGB", (400, 300), (255, 255, 255))
    for x in range(150, 250):
        for y in range(140, 160):
            image.putpixel((x, y), (0, 0, 0))

    box = BoundingBox(150, 140, 250, 160)
    annotation = DocumentAnnotation(
        text="x",
        blocks=[],
        lines=[LineAnnotation("body", "x", box)],
        size=(400, 300),
    )
    rotated, turned = rotate_page(image, annotation, 6.0)
    moved = turned.lines[0].bbox
    assert moved is not None

    dark = [
        (x, y)
        for x in range(rotated.width)
        for y in range(rotated.height)
        if sum(rotated.getpixel((x, y))) < 200
    ]
    assert dark, "expected the mark to survive the rotation"
    for x, y in dark:
        assert moved.left <= x <= moved.right, (x, y)
        assert moved.top <= y <= moved.bottom, (x, y)


def test_rotation_fills_the_corners_with_the_page_colour() -> None:
    """A rotated scan shows paper in the corners, not black."""
    image = Image.new("RGB", (200, 200), (240, 236, 226))
    annotation = DocumentAnnotation(
        text="", blocks=[], lines=[], size=(200, 200)
    )
    rotated, _ = rotate_page(image, annotation, 5.0)
    assert sum(rotated.getpixel((1, 1))) > 300
