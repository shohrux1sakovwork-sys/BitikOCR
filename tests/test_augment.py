"""Tests for bitikocr.data.synthetic.augment."""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest
from PIL import Image

from bitikocr.data.models.geometry import BoundingBox
from bitikocr.data.synthetic.annotation import (
    BlockAnnotation,
    DocumentAnnotation,
    LineAnnotation,
)
from bitikocr.data.synthetic.augment import (
    AugmentationPlan,
    AugmentationProfile,
    Hole,
    apply_plan,
    augment_page,
    homography,
    plan_augmentation,
    rotate_page,
    warp_page,
)
from bitikocr.data.synthetic.augment_gpu import cuda_available


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


def test_rotation_turns_a_blocks_outline_with_the_ink(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    """The outline stays a four-cornered shape, now tilted, and its box is
    the one that encloses it."""
    image, annotation = page
    _, turned = rotate_page(image, annotation, 6.0)
    block = turned.blocks[0]
    assert block.polygon is not None and block.bbox is not None
    assert len(block.polygon) == 4
    xs = [x for x, _ in block.polygon]
    ys = [y for _, y in block.polygon]
    assert len(set(xs)) > 2 and len(set(ys)) > 2, "the outline did not tilt"
    assert block.bbox.left <= min(xs) and max(xs) <= block.bbox.right
    assert block.bbox.top <= min(ys) and max(ys) <= block.bbox.bottom


def test_an_upright_block_is_outlined_by_its_box(
    page: tuple[Image.Image, DocumentAnnotation],
) -> None:
    _, annotation = page
    block = annotation.blocks[0]
    assert block.polygon is None
    assert block.outline == (
        (100, 120),
        (300, 120),
        (300, 160),
        (100, 160),
    )


def test_a_rotated_outline_stays_on_the_page() -> None:
    image = Image.new("RGB", (200, 100), (245, 242, 232))
    corner = BoundingBox(150, 60, 200, 100)
    annotation = DocumentAnnotation(
        text="x",
        blocks=[BlockAnnotation("body", "x", corner)],
        lines=[],
        size=(200, 100),
    )
    _, turned = rotate_page(image, annotation, -8.0)
    polygon = turned.blocks[0].polygon
    assert polygon is not None
    for x, y in polygon:
        assert 0 <= x <= 200 and 0 <= y <= 100


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

    pixels = np.asarray(rotated).sum(axis=2)
    dark = [(int(x), int(y)) for y, x in zip(*np.nonzero(pixels < 200))]
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
    assert np.asarray(rotated)[1, 1].sum() > 300


# -- plans, photographs and the GPU ----------------------------------------


def _marked_page() -> tuple[Image.Image, DocumentAnnotation]:
    """A white page with one black bar and its box."""
    image = Image.new("RGB", (400, 300), (255, 255, 255))
    for x in range(150, 250):
        for y in range(140, 160):
            image.putpixel((x, y), (0, 0, 0))
    box = BoundingBox(150, 140, 250, 160)
    annotation = DocumentAnnotation(
        text="x",
        blocks=[BlockAnnotation("body", "x", box)],
        lines=[LineAnnotation("body", "x", box)],
        size=(400, 300),
    )
    return image, annotation


_PHOTO = AugmentationProfile(
    camera_share=1.0,
    blur=0.0,
    noise=0.0,
    speck_density=0.0,
    vignette=0.0,
    jpeg_quality=None,
)


def test_a_plan_round_trips_through_its_dict() -> None:
    plan = plan_augmentation(random.Random(4), AugmentationProfile.varied())
    assert AugmentationPlan.from_dict(plan.to_dict()) == plan


def test_the_varied_profile_uses_every_effect() -> None:
    rng = random.Random(0)
    seen = {
        effect
        for _ in range(300)
        for effect in plan_augmentation(
            rng, AugmentationProfile.varied()
        ).effects
    }
    assert seen >= {
        "perspective",
        "crease",
        "stain",
        "shadow",
        "photocopy",
        "low_resolution",
        "blur",
    }


def test_the_archive_profile_scans_and_levels_every_page() -> None:
    rng = random.Random(0)
    plans = [
        plan_augmentation(rng, AugmentationProfile.archive())
        for _ in range(200)
    ]
    assert all(plan.capture == "scanner" for plan in plans)
    assert all(plan.paper_level is not None for plan in plans)
    assert not any(plan.stains for plan in plans)
    assert any(plan.holes for plan in plans)


def test_the_varied_profile_neither_levels_nor_punches() -> None:
    rng = random.Random(0)
    for _ in range(100):
        plan = plan_augmentation(rng, AugmentationProfile.varied())
        assert plan.paper_level is None
        assert plan.holes == ()


def test_a_plan_with_holes_round_trips_through_its_dict() -> None:
    plan = plan_augmentation(random.Random(4), AugmentationProfile.archive())
    assert plan.holes
    assert AugmentationPlan.from_dict(plan.to_dict()) == plan


def test_levelling_turns_cream_paper_white_and_keeps_the_ink() -> None:
    image, annotation = _marked_page()
    cream = Image.new("RGB", image.size, (225, 208, 178))
    cream.paste(image.crop((100, 120, 300, 160)), (100, 120))
    plan = AugmentationPlan(strength=1.0, paper_level=(252.0, 252.0, 252.0))
    levelled, _, _ = apply_plan(cream, annotation, plan)
    pixels = np.asarray(levelled).astype(int)
    paper = pixels[20:60, 20:60].reshape(-1, 3)
    assert np.all(np.abs(paper - 252) <= 3)
    assert pixels[120:160, 100:300].min() < 100


def test_a_scan_dulls_the_inks_colour() -> None:
    image, annotation = _marked_page()
    blue = Image.new("RGB", image.size, (255, 255, 255))
    blue.paste((30, 40, 200), (150, 140, 250, 160))
    plan = AugmentationPlan(
        strength=1.0, paper_level=(255.0, 255.0, 255.0), saturation=0.5
    )
    scanned, _, _ = apply_plan(blue, annotation, plan)
    ink = np.asarray(scanned).astype(int)[145:155, 160:240].reshape(-1, 3)
    assert 0 < np.median(ink[:, 2] - ink[:, 0]) < 170 * 0.6


def test_a_punch_hole_never_covers_writing() -> None:
    image = Image.new("RGB", (400, 300), (250, 250, 250))
    box = BoundingBox(0, 120, 400, 160)
    annotation = DocumentAnnotation(
        text="hello",
        blocks=[BlockAnnotation("body", "hello", box)],
        lines=[LineAnnotation("body", "hello", box)],
        size=(400, 300),
    )
    holes = (
        Hole(x=0.05, y=0.2, radius=0.02, shade=20),
        Hole(x=0.05, y=0.47, radius=0.02, shade=20),
    )
    plan = AugmentationPlan(strength=1.0, holes=holes)
    punched, _, _ = apply_plan(image, annotation, plan)
    pixels = np.asarray(punched).sum(axis=2)
    assert pixels[60, 20] < 100
    assert pixels[120:160].min() > 600


def test_a_photographed_page_reports_a_camera() -> None:
    image, annotation = _marked_page()
    plan = plan_augmentation(random.Random(1), _PHOTO)
    _, _, report = apply_plan(image, annotation, plan)
    assert report.capture == "camera"
    assert report.skew


def test_a_photographed_outline_still_holds_its_ink() -> None:
    image, annotation = _marked_page()
    plan = plan_augmentation(random.Random(1), _PHOTO)
    moved, turned, _ = apply_plan(image, annotation, plan)
    outline = turned.blocks[0].outline
    assert outline is not None and len(outline) == 4

    pixels = np.asarray(moved).sum(axis=2)
    dark = [(int(x), int(y)) for y, x in zip(*np.nonzero(pixels < 150))]
    assert dark
    xs = [x for x, _ in outline]
    ys = [y for _, y in outline]
    # The outline is a tilted quadrilateral; its bounds hold all the ink.
    for x, y in dark:
        assert min(xs) - 1 <= x <= max(xs) + 1
        assert min(ys) - 1 <= y <= max(ys) + 1


def test_a_warp_by_the_identity_changes_nothing() -> None:
    image, annotation = _marked_page()
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    moved, kept = warp_page(image, annotation, identity, (255, 255, 255))
    assert np.array_equal(np.asarray(moved), np.asarray(image))
    assert kept.blocks[0].bbox == annotation.blocks[0].bbox


def test_a_homography_sends_its_points_where_asked() -> None:
    source = [(0, 0), (10, 0), (10, 10), (0, 10)]
    target = [(1, 2), (12, 1), (11, 13), (0, 9)]
    matrix = homography(source, target)
    for (x, y), (u, v) in zip(source, target):
        w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
        assert (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / w == (
            pytest.approx(u)
        )
        assert (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / w == (
            pytest.approx(v)
        )


@pytest.mark.skipif(not cuda_available(), reason="needs a CUDA device")
def test_the_gpu_draws_what_the_cpu_draws() -> None:
    image, annotation = _marked_page()
    profile = AugmentationProfile(
        camera_share=1.0, noise=0.0, speck_density=0.0, jpeg_quality=None
    )
    # The light field is drawn by each backend's own generator, so it is
    # left out to compare the rest.
    plan = dataclasses.replace(
        plan_augmentation(random.Random(3), profile), light=0.0
    )
    on_cpu, cpu_truth, _ = apply_plan(image, annotation, plan, "cpu")
    on_gpu, gpu_truth, _ = apply_plan(image, annotation, plan, "cuda")
    assert cpu_truth.blocks == gpu_truth.blocks
    difference = np.abs(
        np.asarray(on_cpu).astype(int) - np.asarray(on_gpu).astype(int)
    )
    # Only the resampling kernels differ between the two.
    assert difference.mean() < 2
