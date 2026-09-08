"""Making two pages of the same form look like two different scans.

A generated page is clean: even lighting, square to the sensor, no grain.
Real archive scans are none of those things, and a recogniser trained only
on clean pages learns the cleanliness. These are the post-processing steps
that spoil a page realistically.

Photometric steps leave every bounding box where it was. The one geometric
step, a slight rotation, moves the ink, so it transforms the annotation with
it rather than leaving the ground truth behind.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from bitikocr.data.synthetic.annotation import (
    BlockAnnotation,
    DocumentAnnotation,
    LineAnnotation,
)
from bitikocr.models.geometry import BoundingBox

__all__ = [
    "AugmentationProfile",
    "AugmentationReport",
    "augment_page",
    "rotate_page",
]


@dataclass(frozen=True)
class AugmentationReport:
    """What was actually done to one page.

    The corpus schema records capture conditions, and a recogniser can be
    evaluated by them, so a page has to be able to say how blurred, skewed
    and noisy it is rather than leaving it to be guessed.

    Args:
        blur: Whether a defocus blur was applied.
        rotation: The skew applied, in degrees.
        noise: How much grain was added, as a band.
    """

    blur: bool = False
    rotation: float = 0.0
    noise: str = "low"

    @property
    def skew(self) -> bool:
        """Whether the page ended up off square at all."""
        return abs(self.rotation) >= 0.05

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serialisable form of the report."""
        return {
            "blur": self.blur,
            "rotation": round(self.rotation, 3),
            "skew": self.skew,
            "noise": self.noise,
        }


@dataclass(frozen=True)
class AugmentationProfile:
    """How hard to spoil a page.

    Args:
        strength: Scales every photometric effect. 0.0 leaves the page as
            rendered.
        max_rotation: Largest scan skew in degrees, either way.
        vignette: How much darker the page edges get, 0.0 to 1.0.
        blur: Largest defocus blur radius in pixels.
        noise: Largest sensor noise, as a pixel standard deviation.
        jpeg_quality: Range a JPEG quality is drawn from, or None to skip
            re-compression.
        speck_density: Dust and ink specks per megapixel.
    """

    strength: float = 1.0
    max_rotation: float = 0.7
    vignette: float = 0.35
    blur: float = 0.9
    noise: float = 7.0
    jpeg_quality: tuple[int, int] | None = (60, 93)
    speck_density: float = 60.0

    @classmethod
    def none(cls) -> AugmentationProfile:
        """Return a profile that changes nothing."""
        return cls(
            strength=0.0,
            max_rotation=0.0,
            vignette=0.0,
            blur=0.0,
            noise=0.0,
            jpeg_quality=None,
            speck_density=0.0,
        )


def augment_page(
    image: Image.Image,
    annotation: DocumentAnnotation,
    rng: random.Random,
    profile: AugmentationProfile | None = None,
) -> tuple[Image.Image, DocumentAnnotation, AugmentationReport]:
    """Spoil a rendered page the way a scanner and time would.

    Args:
        image: The clean rendered page.
        annotation: Its ground truth.
        rng: Random source. The same seed spoils the page the same way.
        profile: How hard to spoil it; a default profile when omitted.

    Returns:
        ``(image, annotation, report)``. The annotation is a new object
        whenever a step moved the ink, and the report says what was done.
    """
    profile = profile or AugmentationProfile()
    if profile.strength <= 0 and profile.max_rotation <= 0:
        return image, annotation, AugmentationReport()

    image = image.convert("RGB")
    degrees = 0.0
    if profile.max_rotation > 0:
        degrees = rng.uniform(-profile.max_rotation, profile.max_rotation)
        image, annotation = rotate_page(image, annotation, degrees)

    image = _apply_paper_and_light(image, rng, profile.strength)
    image = _apply_vignette(image, rng, profile.vignette * profile.strength)
    image = _apply_specks(image, rng, profile.speck_density * profile.strength)

    blurred = profile.blur > 0 and rng.random() < 0.6
    if blurred:
        image = image.filter(
            ImageFilter.GaussianBlur(
                rng.uniform(0.2, profile.blur) * profile.strength
            )
        )

    grain = profile.noise * profile.strength
    if profile.noise > 0:
        image = _apply_noise(image, rng, grain)
    if profile.jpeg_quality is not None:
        image = _apply_jpeg(image, rng, profile.jpeg_quality)

    return (
        image,
        annotation,
        AugmentationReport(
            blur=blurred, rotation=degrees, noise=_noise_band(grain)
        ),
    )


def _noise_band(deviation: float) -> str:
    """Describe a grain level the way the corpus schema does."""
    if deviation < 3.0:
        return "low"
    if deviation < 8.0:
        return "medium"
    return "high"


# -- geometry --------------------------------------------------------------


def rotate_page(
    image: Image.Image, annotation: DocumentAnnotation, degrees: float
) -> tuple[Image.Image, DocumentAnnotation]:
    """Skew a page as if it were laid crookedly on the scanner.

    The page keeps its size, so the ground truth stays in the same
    coordinate space. Boxes are re-derived from their rotated corners, which
    grows them slightly — the cost of keeping them axis-aligned.

    Args:
        image: The page to skew.
        annotation: Its ground truth.
        degrees: Rotation, positive anticlockwise.

    Returns:
        ``(image, annotation)`` with both rotated.
    """
    if abs(degrees) < 0.01:
        return image, annotation

    width, height = image.size
    rotated = image.rotate(
        degrees,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=_border_color(image),
    )
    centre = (width / 2, height / 2)

    def turn(box: BoundingBox | None) -> BoundingBox | None:
        if box is None:
            return None
        corners = [
            _rotate_point(x, y, centre, degrees)
            for x, y in (
                (box.left, box.top),
                (box.right, box.top),
                (box.right, box.bottom),
                (box.left, box.bottom),
            )
        ]
        # Round outwards so the box never clips the ink it follows, then
        # trim to the page: a rotation can carry ink off the edge, and a
        # box that lands entirely outside describes nothing.
        turned = BoundingBox(
            left=max(0, math.floor(min(x for x, _ in corners))),
            top=max(0, math.floor(min(y for _, y in corners))),
            right=min(width, math.ceil(max(x for x, _ in corners))),
            bottom=min(height, math.ceil(max(y for _, y in corners))),
        )
        if turned.right <= turned.left or turned.bottom <= turned.top:
            return None
        return turned

    turned = DocumentAnnotation(
        text=annotation.text,
        blocks=[
            BlockAnnotation(block.kind, block.text, turn(block.bbox))
            for block in annotation.blocks
        ],
        lines=[
            LineAnnotation(line.block, line.text, turn(line.bbox))
            for line in annotation.lines
        ],
        size=annotation.size,
        metadata={**annotation.metadata, "rotation": round(degrees, 3)},
    )
    return rotated, turned


def _rotate_point(
    x: float, y: float, centre: tuple[float, float], degrees: float
) -> tuple[float, float]:
    """Rotate one point about a centre, anticlockwise, in image coordinates."""
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    dx, dy = x - centre[0], y - centre[1]
    return (
        centre[0] + dx * cos + dy * sin,
        centre[1] - dx * sin + dy * cos,
    )


def _border_color(image: Image.Image) -> tuple[int, int, int]:
    """Return the page's edge colour, to fill in behind a rotation."""
    pixels = np.asarray(image.convert("RGB"))
    edges = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    median = np.median(edges, axis=0).astype(int)
    return (int(median[0]), int(median[1]), int(median[2]))


# -- photometry ------------------------------------------------------------


def _apply_paper_and_light(
    image: Image.Image, rng: random.Random, strength: float
) -> Image.Image:
    """Tint the paper, light it unevenly and jitter its contrast."""
    if strength <= 0:
        return image

    pixels = np.asarray(image).astype(np.float32)
    height, width = pixels.shape[:2]

    # Warm or neutral paper only: red stays, green drops a little, blue most.
    green = rng.uniform(0.95, 1.0)
    tint = np.array([1.0, green, rng.uniform(0.84, green)], dtype=np.float32)
    pixels *= 1 - (1 - tint) * strength

    noise_rng = np.random.default_rng(rng.randrange(2**31))
    coarse = (noise_rng.random((4, 4)) * 255).astype(np.uint8)
    gradient = (
        np.asarray(
            Image.fromarray(coarse).resize(
                (width, height), Image.Resampling.BICUBIC
            )
        ).astype(np.float32)
        / 255.0
    )
    pixels *= (1 - strength * rng.uniform(0.05, 0.18) * gradient)[..., None]

    contrast = 1 + strength * rng.uniform(-0.12, 0.12)
    brightness = strength * rng.uniform(-10, 10)
    pixels = (pixels - 128) * contrast + 128 + brightness
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def _apply_vignette(
    image: Image.Image, rng: random.Random, amount: float
) -> Image.Image:
    """Darken the page towards its edges, the way a flatbed lid does."""
    if amount <= 0:
        return image

    width, height = image.size
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = (xs / max(1, width - 1) - 0.5) * 2
    dy = (ys / max(1, height - 1) - 0.5) * 2
    falloff = np.sqrt(dx**2 + dy**2) / math.sqrt(2)

    strength = amount * rng.uniform(0.4, 1.0)
    mask = 1 - strength * falloff**2
    pixels = np.asarray(image).astype(np.float32) * mask[..., None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def _apply_specks(
    image: Image.Image, rng: random.Random, per_megapixel: float
) -> Image.Image:
    """Scatter dust and ink specks over the page."""
    if per_megapixel <= 0:
        return image

    width, height = image.size
    count = int(per_megapixel * (width * height) / 1_000_000)
    if count <= 0:
        return image

    pixels = np.asarray(image).copy()
    noise_rng = np.random.default_rng(rng.randrange(2**31))
    for _ in range(count):
        radius = int(noise_rng.integers(1, 4))
        x = int(noise_rng.integers(0, width))
        y = int(noise_rng.integers(0, height))
        shade = int(noise_rng.integers(20, 120))
        pixels[
            max(0, y - radius) : y + radius,
            max(0, x - radius) : x + radius,
        ] = shade
    return Image.fromarray(pixels)


def _apply_noise(
    image: Image.Image, rng: random.Random, deviation: float
) -> Image.Image:
    """Add sensor grain."""
    if deviation <= 0:
        return image

    noise_rng = np.random.default_rng(rng.randrange(2**31))
    width, height = image.size
    grain = noise_rng.normal(
        0, max(0.1, rng.uniform(deviation * 0.3, deviation)), (height, width, 1)
    )
    pixels = np.asarray(image).astype(np.float32) + grain
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def _apply_jpeg(
    image: Image.Image, rng: random.Random, quality: tuple[int, int]
) -> Image.Image:
    """Re-compress the page, leaving the blocking a real scan arrives with."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(*quality))
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")
