"""Pure image helpers with no knowledge of documents or handwriting."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter

from bitikocr.models.geometry import BoundingBox

__all__ = ["alpha_bounding_box", "light_augment"]


def alpha_bounding_box(
    image: Image.Image, offset: tuple[int, int] = (0, 0)
) -> BoundingBox | None:
    """Find the tight box around the opaque pixels of an RGBA image.

    Args:
        image: An RGBA image whose alpha channel marks the drawn ink.
        offset: Position the image is pasted at, added to the result.

    Returns:
        The box in page coordinates, or None if the image is fully
        transparent.
    """
    box = image.getchannel("A").getbbox()
    if box is None:
        return None
    return BoundingBox(
        left=box[0] + offset[0],
        top=box[1] + offset[1],
        right=box[2] + offset[0],
        bottom=box[3] + offset[1],
    )


def light_augment(
    image: Image.Image, rng: random.Random, strength: float = 1.0
) -> Image.Image:
    """Apply photometric variation that keeps every bounding box valid.

    Paper tint, uneven lighting, a slight blur, sensor noise and contrast
    jitter are applied. Geometric distortions (rotation, warping) are
    deliberately excluded: they would also have to transform the ground truth.

    Args:
        image: The rendered page.
        rng: Random source driving every sampled parameter.
        strength: Scales the whole effect; 0.0 leaves the image untouched.

    Returns:
        A new RGB image the same size as the input.
    """
    image = image.convert("RGB")
    width, height = image.size
    pixels = np.asarray(image).astype(np.float32)

    # Warm or neutral paper only: red stays, green drops a little, blue most.
    green = rng.uniform(0.95, 1.0)
    tint = np.array([1.0, green, rng.uniform(0.84, green)])
    pixels *= 1 - (1 - tint) * strength

    # Uneven lighting: a smooth low-frequency gradient over the whole page.
    noise_rng = np.random.default_rng(rng.randrange(2**31))
    coarse = noise_rng.random((4, 4)).astype(np.float32)
    coarse_image = Image.fromarray((coarse * 255).astype(np.uint8))
    gradient = (
        np.asarray(
            coarse_image.resize((width, height), Image.Resampling.BICUBIC)
        )
        / 255.0
    )
    pixels *= (1 - strength * rng.uniform(0.05, 0.18) * gradient)[..., None]

    contrast = 1 + strength * rng.uniform(-0.12, 0.12)
    brightness = strength * rng.uniform(-10, 10)
    pixels = (pixels - 128) * contrast + 128 + brightness

    out = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        out = out.filter(
            ImageFilter.GaussianBlur(strength * rng.uniform(0.3, 0.9))
        )

    grain = noise_rng.normal(
        0, strength * rng.uniform(2, 7), (height, width, 1)
    ).astype(np.float32)
    noisy = np.asarray(out).astype(np.float32) + grain
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
