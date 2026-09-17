"""Spoiling pages on a GPU.

The effects in :mod:`bitikocr.data.synthetic.augment` touch every pixel of
a page several times, and on the CPU they cost more than drawing the
handwriting does. This module carries out the same plan with torch, so a
corpus build can hand the pixel work to a graphics card.

It is imported only when the ``cuda`` backend is asked for, so the rest of
the generator runs without torch installed. The geometry — and so the
annotation — is computed by the caller from the same plan; this module
only moves pixels.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from bitikocr.data.synthetic.augment import (
    _STAIN_COLOUR,
    AugmentationPlan,
    Matrix,
    _inverse,
    _shadow_mask,
    _speck_positions,
    _stain_alpha,
    _stain_window,
    _vignette_mask,
)

__all__ = ["apply_pixels", "cuda_available"]

_LUMA = (0.299, 0.587, 0.114)


def cuda_available() -> bool:
    """Say whether torch is installed and can see a CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@lru_cache(maxsize=1)
def _torch() -> Any:
    """Import torch once, failing with a message that says what to install."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "The cuda backend needs torch: install the 'gpu' extra"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("The cuda backend needs a CUDA device")
    return torch


def apply_pixels(
    image: Image.Image,
    plan: AugmentationPlan,
    matrix: Matrix,
    fill: tuple[int, int, int] | None,
) -> Image.Image:
    """Carry out a plan's paper, geometry and capture steps on the GPU.

    Args:
        image: The clean RGB page.
        plan: What to do.
        matrix: The transform the plan moves the ink with.
        fill: Colour behind a moved page; unused when nothing moves.

    Returns:
        The spoiled page, not yet re-compressed.
    """
    torch = _torch()
    with torch.inference_mode():
        pixels = (
            torch.from_numpy(np.array(image))
            .to("cuda", non_blocking=True)
            .permute(2, 0, 1)
            .float()
        )
        generator = torch.Generator(device="cuda")
        generator.manual_seed(plan.seed)

        pixels = _paper(torch, pixels, plan, generator)
        if plan.moves_ink:
            assert fill is not None
            pixels = _warp(torch, pixels, matrix, fill)
        pixels = _capture(torch, pixels, plan, generator)

        out = pixels.clamp_(0, 255).round_().to(torch.uint8)
        array = out.permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(array)


def _grid(torch: Any, height: int, width: int) -> tuple[Any, Any]:
    """Return pixel-centre coordinates as two ``(H, W)`` tensors."""
    ys = torch.arange(height, device="cuda", dtype=torch.float32)
    xs = torch.arange(width, device="cuda", dtype=torch.float32)
    return torch.meshgrid(ys, xs, indexing="ij")[::-1]


def _paper(
    torch: Any, pixels: Any, plan: AugmentationPlan, generator: Any
) -> Any:
    """Photocopy, tint, fold, stain and dust the sheet."""
    _, height, width = pixels.shape
    if plan.photocopy:
        luma = torch.tensor(_LUMA, device="cuda").view(3, 1, 1)
        gray = (pixels * luma).sum(0, keepdim=True)
        gray = 255 * (gray / 255) ** plan.photocopy
        pixels = gray.expand(3, -1, -1).clone()

    tint = torch.tensor(plan.tint, device="cuda").view(3, 1, 1)
    pixels *= 1 - (1 - tint) * plan.strength

    if plan.crease is not None:
        pixels = _crease(torch, pixels, plan.crease)
    colour = torch.tensor(_STAIN_COLOUR, device="cuda").view(3, 1, 1)
    for stain in plan.stains:
        top, bottom, left, right = _stain_window(stain, width, height)
        if bottom <= top or right <= left:
            continue
        ys = torch.arange(top, bottom, device="cuda", dtype=torch.float32)
        xs = torch.arange(left, right, device="cuda", dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        alpha = _stain_alpha(xx, yy, stain, width, height)
        pixels[:, top:bottom, left:right] *= 1 - alpha * (1 - colour)

    rng = np.random.default_rng(plan.seed)
    for x, y, radius, shade in _speck_positions(
        plan.specks, width, height, rng
    ):
        pixels[
            :,
            max(0, y - radius) : y + radius,
            max(0, x - radius) : x + radius,
        ] = shade
    return pixels


def _crease(
    torch: Any, pixels: Any, crease: tuple[bool, float, float, float]
) -> Any:
    """Draw a fold: a dark line with a lit ridge beside it."""
    vertical, position, slope, darkness = crease
    _, height, width = pixels.shape
    xs, ys = _grid(torch, height, width)
    across, s, t, along = (
        (width, xs, ys, height) if vertical else (height, ys, xs, width)
    )
    centre = position * across + slope * (t - along / 2)
    distance = (s - centre) / across
    shade = darkness * torch.exp(-((distance / 0.0015) ** 2))
    ridge = 0.5 * darkness * torch.exp(-(((distance - 0.004) / 0.004) ** 2))
    pixels = pixels * (1 - shade)
    return pixels + (255 - pixels) * ridge


def _warp(
    torch: Any,
    pixels: Any,
    matrix: Matrix,
    fill: tuple[int, int, int],
) -> Any:
    """Resample the page through the plan's transform."""
    _, height, width = pixels.shape
    xs, ys = _grid(torch, height, width)
    (a, b, c), (d, e, f), (g, h, i) = _inverse(matrix)
    w = g * xs + h * ys + i
    sx = (a * xs + b * ys + c) / w
    sy = (d * xs + e * ys + f) / w
    grid = torch.stack(
        ((2 * sx + 1) / width - 1, (2 * sy + 1) / height - 1), dim=-1
    ).unsqueeze(0)

    functional = torch.nn.functional
    moved = functional.grid_sample(
        pixels.unsqueeze(0),
        grid,
        mode="bicubic",
        padding_mode="border",
        align_corners=False,
    )[0]
    # How much of each output pixel the page covers, so its edge blends
    # into the surface behind it instead of stepping.
    cover = functional.grid_sample(
        torch.ones((1, 1, height, width), device="cuda"),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0]
    backing = torch.tensor(fill, device="cuda", dtype=torch.float32)
    return moved * cover + backing.view(3, 1, 1) * (1 - cover)


def _capture(
    torch: Any, pixels: Any, plan: AugmentationPlan, generator: Any
) -> Any:
    """Light, shade, vignette, down-sample, blur and grain the capture."""
    functional = torch.nn.functional
    _, height, width = pixels.shape

    if plan.light > 0:
        coarse = torch.rand((1, 1, 4, 4), device="cuda", generator=generator)
        gradient = functional.interpolate(
            coarse, size=(height, width), mode="bicubic", align_corners=False
        )[0].clamp_(0, 1)
        pixels *= 1 - plan.light * gradient

    pixels = (pixels - 128) * plan.contrast + 128 + plan.brightness

    if plan.shadow is not None or plan.vignette > 0:
        xs, ys = _grid(torch, height, width)
        mask = torch.ones((height, width), device="cuda")
        if plan.shadow is not None:
            mask *= _shadow_mask(xs, ys, plan.shadow, width, height, torch.exp)
        if plan.vignette > 0:
            mask *= _vignette_mask(xs, ys, plan.vignette, width, height)
        pixels *= mask

    pixels = pixels.clamp(0, 255)
    if plan.resolution < 1.0:
        small = (
            max(1, round(height * plan.resolution)),
            max(1, round(width * plan.resolution)),
        )
        pixels = functional.interpolate(
            pixels.unsqueeze(0), size=small, mode="area"
        )
        pixels = functional.interpolate(
            pixels, size=(height, width), mode="bilinear", align_corners=False
        )[0]
    if plan.blur > 0:
        pixels = _gaussian_blur(torch, pixels, plan.blur)
    if plan.noise > 0:
        grain = torch.randn(
            (1, height, width), device="cuda", generator=generator
        )
        pixels = pixels + grain * plan.noise
    return pixels


def _gaussian_blur(torch: Any, pixels: Any, sigma: float) -> Any:
    """Blur separably, the way PIL's Gaussian blur does."""
    radius = max(1, math.ceil(sigma * 3))
    offsets = torch.arange(
        -radius, radius + 1, device="cuda", dtype=torch.float32
    )
    kernel = torch.exp(-(offsets**2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    functional = torch.nn.functional
    batch = pixels.unsqueeze(1)
    batch = functional.pad(batch, (radius, radius, radius, radius), "replicate")
    batch = functional.conv2d(batch, kernel.view(1, 1, 1, -1))
    batch = functional.conv2d(batch, kernel.view(1, 1, -1, 1))
    return batch[:, 0]
