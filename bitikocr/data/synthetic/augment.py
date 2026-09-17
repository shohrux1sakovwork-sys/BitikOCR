"""Making two pages of the same form look like two different captures.

A generated page is clean: even lighting, square to the sensor, no grain.
Real archive pages are none of those things, and a recogniser trained only
on clean pages learns the cleanliness. These are the post-processing steps
that spoil a page realistically.

Spoiling is two steps. :func:`plan_augmentation` makes every random choice
up front — how the paper is tinted, where a fold runs, whether the page was
scanned or photographed — into an :class:`AugmentationPlan`, a small record
that can be stored beside the page and compared with another page's.
:func:`apply_plan` then carries the plan out, on the CPU with numpy or on a
GPU through :mod:`bitikocr.data.synthetic.augment_gpu`.

The steps run in the order the page met them:

1. **Paper** — a photocopy, a tint, a fold, stains and dust. They belong to
   the sheet, so they move with it in the next step.
2. **Geometry** — a scanner's skew, or a phone's perspective with the table
   showing round the page. This is the only step that moves the ink, so it
   carries the annotation with it: every outline goes through the same
   transform as the pixels, and every box is re-derived from the result.
3. **Capture** — uneven light, a hand's shadow, a lens vignette, a low
   resolution, defocus and sensor grain.
4. **Compression** — the JPEG blocking a page arrives with.

The page keeps its size throughout, so the ground truth stays in the same
coordinate space.
"""

from __future__ import annotations

import io
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageFilter

from bitikocr.data.models.geometry import BoundingBox, Polygon
from bitikocr.data.synthetic.annotation import (
    BlockAnnotation,
    DocumentAnnotation,
    LineAnnotation,
)

__all__ = [
    "AugmentationPlan",
    "AugmentationProfile",
    "AugmentationReport",
    "Backend",
    "Capture",
    "Stain",
    "apply_plan",
    "augment_page",
    "homography",
    "plan_augmentation",
    "plan_matrix",
    "rotate_page",
    "warp_page",
]

#: Where the pixels are pushed around.
Backend = Literal["cpu", "cuda"]

#: How a page reached the corpus, in the schema's words.
Capture = Literal["scanner", "camera"]

#: A 3x3 projective transform, as nested rows.
Matrix = tuple[tuple[float, float, float], ...]

#: Surfaces a photographed page lies on: a dark desk, a grey one, a white
#: one, a black cloth and a blue one.
_TABLES: tuple[tuple[int, int, int], ...] = (
    (72, 52, 36),
    (118, 116, 112),
    (212, 210, 204),
    (32, 32, 34),
    (58, 76, 104),
)

#: Colour a coffee or tea stain leaves, multiplied into the paper.
_STAIN_COLOUR = (0.66, 0.5, 0.3)


@dataclass(frozen=True)
class AugmentationReport:
    """What was actually done to one page.

    The corpus schema records capture conditions, and a recogniser can be
    evaluated by them, so a page has to be able to say how blurred, skewed
    and noisy it is rather than leaving it to be guessed.

    Args:
        blur: Whether a defocus blur was applied.
        rotation: The in-plane skew applied, in degrees.
        noise: How much grain was added, as a band.
        capture: What the page was captured with.
        perspective: Whether the page was photographed at an angle.
    """

    blur: bool = False
    rotation: float = 0.0
    noise: str = "low"
    capture: Capture = "scanner"
    perspective: bool = False

    @property
    def skew(self) -> bool:
        """Whether the page ended up off square at all."""
        return self.perspective or abs(self.rotation) >= 0.05

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serialisable form of the report."""
        return {
            "blur": self.blur,
            "rotation": round(self.rotation, 3),
            "skew": self.skew,
            "noise": self.noise,
            "capture": self.capture,
        }


@dataclass(frozen=True)
class AugmentationProfile:
    """How hard to spoil a page.

    The defaults describe a flatbed scan with light wear. The effects added
    for the corpus build — photographs, shadows, folds, stains, photocopies
    and low resolutions — are off unless asked for; :meth:`varied` asks for
    all of them.

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
        camera_share: Share of pages photographed rather than scanned.
        max_camera_rotation: Largest in-plane turn of a photographed page.
        shadow: Chance a shadow falls across the page.
        crease: Chance the page was folded.
        stain: Chance the page carries stains.
        photocopy: Chance the page is a photocopy rather than an original.
        low_resolution: Chance the page was captured at a low resolution.
    """

    strength: float = 1.0
    max_rotation: float = 0.7
    vignette: float = 0.35
    blur: float = 0.9
    noise: float = 7.0
    jpeg_quality: tuple[int, int] | None = (60, 93)
    speck_density: float = 60.0
    camera_share: float = 0.0
    max_camera_rotation: float = 3.0
    shadow: float = 0.0
    crease: float = 0.0
    stain: float = 0.0
    photocopy: float = 0.0
    low_resolution: float = 0.0

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

    @classmethod
    def varied(cls, strength: float = 1.0) -> AugmentationProfile:
        """Return the profile the corpus is built with.

        A quarter of the pages are photographed, and every kind of wear
        appears often enough to be learned without dominating the set.

        Args:
            strength: Scales every photometric effect.

        Returns:
            The profile.
        """
        return cls(
            strength=strength,
            camera_share=0.25,
            shadow=0.2,
            crease=0.2,
            stain=0.12,
            photocopy=0.1,
            low_resolution=0.15,
        )

    @property
    def is_empty(self) -> bool:
        """Whether this profile can change a page at all."""
        return self.strength <= 0 and self.max_rotation <= 0


@dataclass(frozen=True)
class Stain:
    """One stain on the paper, in fractions of the page.

    Args:
        x: Centre, across.
        y: Centre, down.
        rx: Radius across.
        ry: Radius down.
        alpha: How dark it is, 0.0 to 1.0.
    """

    x: float
    y: float
    rx: float
    ry: float
    alpha: float


@dataclass(frozen=True)
class AugmentationPlan:
    """Every random choice one page's spoiling takes, made in advance.

    Positions are fractions of the page, so a plan does not depend on the
    page's size. Seeds stand in for the full-page noise fields, which are
    drawn when the plan is applied.

    Args:
        strength: The profile strength the plan was drawn at.
        capture: What the page was captured with.
        rotation: In-plane turn in degrees, positive anticlockwise.
        corners: Where a photographed page's corners land, clockwise from
            top-left, as fractions of the frame; None for a scan.
        table: What shows round a photographed page.
        photocopy: Contrast exponent of a photocopy; 0.0 for an original.
        tint: Per-channel multiplier the paper colour gets.
        crease: ``(vertical, position, slope, darkness)`` of a fold, or None.
        stains: The stains on the paper.
        specks: Dust specks per megapixel.
        light: How unevenly the page is lit.
        contrast: Contrast multiplier.
        brightness: Brightness offset.
        shadow: ``(angle, offset, darkness, softness)`` of a shadow, or None.
        vignette: How much darker the edges get.
        resolution: Scale the page was captured at; 1.0 for full.
        blur: Defocus radius in pixels; 0.0 for none.
        noise: Sensor grain as a pixel standard deviation.
        jpeg_quality: Re-compression quality, or None.
        seed: Seeds the noise fields drawn when the plan is applied.
    """

    strength: float = 0.0
    capture: Capture = "scanner"
    rotation: float = 0.0
    corners: tuple[tuple[float, float], ...] | None = None
    table: tuple[int, int, int] | None = None
    photocopy: float = 0.0
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    crease: tuple[bool, float, float, float] | None = None
    stains: tuple[Stain, ...] = ()
    specks: float = 0.0
    light: float = 0.0
    contrast: float = 1.0
    brightness: float = 0.0
    shadow: tuple[float, float, float, float] | None = None
    vignette: float = 0.0
    resolution: float = 1.0
    blur: float = 0.0
    noise: float = 0.0
    jpeg_quality: int | None = None
    seed: int = 0
    effects: tuple[str, ...] = field(default=())

    @property
    def is_identity(self) -> bool:
        """Whether applying the plan would leave the page as it is."""
        return self == AugmentationPlan(seed=self.seed)

    @property
    def moves_ink(self) -> bool:
        """Whether the plan moves the ink, and so the annotation."""
        return self.corners is not None or abs(self.rotation) >= 0.01

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the plan."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AugmentationPlan:
        """Rebuild a plan from its JSON form.

        Args:
            payload: A decoded plan.

        Returns:
            The plan.
        """
        data = dict(payload)
        if data.get("corners") is not None:
            data["corners"] = tuple(tuple(c) for c in data["corners"])
        for key in ("table", "tint", "crease", "shadow"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        data["stains"] = tuple(Stain(**s) for s in data.get("stains", ()))
        data["effects"] = tuple(data.get("effects", ()))
        return cls(**data)


# -- planning --------------------------------------------------------------


def plan_augmentation(
    rng: random.Random, profile: AugmentationProfile | None = None
) -> AugmentationPlan:
    """Make every random choice one page's spoiling needs.

    Args:
        rng: Random source. The same seed plans the same spoiling.
        profile: How hard to spoil; a default profile when omitted.

    Returns:
        The plan, ready for :func:`apply_plan`.
    """
    profile = profile or AugmentationProfile()
    if profile.is_empty:
        return AugmentationPlan()
    strength = profile.strength
    effects: list[str] = []

    camera = rng.random() < profile.camera_share
    capture: Capture = "camera" if camera else "scanner"
    limit = profile.max_camera_rotation if camera else profile.max_rotation
    rotation = rng.uniform(-limit, limit) if limit > 0 else 0.0

    corners: tuple[tuple[float, float], ...] | None = None
    table: tuple[int, int, int] | None = None
    if camera:
        corners = _sample_corners(rng)
        table = rng.choice(_TABLES)
        effects.append("perspective")

    photocopy = 0.0
    if not camera and rng.random() < profile.photocopy:
        photocopy = rng.uniform(1.4, 2.4)
        effects.append("photocopy")

    # Warm or neutral paper only: red stays, green drops a little, blue
    # most. A photocopy comes out on fresh white paper.
    tint = (1.0, 1.0, 1.0)
    if not photocopy:
        green = rng.uniform(0.95, 1.0)
        tint = (1.0, green, rng.uniform(0.84, green))

    crease = None
    if rng.random() < profile.crease:
        position = rng.choice((0.33, 0.5, 0.67)) + rng.uniform(-0.03, 0.03)
        crease = (
            rng.random() < 0.3,
            position,
            rng.uniform(-0.02, 0.02),
            rng.uniform(0.08, 0.2),
        )
        effects.append("crease")

    stains: tuple[Stain, ...] = ()
    if rng.random() < profile.stain:
        stains = tuple(
            Stain(
                x=rng.uniform(0.05, 0.95),
                y=rng.uniform(0.05, 0.95),
                rx=rng.uniform(0.03, 0.12),
                ry=rng.uniform(0.03, 0.12),
                alpha=rng.uniform(0.12, 0.35),
            )
            for _ in range(rng.randint(1, 3))
        )
        effects.append("stain")

    shadow = None
    if rng.random() < (profile.shadow * (2.5 if camera else 0.4)):
        shadow = (
            rng.uniform(0, 2 * math.pi),
            rng.uniform(0.1, 0.45),
            rng.uniform(0.15, 0.4),
            rng.uniform(0.02, 0.12),
        )
        effects.append("shadow")

    resolution = 1.0
    if rng.random() < profile.low_resolution:
        resolution = rng.uniform(0.45, 0.8)
        effects.append("low_resolution")

    blur = 0.0
    if profile.blur > 0 and rng.random() < 0.6:
        blur = rng.uniform(0.2, profile.blur) * strength
        if camera:
            blur *= 1.4
    if blur > 0:
        effects.append("blur")

    noise = 0.0
    if profile.noise > 0:
        noise = max(0.1, rng.uniform(0.3, 1.0) * profile.noise * strength)
        if camera:
            noise *= 1.3

    jpeg_quality = None
    if profile.jpeg_quality is not None:
        jpeg_quality = rng.randint(*profile.jpeg_quality)

    return AugmentationPlan(
        strength=strength,
        capture=capture,
        rotation=rotation,
        corners=corners,
        table=table,
        photocopy=photocopy,
        tint=tint,
        crease=crease,
        stains=stains,
        specks=profile.speck_density * strength,
        light=strength * rng.uniform(0.05, 0.18) * (1.8 if camera else 1.0),
        contrast=1 + strength * rng.uniform(-0.12, 0.12),
        brightness=strength * rng.uniform(-10, 10),
        shadow=shadow,
        vignette=profile.vignette * strength * rng.uniform(0.4, 1.0),
        resolution=resolution,
        blur=blur,
        noise=noise,
        jpeg_quality=jpeg_quality,
        seed=rng.randrange(2**31),
        effects=tuple(effects),
    )


def _sample_corners(
    rng: random.Random,
) -> tuple[tuple[float, float], ...]:
    """Place a photographed page in the frame, a little off square."""
    scale = rng.uniform(0.82, 0.95)
    margin_x = (1 - scale) / 2
    margin_y = (1 - scale) / 2
    shift_x = rng.uniform(-margin_x, margin_x) * 0.5
    shift_y = rng.uniform(-margin_y, margin_y) * 0.5
    # How far the far edge recedes: the camera leans one way.
    lean = rng.uniform(0.0, 0.05)
    side = rng.choice(("top", "bottom", "left", "right"))
    base = [
        (margin_x, margin_y),
        (1 - margin_x, margin_y),
        (1 - margin_x, 1 - margin_y),
        (margin_x, 1 - margin_y),
    ]
    squeeze = {
        "top": ((lean, 0), (-lean, 0), (0, 0), (0, 0)),
        "bottom": ((0, 0), (0, 0), (-lean, 0), (lean, 0)),
        "left": ((0, lean), (0, 0), (0, 0), (0, -lean)),
        "right": ((0, 0), (0, lean), (0, -lean), (0, 0)),
    }[side]
    corners = []
    for (x, y), (dx, dy) in zip(base, squeeze):
        jitter = 0.02
        corners.append(
            (
                min(
                    1.0,
                    max(0.0, x + dx + shift_x + rng.uniform(-jitter, jitter)),
                ),
                min(
                    1.0,
                    max(0.0, y + dy + shift_y + rng.uniform(-jitter, jitter)),
                ),
            )
        )
    return tuple(corners)


# -- applying --------------------------------------------------------------


def augment_page(
    image: Image.Image,
    annotation: DocumentAnnotation,
    rng: random.Random,
    profile: AugmentationProfile | None = None,
    backend: Backend = "cpu",
) -> tuple[Image.Image, DocumentAnnotation, AugmentationReport]:
    """Spoil a rendered page the way a scanner or a camera and time would.

    Args:
        image: The clean rendered page.
        annotation: Its ground truth.
        rng: Random source. The same seed spoils the page the same way.
        profile: How hard to spoil it; a default profile when omitted.
        backend: Where to push the pixels.

    Returns:
        ``(image, annotation, report)``. The annotation is a new object
        whenever a step moved the ink, and the report says what was done.
    """
    profile = profile or AugmentationProfile()
    if profile.is_empty:
        return image, annotation, AugmentationReport()
    plan = plan_augmentation(rng, profile)
    return apply_plan(image, annotation, plan, backend)


def apply_plan(
    image: Image.Image,
    annotation: DocumentAnnotation,
    plan: AugmentationPlan,
    backend: Backend = "cpu",
) -> tuple[Image.Image, DocumentAnnotation, AugmentationReport]:
    """Carry out a plan on one page.

    Both backends draw the same effects from the same plan. Their noise
    fields come from different generators, so the two do not agree pixel
    for pixel, but each is reproducible on its own.

    Args:
        image: The clean rendered page.
        annotation: Its ground truth.
        plan: What to do, from :func:`plan_augmentation`.
        backend: ``"cpu"`` for numpy, ``"cuda"`` for the GPU.

    Returns:
        ``(image, annotation, report)``.
    """
    if plan.is_identity:
        return image, annotation, AugmentationReport()

    image = image.convert("RGB")
    matrix = plan_matrix(plan, image.size)
    fill = plan.table
    if plan.moves_ink and fill is None:
        fill = _border_color(image)

    if backend == "cuda":
        from bitikocr.data.synthetic import augment_gpu

        image = augment_gpu.apply_pixels(image, plan, matrix, fill)
    else:
        image = _apply_paper(image, plan)
        if plan.moves_ink:
            assert fill is not None
            image = _warp_pixels(image, plan, matrix, fill)
        image = _apply_capture(image, plan)

    if plan.moves_ink:
        annotation = _transform_annotation(
            annotation,
            image.size,
            lambda x, y: _project(matrix, x, y),
            plan.rotation,
        )
    if plan.jpeg_quality is not None:
        image = _apply_jpeg(image, plan.jpeg_quality)

    return (
        image,
        annotation,
        AugmentationReport(
            blur=plan.blur > 0,
            rotation=plan.rotation,
            noise=_noise_band(plan.noise),
            capture=plan.capture,
            perspective=plan.corners is not None,
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


def homography(
    source: Sequence[tuple[float, float]],
    target: Sequence[tuple[float, float]],
) -> Matrix:
    """Solve the projective transform taking four points onto four others.

    Args:
        source: Four points, no three on a line.
        target: Where each of them goes.

    Returns:
        The 3x3 transform, normalised so its last entry is 1.
    """
    rows = []
    values = []
    for (x, y), (u, v) in zip(source, target):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.extend([u, v])
    solved = np.linalg.solve(np.array(rows, float), np.array(values, float))
    h = [*solved.tolist(), 1.0]
    return (tuple(h[0:3]), tuple(h[3:6]), tuple(h[6:9]))


def plan_matrix(plan: AugmentationPlan, size: tuple[int, int]) -> Matrix:
    """Return the transform a plan moves the page's ink with.

    Args:
        plan: The plan.
        size: The page's ``(width, height)``.

    Returns:
        The 3x3 transform from clean page to spoiled page.
    """
    width, height = size
    turn = _rotation_matrix(plan.rotation, (width / 2, height / 2))
    if plan.corners is None:
        return turn
    page = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    placed = [(x * width, y * height) for x, y in plan.corners]
    return _compose(turn, homography(page, placed))


def _rotation_matrix(degrees: float, centre: tuple[float, float]) -> Matrix:
    """Return the anticlockwise rotation about a centre, in image space."""
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    cx, cy = centre
    return (
        (cos, sin, cx - cos * cx - sin * cy),
        (-sin, cos, cy + sin * cx - cos * cy),
        (0.0, 0.0, 1.0),
    )


def _compose(outer: Matrix, inner: Matrix) -> Matrix:
    """Return the transform applying ``inner`` first, then ``outer``."""
    product = np.array(outer, float) @ np.array(inner, float)
    product /= product[2, 2]
    return tuple(tuple(row) for row in product.tolist())


def _project(matrix: Matrix, x: float, y: float) -> tuple[float, float]:
    """Send one point through a projective transform."""
    (a, b, c), (d, e, f), (g, h, i) = matrix
    w = g * x + h * y + i
    return (a * x + b * y + c) / w, (d * x + e * y + f) / w


def _inverse(matrix: Matrix) -> Matrix:
    """Return a transform's inverse, normalised."""
    inverse = np.linalg.inv(np.array(matrix, float))
    inverse /= inverse[2, 2]
    return tuple(tuple(row) for row in inverse.tolist())


def warp_page(
    image: Image.Image,
    annotation: DocumentAnnotation,
    matrix: Matrix,
    fill: tuple[int, int, int],
) -> tuple[Image.Image, DocumentAnnotation]:
    """Move a page's ink through a projective transform, truth and all.

    Args:
        image: The page.
        annotation: Its ground truth.
        matrix: The transform, from clean page to moved page.
        fill: Colour of whatever the page no longer covers.

    Returns:
        ``(image, annotation)`` with both moved.
    """
    moved = image.convert("RGB").transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        _perspective_data(matrix),
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )
    turned = _transform_annotation(
        annotation, image.size, lambda x, y: _project(matrix, x, y), 0.0
    )
    return moved, turned


def _perspective_data(matrix: Matrix) -> tuple[float, ...]:
    """Return PIL's eight coefficients, which map output back to input."""
    inverse = _inverse(matrix)
    return tuple(value for row in inverse for value in row)[:8]


def rotate_page(
    image: Image.Image, annotation: DocumentAnnotation, degrees: float
) -> tuple[Image.Image, DocumentAnnotation]:
    """Skew a page as if it were laid crookedly on the scanner.

    The page keeps its size, so the ground truth stays in the same
    coordinate space. A block's outline is turned with the ink and kept as
    the tilted quadrilateral it now is; its box is re-derived from those
    corners, which grows it slightly — the cost of keeping it axis-aligned.

    Args:
        image: The page to skew.
        annotation: Its ground truth.
        degrees: Rotation, positive anticlockwise.

    Returns:
        ``(image, annotation)`` with both rotated.
    """
    if abs(degrees) < 0.01:
        return image, annotation

    rotated = _rotate_pixels(image, degrees, _border_color(image))
    matrix = _rotation_matrix(degrees, (image.width / 2, image.height / 2))
    turned = _transform_annotation(
        annotation,
        image.size,
        lambda x, y: _project(matrix, x, y),
        degrees,
    )
    return rotated, turned


def _rotate_pixels(
    image: Image.Image, degrees: float, fill: tuple[int, int, int]
) -> Image.Image:
    """Turn a page's pixels about its centre, keeping its size."""
    return image.rotate(
        degrees,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )


def _warp_pixels(
    image: Image.Image,
    plan: AugmentationPlan,
    matrix: Matrix,
    fill: tuple[int, int, int],
) -> Image.Image:
    """Move the pixels the way the plan moves the ink."""
    if plan.corners is None:
        return _rotate_pixels(image, plan.rotation, fill)
    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        _perspective_data(matrix),
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )


def _transform_annotation(
    annotation: DocumentAnnotation,
    size: tuple[int, int],
    move: Callable[[float, float], tuple[float, float]],
    degrees: float,
) -> DocumentAnnotation:
    """Carry a page's ground truth through a point transform."""
    width, height = size

    def move_points(outline: Polygon) -> list[tuple[float, float]]:
        return [move(x, y) for x, y in outline]

    def move_polygon(outline: Polygon | None) -> Polygon | None:
        if outline is None:
            return None
        # Clamped to the page: a corner carried past the edge describes
        # ink that was clipped away, not ink that is there.
        return tuple(
            (
                min(width, max(0, round(x))),
                min(height, max(0, round(y))),
            )
            for x, y in move_points(outline)
        )

    def move_box(box: BoundingBox | None) -> BoundingBox | None:
        if box is None:
            return None
        corners = move_points(box.to_polygon())
        # Round outwards so the box never clips the ink it follows, then
        # trim to the page: a move can carry ink off the edge, and a box
        # that lands entirely outside describes nothing. A projective map
        # keeps a box's image convex, so its corners bound all of it.
        moved = BoundingBox(
            left=max(0, math.floor(min(x for x, _ in corners))),
            top=max(0, math.floor(min(y for _, y in corners))),
            right=min(width, math.ceil(max(x for x, _ in corners))),
            bottom=min(height, math.ceil(max(y for _, y in corners))),
        )
        if moved.right <= moved.left or moved.bottom <= moved.top:
            return None
        return moved

    return DocumentAnnotation(
        text=annotation.text,
        blocks=[
            _move_block(block, move_box, move_polygon)
            for block in annotation.blocks
        ],
        lines=[
            LineAnnotation(line.block, line.text, move_box(line.bbox))
            for line in annotation.lines
        ],
        size=annotation.size,
        metadata={**annotation.metadata, "rotation": round(degrees, 3)},
    )


def _move_block(
    block: BlockAnnotation,
    move_box: Callable[[BoundingBox | None], BoundingBox | None],
    move_polygon: Callable[[Polygon | None], Polygon | None],
) -> BlockAnnotation:
    """Move one block, dropping its outline if it left the page."""
    box = move_box(block.bbox)
    outline = move_polygon(block.outline) if box is not None else None
    return BlockAnnotation(block.kind, block.text, box, outline)


def _border_color(image: Image.Image) -> tuple[int, int, int]:
    """Return the page's edge colour, to fill in behind a rotation."""
    pixels = np.asarray(image.convert("RGB"))
    edges = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    median = np.median(edges, axis=0).astype(int)
    return (int(median[0]), int(median[1]), int(median[2]))


# -- photometry on the CPU -------------------------------------------------


def _apply_paper(image: Image.Image, plan: AugmentationPlan) -> Image.Image:
    """Photocopy, tint, fold, stain and dust the sheet."""
    pixels: np.ndarray = np.asarray(image).astype(np.float32)
    rng = np.random.default_rng(plan.seed)

    if plan.photocopy:
        gray = pixels @ np.array([0.299, 0.587, 0.114], np.float32)
        gray = 255 * (gray / 255) ** plan.photocopy
        pixels = np.repeat(gray[..., None], 3, axis=2)

    strength = plan.strength
    pixels *= 1 - (1 - np.array(plan.tint, np.float32)) * strength

    if plan.crease is not None:
        pixels = _crease(pixels, plan.crease)
    for stain in plan.stains:
        _stain(pixels, stain)
    _specks(pixels, plan.specks, rng)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def _crease(
    pixels: np.ndarray, crease: tuple[bool, float, float, float]
) -> np.ndarray:
    """Draw a fold: a dark line with a lit ridge beside it."""
    vertical, position, slope, darkness = crease
    height, width = pixels.shape[:2]
    across, along = (width, height) if vertical else (height, width)
    t = np.arange(along, dtype=np.float32)
    s = np.arange(across, dtype=np.float32)
    centre = position * across + slope * (t - along / 2)
    distance = (s[None, :] - centre[:, None]) / across
    shade = darkness * np.exp(-((distance / 0.0015) ** 2))
    ridge = 0.5 * darkness * np.exp(-(((distance - 0.004) / 0.004) ** 2))
    if not vertical:
        # Laid out as (column, row) above; turn it back to (row, column).
        shade, ridge = shade.T, ridge.T
    pixels = pixels * (1 - shade[..., None])
    return pixels + (255 - pixels) * ridge[..., None]


def _stain(pixels: np.ndarray, stain: Stain) -> None:
    """Soak one stain into the paper, darker at its dried rim."""
    height, width = pixels.shape[:2]
    top, bottom, left, right = _stain_window(stain, width, height)
    if bottom <= top or right <= left:
        return
    ys, xs = np.mgrid[top:bottom, left:right].astype(np.float32)
    alpha = _stain_alpha(xs, ys, stain, width, height)
    colour = np.array(_STAIN_COLOUR, np.float32)
    window = pixels[top:bottom, left:right]
    window *= 1 - alpha[..., None] * (1 - colour)


def _stain_window(
    stain: Stain, width: int, height: int
) -> tuple[int, int, int, int]:
    """Return the rows and columns a stain can reach."""
    reach_x = stain.rx * width * 1.3
    reach_y = stain.ry * height * 1.3
    return (
        max(0, int(stain.y * height - reach_y)),
        min(height, int(stain.y * height + reach_y) + 1),
        max(0, int(stain.x * width - reach_x)),
        min(width, int(stain.x * width + reach_x) + 1),
    )


def _stain_alpha(
    xs: Any, ys: Any, stain: Stain, width: int, height: int
) -> Any:
    """Return a stain's opacity over a window; works on arrays and tensors."""
    dx = (xs - stain.x * width) / (stain.rx * width)
    dy = (ys - stain.y * height) / (stain.ry * height)
    radius = (dx * dx + dy * dy) ** 0.5
    inside = 1 / (1 + (2.718281828 ** ((radius - 1) / 0.08)))
    rim = 2.718281828 ** (-(((radius - 1) / 0.05) ** 2))
    return stain.alpha * (0.6 * inside + rim)


def _specks(
    pixels: np.ndarray, per_megapixel: float, rng: np.random.Generator
) -> None:
    """Scatter dust and ink specks over the page."""
    height, width = pixels.shape[:2]
    for x, y, radius, shade in _speck_positions(
        per_megapixel, width, height, rng
    ):
        pixels[
            max(0, y - radius) : y + radius,
            max(0, x - radius) : x + radius,
        ] = shade


def _speck_positions(
    per_megapixel: float,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    """Return ``(x, y, radius, shade)`` for every speck on a page."""
    count = int(per_megapixel * (width * height) / 1_000_000)
    if count <= 0:
        return []
    radii = rng.integers(1, 4, count)
    xs = rng.integers(0, width, count)
    ys = rng.integers(0, height, count)
    shades = rng.integers(20, 120, count)
    return [
        (int(x), int(y), int(r), int(s))
        for x, y, r, s in zip(xs, ys, radii, shades)
    ]


def _apply_capture(image: Image.Image, plan: AugmentationPlan) -> Image.Image:
    """Light, shade, vignette, down-sample, blur and grain the capture."""
    pixels: np.ndarray = np.asarray(image).astype(np.float32)
    height, width = pixels.shape[:2]
    rng = np.random.default_rng(plan.seed + 1)

    if plan.light > 0:
        coarse = (rng.random((4, 4)) * 255).astype(np.uint8)
        gradient = (
            np.asarray(
                Image.fromarray(coarse).resize(
                    (width, height), Image.Resampling.BICUBIC
                )
            ).astype(np.float32)
            / 255.0
        )
        pixels *= (1 - plan.light * gradient)[..., None]

    pixels = (pixels - 128) * plan.contrast + 128 + plan.brightness

    if plan.shadow is not None or plan.vignette > 0:
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        mask = np.ones((height, width), np.float32)
        if plan.shadow is not None:
            mask *= _shadow_mask(xs, ys, plan.shadow, width, height, np.exp)
        if plan.vignette > 0:
            mask *= _vignette_mask(xs, ys, plan.vignette, width, height)
        pixels *= mask[..., None]

    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    if plan.resolution < 1.0:
        small = (
            max(1, round(width * plan.resolution)),
            max(1, round(height * plan.resolution)),
        )
        image = image.resize(small, Image.Resampling.BOX).resize(
            (width, height), Image.Resampling.BILINEAR
        )
    if plan.blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(plan.blur))
    if plan.noise > 0:
        grain = rng.normal(0, plan.noise, (height, width, 1))
        noisy = np.asarray(image).astype(np.float32) + grain
        image = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    return image


def _shadow_mask(
    xs: Any,
    ys: Any,
    shadow: tuple[float, float, float, float],
    width: int,
    height: int,
    exp: Callable[[Any], Any],
) -> Any:
    """Return how much light a shadow leaves; works on arrays and tensors."""
    angle, offset, darkness, softness = shadow
    diagonal = math.hypot(width, height)
    along = (
        (xs - width / 2) * math.cos(angle) + (ys - height / 2) * math.sin(angle)
    ) / diagonal
    covered = 1 / (1 + exp(-(along - offset) / softness))
    return 1 - darkness * covered


def _vignette_mask(
    xs: Any, ys: Any, amount: float, width: int, height: int
) -> Any:
    """Return how much light reaches each pixel under a vignette."""
    dx = (xs / max(1, width - 1) - 0.5) * 2
    dy = (ys / max(1, height - 1) - 0.5) * 2
    falloff = (dx * dx + dy * dy) / 2
    return 1 - amount * falloff


def _apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Re-compress the page, leaving the blocking a real capture has."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")
