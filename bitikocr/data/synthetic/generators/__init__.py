"""Document generators and the registry that resolves them by name.

Adding a document type means adding a module here and one entry in
:data:`GENERATOR_TYPES`. Nothing else in the project needs to change.

Documents that fill a pre-printed form share :class:`FormGenerator` and
differ only in which measured layout they fill, so a new certificate is
usually a layout asset plus a four-line subclass.
"""

from __future__ import annotations

from pathlib import Path

from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic.generators.ariza import ArizaGenerator
from bitikocr.data.synthetic.generators.base import (
    DEFAULT_INK_STRENGTH,
    DocumentGenerator,
    FieldValues,
    SyntheticDocument,
)
from bitikocr.data.synthetic.generators.birth_certificate import (
    BirthCertificateGenerator,
)
from bitikocr.data.synthetic.generators.death_certificate import (
    DeathCertificateGenerator,
)
from bitikocr.data.synthetic.generators.form import FormGenerator, FormOptions

__all__ = [
    "DEFAULT_INK_STRENGTH",
    "GENERATOR_TYPES",
    "ArizaGenerator",
    "BirthCertificateGenerator",
    "DeathCertificateGenerator",
    "DocumentGenerator",
    "FieldValues",
    "FormGenerator",
    "FormOptions",
    "SyntheticDocument",
    "available_document_types",
    "create_generator",
]

GENERATOR_TYPES: dict[str, type[DocumentGenerator]] = {
    ArizaGenerator.name: ArizaGenerator,
    BirthCertificateGenerator.name: BirthCertificateGenerator,
    DeathCertificateGenerator.name: DeathCertificateGenerator,
}


def available_document_types() -> tuple[str, ...]:
    """Return every registered document type name, sorted."""
    return tuple(sorted(GENERATOR_TYPES))


def create_generator(
    document_type: str,
    config: SyntheticConfig,
    font_path: Path | str | None = None,
    template: str | None = None,
    ink_strength: float = DEFAULT_INK_STRENGTH,
) -> DocumentGenerator:
    """Build the generator registered under a document type name.

    Args:
        document_type: A name from :func:`available_document_types`.
        config: Paths the generator reads its assets from.
        font_path: Force a specific handwriting font instead of sampling.
        template: Fill a specific form variant. Only document types that
            fill a measured printed form accept one.
        ink_strength: How heavily the pen writes.

    Returns:
        A ready-to-use generator.

    Raises:
        KeyError: If no generator is registered under that name.
        ValueError: If a template is given for a document type that does
            not use one.
    """
    try:
        generator_type = GENERATOR_TYPES[document_type]
    except KeyError as error:
        known = ", ".join(available_document_types())
        raise KeyError(
            f"Unknown document type {document_type!r}. Known types: {known}"
        ) from error

    if template is not None:
        return generator_type.with_template(
            config, font_path, template, ink_strength
        )
    return generator_type(config, font_path, ink_strength=ink_strength)
