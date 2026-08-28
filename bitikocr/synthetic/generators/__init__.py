"""Document generators and the registry that resolves them by name.

Adding a document type means adding a module here and one entry in
:data:`GENERATOR_TYPES`. Nothing else in the project needs to change.
"""

from __future__ import annotations

from pathlib import Path

from bitikocr.config import SyntheticConfig
from bitikocr.synthetic.generators.ariza import ArizaGenerator
from bitikocr.synthetic.generators.base import (
    DocumentGenerator,
    FieldValues,
    SyntheticDocument,
)
from bitikocr.synthetic.generators.death_certificate import (
    DeathCertificateGenerator,
    DeathCertificateOptions,
)

__all__ = [
    "GENERATOR_TYPES",
    "ArizaGenerator",
    "DeathCertificateGenerator",
    "DeathCertificateOptions",
    "DocumentGenerator",
    "FieldValues",
    "SyntheticDocument",
    "available_document_types",
    "create_generator",
]

GENERATOR_TYPES: dict[str, type[DocumentGenerator]] = {
    ArizaGenerator.name: ArizaGenerator,
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
) -> DocumentGenerator:
    """Build the generator registered under a document type name.

    Args:
        document_type: A name from :func:`available_document_types`.
        config: Paths the generator reads its assets from.
        font_path: Force a specific handwriting font instead of sampling.
        template: Fill a specific form variant. Only document types that
            fill a measured printed form accept one.

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
        return generator_type.with_template(config, font_path, template)
    return generator_type(config, font_path)
