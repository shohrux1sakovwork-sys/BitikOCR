"""The Uzbek death certificate — ``O'lim haqida guvohnoma``.

All the filling logic lives in :mod:`bitikocr.data.synthetic.generators.form`;
this module only names the document type and the form variant it fills by
default. Older single-page variants are added as layouts, not as code.
"""

from __future__ import annotations

from typing import ClassVar

from bitikocr.data.synthetic.generators.form import FormGenerator

__all__ = ["DEFAULT_TEMPLATE", "DeathCertificateGenerator"]

DEFAULT_TEMPLATE = "death_certificate_bilingual"


class DeathCertificateGenerator(FormGenerator):
    """Fill a death certificate form with a clerk's handwriting."""

    name: ClassVar[str] = "death_certificate"
    default_template: ClassVar[str] = DEFAULT_TEMPLATE
