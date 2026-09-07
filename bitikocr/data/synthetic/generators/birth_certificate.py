"""The Uzbek birth certificate — ``Tug'ilganlik haqida guvohnoma``.

All the filling logic lives in :mod:`bitikocr.data.synthetic.generators.form`;
this module only names the document type and the form variant it fills by
default. A single-language variant is added as a layout, not as code.
"""

from __future__ import annotations

from typing import ClassVar

from bitikocr.data.synthetic.generators.form import FormGenerator

__all__ = ["DEFAULT_TEMPLATE", "BirthCertificateGenerator"]

DEFAULT_TEMPLATE = "birth_certificate_bilingual"


class BirthCertificateGenerator(FormGenerator):
    """Fill a birth certificate form with a clerk's handwriting."""

    name: ClassVar[str] = "birth_certificate"
    default_template: ClassVar[str] = DEFAULT_TEMPLATE
