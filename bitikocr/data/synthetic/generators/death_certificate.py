"""The Uzbek death certificate — ``O'lim haqida guvohnoma``.

All the filling logic lives in :mod:`bitikocr.data.synthetic.generators.form`;
this module only names the document type and the form variant it fills by
default. Two variants ship as layouts, not as code:

- ``death_certificate_bilingual`` — the two-page Latin/Cyrillic form, the
  default.
- ``death_certificate_cyrillic_single`` — the older single-page form printed
  in Cyrillic only. It has no citizenship cell, gives the issue day and
  month one shared cell, and typesets a form series beside the serial.

One record fills either: the sampler emits every spelling the variants
need, and a template writes the fields it has cells for.
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
