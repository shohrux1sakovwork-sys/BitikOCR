"""Synthetic handwritten-document generation for HTR training data.

The module is layered so each piece has one job:

* :mod:`~bitikocr.synthetic.fonts` — font discovery, coverage and metrics.
* :mod:`~bitikocr.synthetic.style` — the sampled "writer" behind a page.
* :mod:`~bitikocr.synthetic.hand` — turning text into handwritten ink.
* :mod:`~bitikocr.synthetic.effects` — signatures and office seals.
* :mod:`~bitikocr.synthetic.layout` — placing ink and collecting ground truth.
* :mod:`~bitikocr.synthetic.templates` — measured geometry of printed forms.
* :mod:`~bitikocr.synthetic.generators` — one module per document type.
* :mod:`~bitikocr.synthetic.dataset` — writing batches of samples to disk.

Typical use::

    config = SyntheticConfig.from_env()
    generator = create_generator("ariza", config)
    document = generator.generate(fields, seed=42)
"""

from bitikocr.synthetic.dataset import (
    DatasetSummary,
    SampleFiles,
    draw_annotations,
    generate_dataset,
    write_sample,
)
from bitikocr.synthetic.fonts import FontInfo, FontLibrary
from bitikocr.synthetic.generators import (
    ArizaGenerator,
    DeathCertificateGenerator,
    DeathCertificateOptions,
    DocumentGenerator,
    SyntheticDocument,
    available_document_types,
    create_generator,
)
from bitikocr.synthetic.hand import Hand
from bitikocr.synthetic.layout import Page, wrap_text
from bitikocr.synthetic.sample_data import sample_fields_for
from bitikocr.synthetic.style import HandwritingStyle, sample_style
from bitikocr.synthetic.templates import FieldGeometry, FormTemplate, MarkArea

__all__ = [
    "ArizaGenerator",
    "DatasetSummary",
    "DeathCertificateGenerator",
    "DeathCertificateOptions",
    "DocumentGenerator",
    "FieldGeometry",
    "FontInfo",
    "FontLibrary",
    "FormTemplate",
    "Hand",
    "HandwritingStyle",
    "MarkArea",
    "Page",
    "SampleFiles",
    "SyntheticDocument",
    "available_document_types",
    "create_generator",
    "draw_annotations",
    "generate_dataset",
    "sample_fields_for",
    "sample_style",
    "wrap_text",
    "write_sample",
]
