"""Synthetic handwritten-document generation for HTR training data.

The module is layered so each piece has one job:

* :mod:`~bitikocr.synthetic.fonts` — font discovery, coverage and metrics.
* :mod:`~bitikocr.synthetic.style` — the sampled "writer" behind a page.
* :mod:`~bitikocr.synthetic.hand` — turning text into handwritten ink.
* :mod:`~bitikocr.synthetic.effects` — signatures and office seals.
* :mod:`~bitikocr.synthetic.layout` — placing ink and collecting ground truth.
* :mod:`~bitikocr.synthetic.templates` — measured geometry of printed forms.
* :mod:`~bitikocr.synthetic.generators` — one module per document type.
* :mod:`~bitikocr.synthetic.scripts` — the Latin and Cyrillic alphabets.
* :mod:`~bitikocr.synthetic.corpus` — the Uzbek vocabulary records draw on.
* :mod:`~bitikocr.synthetic.records` — sampling a document's field values.
* :mod:`~bitikocr.synthetic.augment` — spoiling a clean page like a scan.
* :mod:`~bitikocr.synthetic.dataset` — writing batches of samples to disk.

Typical use::

    config = SyntheticConfig.from_env()
    generator = create_generator("ariza", config)
    document = generator.generate(fields, seed=42)
"""

from bitikocr.synthetic.augment import AugmentationProfile, augment_page
from bitikocr.synthetic.dataset import (
    DatasetLayout,
    DatasetSummary,
    SampleFiles,
    draw_annotations,
    read_records,
    render_records,
    write_records,
)
from bitikocr.synthetic.fonts import FontInfo, FontLibrary
from bitikocr.synthetic.generators import (
    ArizaGenerator,
    BirthCertificateGenerator,
    DeathCertificateGenerator,
    DocumentGenerator,
    FormGenerator,
    FormOptions,
    SyntheticDocument,
    available_document_types,
    create_generator,
)
from bitikocr.synthetic.hand import Hand
from bitikocr.synthetic.layout import Page, wrap_text
from bitikocr.synthetic.records import (
    DocumentRecord,
    sample_record,
    sample_records,
)
from bitikocr.synthetic.scripts import Script, to_cyrillic
from bitikocr.synthetic.style import HandwritingStyle, sample_style
from bitikocr.synthetic.templates import FieldGeometry, FormTemplate, MarkArea

__all__ = [
    "ArizaGenerator",
    "AugmentationProfile",
    "BirthCertificateGenerator",
    "DatasetLayout",
    "DatasetSummary",
    "DeathCertificateGenerator",
    "DocumentGenerator",
    "DocumentRecord",
    "FieldGeometry",
    "FontInfo",
    "FontLibrary",
    "FormGenerator",
    "FormOptions",
    "FormTemplate",
    "Hand",
    "HandwritingStyle",
    "MarkArea",
    "Page",
    "SampleFiles",
    "Script",
    "SyntheticDocument",
    "augment_page",
    "available_document_types",
    "create_generator",
    "draw_annotations",
    "read_records",
    "render_records",
    "sample_record",
    "sample_records",
    "sample_style",
    "to_cyrillic",
    "wrap_text",
    "write_records",
]
