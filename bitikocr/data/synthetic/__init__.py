"""Synthetic handwritten-document generation for HTR training data.

The module is layered so each piece has one job:

* :mod:`~bitikocr.data.synthetic.fonts` — font discovery, coverage and metrics.
* :mod:`~bitikocr.data.synthetic.style` — the sampled "writer" behind a page.
* :mod:`~bitikocr.data.synthetic.hand` — turning text into handwritten ink.
* :mod:`~bitikocr.data.synthetic.effects` — signatures and office seals.
* :mod:`~bitikocr.data.synthetic.layout` — placing ink and collecting ground truth.
* :mod:`~bitikocr.data.synthetic.templates` — measured geometry of printed forms.
* :mod:`~bitikocr.data.synthetic.generators` — one module per document type.
* :mod:`~bitikocr.data.synthetic.scripts` — the Latin and Cyrillic alphabets.
* :mod:`~bitikocr.data.synthetic.corpus` — the Uzbek vocabulary records draw on.
* :mod:`~bitikocr.data.synthetic.records` — sampling a document's field values.
* :mod:`~bitikocr.data.synthetic.augment` — spoiling a clean page like a scan.
* :mod:`~bitikocr.data.synthetic.dataset` — writing batches of samples to disk.

Typical use::

    config = SyntheticConfig.from_env()
    generator = create_generator("ariza", config)
    document = generator.generate(fields, seed=42)
"""

from bitikocr.data.synthetic.augment import AugmentationProfile, augment_page
from bitikocr.data.synthetic.dataset import (
    DatasetLayout,
    DatasetSummary,
    SampleFiles,
    draw_annotations,
    read_records,
    render_records,
    write_records,
)
from bitikocr.data.synthetic.fonts import FontInfo, FontLibrary
from bitikocr.data.synthetic.generators import (
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
from bitikocr.data.synthetic.hand import Hand
from bitikocr.data.synthetic.layout import Page, wrap_text
from bitikocr.data.synthetic.records import (
    DEFAULT_LATIN_SHARE,
    DocumentRecord,
    sample_record,
    sample_records,
)
from bitikocr.data.synthetic.scripts import Script, to_cyrillic
from bitikocr.data.synthetic.style import HandwritingStyle, sample_style
from bitikocr.data.synthetic.templates import (
    FieldGeometry,
    FormTemplate,
    MarkArea,
)

__all__ = [
    "DEFAULT_LATIN_SHARE",
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
