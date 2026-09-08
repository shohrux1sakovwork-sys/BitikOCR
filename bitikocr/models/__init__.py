"""The vocabulary the whole project describes documents in.

Two things live here, and only things of this kind belong: the corpus
interchange schema, which every document conforms to however it was
produced, and the geometry the schema is written in.

Nothing here knows how a document is made or read. A structure that serves
one producer belongs to that producer instead — the synthetic generator's
own ground truth is in :mod:`bitikocr.data.synthetic.annotation`.
"""

from bitikocr.models.geometry import BoundingBox

__all__ = ["BoundingBox"]
