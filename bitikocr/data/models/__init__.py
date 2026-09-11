"""The vocabulary the corpus is described in.

Two things live here, and only things of this kind belong: the interchange
schema every document conforms to however it was produced, and the geometry
that schema is written in. They sit beside the producers under
:mod:`bitikocr.data` because they describe the corpus, not any one way of
filling it — :mod:`~bitikocr.data.synthetic` is merely its first producer,
and real scans land beside it.

Nothing here knows how a document is made or read. A structure that serves
one producer belongs to that producer instead: the synthetic generator's own
ground truth is in :mod:`bitikocr.data.synthetic.annotation`, and
:mod:`~bitikocr.data.synthetic.export` translates it into the schema.
"""

from bitikocr.data.models.geometry import BoundingBox

__all__ = ["BoundingBox"]
