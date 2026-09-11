"""Everything to do with the training corpus, on both sides of it.

:mod:`bitikocr.data.synthetic` produces handwritten Uzbek documents together
with exact ground truth, and :mod:`bitikocr.data.sft_dataset` loads
image/transcription pairs and collates them into batches for supervised
training. :mod:`bitikocr.data.models` holds the schema both of them
describe a document in.

Real archive scans and the pipelines that prepare them belong here too, as
they arrive.
"""
