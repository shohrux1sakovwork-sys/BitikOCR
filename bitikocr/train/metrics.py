"""Compute corpus error rates for generated OCR transcriptions."""

import jiwer


def compute_ocr_metrics(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Compute corpus CER and WER using JiWER's default transformations.

    Args:
        predictions: Generated transcriptions in dataset order.
        references: Corresponding ground-truth transcriptions.

    Returns:
        Character and word error rates, including insertion counts when all
        references are blank. Error rates may exceed one.

    Raises:
        ValueError: The lists are empty or have different lengths.
    """
    if not references or len(predictions) != len(references):
        raise ValueError("Provide equal, nonempty prediction/reference lists.")
    return {
        "cer": float(jiwer.cer(references, predictions)),
        "wer": float(jiwer.wer(references, predictions)),
    }
