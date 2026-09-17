"""Check corpus OCR scores, including blank-document hallucinations."""

import pytest

from bitikocr.train.metrics import compute_ocr_metrics


@pytest.mark.parametrize(
    ("predictions", "references", "expected"),
    [
        (["a", "b d"], ["a", "b c"], {"cer": 1 / 4, "wer": 1 / 3}),
        (["abc"], ["abc"], {"cer": 0, "wer": 0}),
        ([""], ["abc"], {"cer": 1, "wer": 1}),
        (["", ""], ["", ""], {"cer": 0, "wer": 0}),
        (["abc def", ""], ["", ""], {"cer": 7, "wer": 2}),
        (["a", "x"], ["a", ""], {"cer": 1, "wer": 1}),
    ],
)
def test_corpus_scores(predictions, references, expected) -> None:
    """Preserve corpus weighting and JiWER's insertion-count convention."""
    assert compute_ocr_metrics(predictions, references) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("predictions", "references"),
    [([], []), (["x"], []), ([], [""]), (["", ""], [""])],
)
def test_invalid_metric_inputs(predictions, references) -> None:
    """Reject missing examples and mismatched pairs rather than scoring zero."""
    with pytest.raises(ValueError, match="equal, nonempty"):
        compute_ocr_metrics(predictions, references)
