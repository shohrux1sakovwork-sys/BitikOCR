"""Tests for bitikocr.synthetic.dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from bitikocr.synthetic.augment import AugmentationProfile
from bitikocr.synthetic.dataset import (
    DatasetLayout,
    read_records,
    render_records,
    write_records,
)
from bitikocr.synthetic.generators import ArizaGenerator
from bitikocr.synthetic.records import DocumentRecord, sample_records


@pytest.fixture()
def ariza_records() -> list[DocumentRecord]:
    """Two sampled ariza records."""
    return sample_records("ariza", 2, random.Random(3), "cyrillic")


# -- records on disk -------------------------------------------------------


def test_records_round_trip_through_jsonl(
    ariza_records: list[DocumentRecord], tmp_path: Path
) -> None:
    path = write_records(ariza_records, tmp_path / "metadata.jsonl")
    assert read_records(path) == ariza_records


def test_the_metadata_file_is_one_json_object_per_line(
    ariza_records: list[DocumentRecord], tmp_path: Path
) -> None:
    path = write_records(ariza_records, tmp_path / "metadata.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(ariza_records)
    assert json.loads(lines[0])["document_type"] == "ariza"


def test_writing_records_creates_missing_directories(
    ariza_records: list[DocumentRecord], tmp_path: Path
) -> None:
    path = write_records(ariza_records, tmp_path / "deep" / "meta.jsonl")
    assert path.is_file()


def test_a_malformed_metadata_line_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "metadata.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_records(path)


def test_a_record_missing_its_seed_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "metadata.jsonl"
    path.write_text(
        json.dumps({"document_type": "ariza", "fields": {}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed record"):
        read_records(path)


# -- rendering -------------------------------------------------------------


def test_a_render_writes_an_image_and_a_label_per_record(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    summary = render_records(
        ariza_generator, ariza_records, tmp_path, augmentation=None
    )

    assert len(summary) == 2
    for sample in summary.samples:
        assert sample.image.is_file()
        assert sample.label.is_file()
        assert sample.preview is None


def test_images_and_labels_live_in_their_own_directories(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    summary = render_records(ariza_generator, ariza_records, tmp_path)
    layout = DatasetLayout(tmp_path)
    for sample in summary.samples:
        assert sample.image.parent == layout.images
        assert sample.label.parent == layout.labels


def test_the_label_holds_the_ground_truth(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    summary = render_records(ariza_generator, ariza_records, tmp_path)
    payload = json.loads(summary.samples[0].label.read_text(encoding="utf-8"))
    assert payload["document_type"] == "ariza"
    assert payload["lines"]
    assert payload["script"] == "cyrillic"


def test_the_index_describes_every_page(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    summary = render_records(ariza_generator, ariza_records, tmp_path)
    lines = summary.layout.index.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(summary.samples)

    entry = json.loads(lines[0])
    assert entry["image"].startswith("images/")
    assert entry["label"].startswith("labels/")
    assert entry["script"] == "cyrillic"
    assert entry["text"]
    # The paths are relative, so the dataset can be moved or mounted.
    assert (summary.layout.root / entry["image"]).is_file()


def test_box_overlays_are_written_on_request(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    summary = render_records(
        ariza_generator, ariza_records, tmp_path, draw_boxes=True
    )
    preview = summary.samples[0].preview
    assert preview is not None and preview.is_file()
    assert preview.parent == summary.layout.previews


def test_a_render_is_reproducible_from_its_records(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    """The seed rides on the record, so re-rendering repeats the page."""
    first = render_records(
        ariza_generator,
        ariza_records,
        tmp_path / "a",
        augmentation=AugmentationProfile(),
    )
    second = render_records(
        ariza_generator,
        ariza_records,
        tmp_path / "b",
        augmentation=AugmentationProfile(),
    )
    assert [s.stem for s in first.samples] == [s.stem for s in second.samples]
    assert (
        first.samples[0].image.read_bytes()
        == second.samples[0].image.read_bytes()
    )


def test_rendering_nothing_is_rejected(
    ariza_generator: ArizaGenerator, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="At least one record"):
        render_records(ariza_generator, [], tmp_path)


def test_a_record_the_fonts_cannot_write_is_skipped_not_fatal(
    ariza_generator: ArizaGenerator, tmp_path: Path
) -> None:
    """One unusable record must not lose the rest of the batch."""
    good = sample_records("ariza", 1, random.Random(3), "cyrillic")[0]
    unwritable = DocumentRecord(
        document_type="ariza",
        script="latin",
        seed=1,
        fields={"recipient": "日本語", "applicant": "x", "body": "y"},
    )

    summary = render_records(ariza_generator, [unwritable, good], tmp_path)
    assert len(summary) == 1
    assert len(summary.skipped) == 1
    assert summary.skipped[0][0] == 0
