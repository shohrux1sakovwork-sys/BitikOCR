"""Tests for bitikocr.data.synthetic.dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from bitikocr.data.synthetic.augment import AugmentationProfile
from bitikocr.data.synthetic.dataset import (
    DatasetLayout,
    read_records,
    render_records,
    write_records,
)
from bitikocr.data.synthetic.generators import ArizaGenerator
from bitikocr.data.synthetic.records import DocumentRecord, sample_records


@pytest.fixture()
def ariza_records() -> list[DocumentRecord]:
    """Two sampled ariza records."""
    return sample_records("ariza", 2, random.Random(3), "cyrillic")


@pytest.fixture()
def layout(tmp_path: Path) -> DatasetLayout:
    """An empty dataset directory."""
    return DatasetLayout(tmp_path)


# -- facts on disk ---------------------------------------------------------


def test_facts_round_trip_through_their_files(
    ariza_records: list[DocumentRecord], layout: DatasetLayout
) -> None:
    write_records(ariza_records, layout)
    assert read_records(layout) == ariza_records


def test_one_facts_file_is_written_per_document(
    ariza_records: list[DocumentRecord], layout: DatasetLayout
) -> None:
    paths = write_records(ariza_records, layout)
    assert len(paths) == len(ariza_records)
    assert all(path.parent == layout.facts for path in paths)

    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["document_type"] == "ariza"
    assert payload["fields"]


def test_writing_facts_creates_missing_directories(
    ariza_records: list[DocumentRecord], tmp_path: Path
) -> None:
    layout = DatasetLayout(tmp_path / "deep" / "set")
    assert write_records(ariza_records, layout)[0].is_file()


def test_a_dataset_without_facts_is_reported(layout: DatasetLayout) -> None:
    with pytest.raises(FileNotFoundError, match="No facts directory"):
        read_records(layout)


def test_a_malformed_facts_file_is_reported(layout: DatasetLayout) -> None:
    layout.facts.mkdir(parents=True)
    (layout.facts / "broken.json").write_text("{not json}", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_records(layout)


def test_a_record_missing_its_seed_is_rejected(layout: DatasetLayout) -> None:
    layout.facts.mkdir(parents=True)
    (layout.facts / "a.json").write_text(
        json.dumps({"document_type": "ariza", "fields": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed record"):
        read_records(layout)


# -- rendering -------------------------------------------------------------


def test_a_render_writes_every_file_a_sample_needs(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    summary = render_records(
        ariza_generator, ariza_records, tmp_path, augmentation=None
    )

    assert len(summary) == 2
    for sample in summary.samples:
        assert sample.facts.is_file()
        assert sample.image.is_file()
        assert sample.annotation.is_file()
        assert sample.preview is None


def test_each_kind_of_file_lives_in_its_own_directory(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    summary = render_records(ariza_generator, ariza_records, tmp_path)
    layout = DatasetLayout(tmp_path)
    for sample in summary.samples:
        assert sample.facts.parent == layout.facts
        assert sample.image.parent == layout.images
        assert sample.annotation.parent == layout.annotations


def test_a_samples_files_share_one_id(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    """A training pipeline should be able to pair files without the index."""
    summary = render_records(ariza_generator, ariza_records, tmp_path)
    for position, sample in enumerate(summary.samples):
        assert sample.id == f"doc_{position:06d}"
        assert sample.facts.stem == sample.id
        assert sample.image.stem == sample.id
        assert sample.annotation.stem == sample.id


def test_the_id_prefix_can_be_changed(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    """Sets merged into one corpus must not collide on ids."""
    summary = render_records(
        ariza_generator, ariza_records, tmp_path, prefix="ariza"
    )
    assert summary.samples[0].id == "ariza_000000"


def test_the_annotation_holds_the_ground_truth(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    summary = render_records(ariza_generator, ariza_records, tmp_path)
    payload = json.loads(
        summary.samples[0].annotation.read_text(encoding="utf-8")
    )
    assert payload["id"] == summary.samples[0].id
    assert payload["metadata"]["document_type"] == "ariza"
    assert payload["metadata"]["primary_script"] == "cyrillic"
    assert payload["target"]["text"]
    assert payload["target"]["parts"]


def test_the_facts_beside_a_page_describe_that_page(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    summary = render_records(ariza_generator, ariza_records, tmp_path)
    payload = json.loads(summary.samples[0].facts.read_text(encoding="utf-8"))
    assert payload["id"] == summary.samples[0].id
    assert payload["image"].endswith(".png")

    written = {fact["evidence_text"] for fact in payload["facts"]}
    assert ariza_records[0].fields["signature_name"] in written


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
    assert entry["annotation"].startswith("annotations/")
    assert entry["facts"].startswith("facts/")
    assert entry["script"] == "cyrillic"
    assert entry["text"]
    # The paths are relative, so the dataset can be moved or mounted.
    for key in ("image", "annotation", "facts"):
        assert (summary.layout.root / entry[key]).is_file(), key


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


def test_a_render_is_reproducible_from_its_facts(
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
    assert [s.id for s in first.samples] == [s.id for s in second.samples]
    assert (
        first.samples[0].image.read_bytes()
        == second.samples[0].image.read_bytes()
    )


def test_edited_facts_are_rendered_as_edited(
    ariza_generator: ArizaGenerator,
    ariza_records: list[DocumentRecord],
    tmp_path: Path,
) -> None:
    """Facts are meant to be reviewed and corrected before drawing."""
    layout = DatasetLayout(tmp_path)
    write_records(ariza_records, layout)

    path = min(layout.facts.glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fields"]["signature_name"] = "Ўзгартирилган"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    summary = render_records(
        ariza_generator, read_records(layout), tmp_path, augmentation=None
    )
    annotation = json.loads(
        summary.samples[0].annotation.read_text(encoding="utf-8")
    )
    assert "Ўзгартирилган" in annotation["target"]["text"]


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
