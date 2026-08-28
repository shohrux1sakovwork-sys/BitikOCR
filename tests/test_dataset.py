"""Tests for bitikocr.synthetic.dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bitikocr.synthetic.dataset import generate_dataset
from bitikocr.synthetic.generators import ArizaGenerator


def test_a_run_writes_an_image_and_an_annotation_per_sample(
    ariza_generator: ArizaGenerator,
    ariza_fields: dict[str, Any],
    tmp_path: Path,
) -> None:
    summary = generate_dataset(
        generator=ariza_generator,
        field_sets=[ariza_fields],
        count=2,
        output_dir=tmp_path,
        seed=3,
    )

    assert len(summary) == 2
    for sample in summary.samples:
        assert sample.image.is_file()
        assert sample.annotation.is_file()
        assert sample.preview is None


def test_the_annotation_file_holds_the_ground_truth(
    ariza_generator: ArizaGenerator,
    ariza_fields: dict[str, Any],
    tmp_path: Path,
) -> None:
    summary = generate_dataset(
        generator=ariza_generator,
        field_sets=[ariza_fields],
        count=1,
        output_dir=tmp_path,
        seed=3,
    )
    payload = json.loads(
        summary.samples[0].annotation.read_text(encoding="utf-8")
    )
    assert payload["document_type"] == "ariza"
    assert payload["lines"]


def test_box_overlays_are_written_on_request(
    ariza_generator: ArizaGenerator,
    ariza_fields: dict[str, Any],
    tmp_path: Path,
) -> None:
    summary = generate_dataset(
        generator=ariza_generator,
        field_sets=[ariza_fields],
        count=1,
        output_dir=tmp_path,
        seed=3,
        draw_boxes=True,
    )
    preview = summary.samples[0].preview
    assert preview is not None and preview.is_file()


def test_a_run_is_reproducible_from_its_seed(
    ariza_generator: ArizaGenerator,
    ariza_fields: dict[str, Any],
    tmp_path: Path,
) -> None:
    stems = []
    for run in ("first", "second"):
        summary = generate_dataset(
            generator=ariza_generator,
            field_sets=[ariza_fields],
            count=2,
            output_dir=tmp_path / run,
            seed=99,
        )
        stems.append([sample.image.name for sample in summary.samples])
    assert stems[0] == stems[1]


def test_the_output_directory_is_created(
    ariza_generator: ArizaGenerator,
    ariza_fields: dict[str, Any],
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "run"
    generate_dataset(
        generator=ariza_generator,
        field_sets=[ariza_fields],
        count=1,
        output_dir=target,
        seed=3,
    )
    assert target.is_dir()


def test_a_non_positive_count_is_rejected(
    ariza_generator: ArizaGenerator,
    ariza_fields: dict[str, Any],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        generate_dataset(
            generator=ariza_generator,
            field_sets=[ariza_fields],
            count=0,
            output_dir=tmp_path,
        )


def test_an_empty_field_pool_is_rejected(
    ariza_generator: ArizaGenerator, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="At least one field set"):
        generate_dataset(
            generator=ariza_generator,
            field_sets=[],
            count=1,
            output_dir=tmp_path,
        )
