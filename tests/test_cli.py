"""Tests for bitikocr.cli."""

from __future__ import annotations

from pathlib import Path

import pytest

from bitikocr.cli import main
from bitikocr.config import ENV_FONTS_DIR
from bitikocr.synthetic.dataset import read_records


def test_listing_document_types(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["synth", "list-types"]) == 0
    assert capsys.readouterr().out.split() == [
        "ariza",
        "birth_certificate",
        "death_certificate",
    ]


def test_listing_fonts(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["synth", "list-fonts"]) == 0
    assert "font(s) in" in capsys.readouterr().out


def test_generating_writes_the_requested_number_of_samples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "synth",
            "generate",
            "ariza",
            "--count",
            "1",
            "--seed",
            "4",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(list((tmp_path / "images").glob("*.png"))) == 1
    assert len(list((tmp_path / "labels").glob("*.json"))) == 1
    assert (tmp_path / "metadata.jsonl").is_file()
    assert (tmp_path / "index.jsonl").is_file()
    assert str(tmp_path) in capsys.readouterr().out


def test_a_missing_fonts_directory_is_reported_not_raised(
    tmp_path: Path,
) -> None:
    assert main(["synth", "list-fonts", "--fonts-dir", str(tmp_path)]) == 1


def test_the_fonts_directory_can_come_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_FONTS_DIR, str(tmp_path))
    assert main(["synth", "list-fonts"]) == 1


def test_an_unknown_document_type_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        main(["synth", "generate", "passport"])


def test_listing_templates(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["synth", "list-templates"]) == 0
    printed = capsys.readouterr().out
    assert "birth_certificate_bilingual" in printed
    assert "death_certificate_bilingual" in printed
    assert "keep-out: qr_code" in printed


def test_metadata_is_written_without_rendering(tmp_path: Path) -> None:
    """Stage one must be usable on its own, before any page is drawn."""
    exit_code = main(
        [
            "synth",
            "metadata",
            "birth_certificate",
            "--count",
            "3",
            "--seed",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    lines = (
        (tmp_path / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(lines) == 3
    assert not (tmp_path / "images").exists()


def test_metadata_can_be_rendered_afterwards(tmp_path: Path) -> None:
    assert (
        main(
            [
                "synth",
                "metadata",
                "ariza",
                "--count",
                "2",
                "--seed",
                "1",
                "--script",
                "cyrillic",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert main(["synth", "render", str(tmp_path), "--augment", "0"]) == 0
    assert len(list((tmp_path / "images").glob("*.png"))) == 2


def test_rendering_an_empty_dataset_is_reported(tmp_path: Path) -> None:
    (tmp_path / "metadata.jsonl").write_text("", encoding="utf-8")
    assert main(["synth", "render", str(tmp_path)]) == 1


def test_rendering_without_metadata_is_reported(tmp_path: Path) -> None:
    assert main(["synth", "render", str(tmp_path)]) == 1


def test_one_script_can_be_forced(tmp_path: Path) -> None:
    main(
        [
            "synth",
            "metadata",
            "death_certificate",
            "--count",
            "4",
            "--seed",
            "2",
            "--script",
            "cyrillic",
            "--output-dir",
            str(tmp_path),
        ]
    )
    records = read_records(tmp_path / "metadata.jsonl")
    assert {record.script for record in records} == {"cyrillic"}
