"""Tests for bitikocr.cli."""

from __future__ import annotations

from pathlib import Path

import pytest

from bitikocr.cli import main
from bitikocr.config import ENV_FONTS_DIR


def test_listing_document_types(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["synth", "list-types"]) == 0
    assert capsys.readouterr().out.split() == ["ariza", "death_certificate"]


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
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
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
