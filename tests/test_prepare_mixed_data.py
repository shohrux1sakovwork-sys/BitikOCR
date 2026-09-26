"""Tests for rewriting gold notes in the benchmark's conventions."""

import pytest

from scripts.prepare_mixed_data import convert_notes


def test_signatures_become_the_signature_tag() -> None:
    assert convert_notes("Nabiyev B. [imzo] 01.11.2023") == (
        "Nabiyev B. <signature> 01.11.2023"
    )
    assert convert_notes("[имзо] У.Турдиқул") == "<signature> У.Турдиқул"
    assert convert_notes("[қолы]") == "<signature>"


def test_a_stamp_keeps_its_text_and_marks_gaps_illegible() -> None:
    text = convert_notes("Ariza\n[Штамп: BUXORO VILOYAT ... KIRIM № 3]")
    assert text == "Ariza\n\n<stamp>\nBUXORO VILOYAT [...] KIRIM № 3\n</stamp>"


def test_a_bare_seal_is_an_empty_stamp() -> None:
    assert convert_notes("[Муҳр]") == "<stamp/>"


def test_a_resolution_keeps_its_text_and_its_signature() -> None:
    text = "[Резолюция: В.Тоғаровга ҳал этиш учун [имзо] 02.11.23й]"
    assert convert_notes(text) == (
        "В.Тоғаровга ҳал этиш учун <signature> 02.11.23й"
    )


def test_a_crossed_out_word_is_an_empty_deletion() -> None:
    assert convert_notes("турар [ўчирилган] жой") == (
        "турар <deleted></deleted> жой"
    )


def test_notes_about_the_scan_are_dropped() -> None:
    assert convert_notes("Ariza\n[Фото санаси: 01/11/2023 21:24:59]") == "Ariza"


def test_the_benchmarks_illegible_mark_is_left_alone() -> None:
    assert convert_notes("KELGAN [...] М-1505") == "KELGAN [...] М-1505"


def test_an_unknown_note_stops_the_run() -> None:
    with pytest.raises(ValueError, match="no rule"):
        convert_notes("[Нимадир]")
