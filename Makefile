# Force UTF-8 I/O so the tools can report on files holding Cyrillic text
# even when the shell's code page is not UTF-8 (Windows).
export PYTHONUTF8 = 1

.PHONY: install style style-check lint-check type-check test checks

install:
	uv sync --extra data

style:
	uv run isort bitikocr tests
	uv run black bitikocr tests

style-check:
	uv run isort --check-only bitikocr tests
	uv run black --check bitikocr tests

lint-check:
	uv run ruff check bitikocr tests

type-check:
	uv run mypy

test:
	uv run pytest

checks: style-check lint-check type-check test
