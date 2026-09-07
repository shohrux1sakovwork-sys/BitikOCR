# Force UTF-8 I/O so the tools can report on files holding Cyrillic text
# even when the shell's code page is not UTF-8 (Windows).
export PYTHONUTF8 = 1

.PHONY: install style style-check lint-check type-check test checks

install:
	uv sync --extra data

style:
	uv run isort bitikocr
	uv run black bitikocr

style-check:
	uv run isort --check-only bitikocr
	uv run black --check bitikocr

lint-check:
	uv run ruff check bitikocr

type-check:
	uv run mypy

test:
	uv run pytest

checks: style-check lint-check type-check test
