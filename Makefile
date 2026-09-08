# Force UTF-8 I/O so the tools can report on files holding Cyrillic text
# even when the shell's code page is not UTF-8 (Windows).
export PYTHONUTF8 = 1

# The package, its entry points and its suite: everything that is checked.
SOURCES = bitikocr scripts tests

.PHONY: install style style-check lint-check type-check test checks

install:
	uv sync --extra data

style:
	uv run isort $(SOURCES)
	uv run black $(SOURCES)

style-check:
	uv run isort --check-only $(SOURCES)
	uv run black --check $(SOURCES)

lint-check:
	uv run ruff check $(SOURCES)

type-check:
	uv run mypy

test:
	uv run pytest

checks: style-check lint-check type-check test
