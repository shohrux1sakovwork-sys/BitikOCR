# Force UTF-8 I/O so the tools can report on files holding Cyrillic text
# even when the shell's code page is not UTF-8 (Windows).
export PYTHONUTF8 = 1

# Both halves of the project are checked together, so both extras are
# needed: the generator draws pages, the training pipeline reads them.
UV_RUN ?= uv run --locked --extra data --extra train

# The package, its entry points and its suite: everything that is checked.
SOURCES = bitikocr scripts tests

.PHONY: install style style-check lint-check type-check test checks

install:
	uv sync --extra data --extra train

style:
	$(UV_RUN) isort $(SOURCES)
	$(UV_RUN) black $(SOURCES)

style-check:
	$(UV_RUN) isort --check-only $(SOURCES)
	$(UV_RUN) black --check $(SOURCES)

lint-check:
	$(UV_RUN) ruff check $(SOURCES)

# No argument: mypy reads the paths from [tool.mypy] in pyproject.toml.
type-check:
	$(UV_RUN) mypy

test:
	$(UV_RUN) python -m pytest -q

checks: style-check lint-check type-check test
