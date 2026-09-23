UV_RUN ?= uv run --locked --extra train

.PHONY: style style-check lint-check type-check checks test

style:
	$(UV_RUN) isort bitikocr tests scripts
	$(UV_RUN) black bitikocr tests scripts

style-check:
	$(UV_RUN) isort --check-only bitikocr tests scripts
	$(UV_RUN) black --check bitikocr tests scripts

lint-check:
	$(UV_RUN) ruff check bitikocr tests scripts

type-check:
	$(UV_RUN) mypy bitikocr scripts

checks: style-check lint-check type-check

test:
	$(UV_RUN) python -m pytest -q
