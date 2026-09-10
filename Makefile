UV_RUN ?= uv run --locked --extra train

.PHONY: style style-check lint-check type-check checks test

style:
	$(UV_RUN) isort bitikocr tests
	$(UV_RUN) black bitikocr tests

style-check:
	$(UV_RUN) isort --check-only bitikocr tests
	$(UV_RUN) black --check bitikocr tests

lint-check:
	$(UV_RUN) ruff check bitikocr tests

type-check:
	$(UV_RUN) mypy bitikocr

checks: style-check lint-check type-check

test:
	$(UV_RUN) python -m pytest -q
