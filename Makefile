# Football Synthesizer — reproducible targets (filled in as modules land).
.PHONY: help install test lint sources

help:
	@echo "install   - pip install -e . (add [cv] extra for the generator)"
	@echo "test      - run pytest"
	@echo "lint      - ruff check + format"
	@echo "sources   - list the configured data sources"

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

sources:
	python -m ingest.sources --list
