.PHONY: help install test lint format verify benchmark clean

PYTHON ?= python

help:
	@echo "install    install package plus development and figure dependencies"
	@echo "test       run the scientific and regression suite"
	@echo "lint       check lint and formatting"
	@echo "format     apply the configured formatter"
	@echo "verify     run lint and tests"
	@echo "benchmark  run the bounded end-to-end performance benchmark"
	@echo "clean      remove reproducible local caches and build products"

install:
	$(PYTHON) -m pip install -e ".[dev,figures]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

verify: lint test

benchmark:
	PYTHONPATH=src $(PYTHON) scripts/benchmark_performance.py

clean:
	$(PYTHON) -c 'from pathlib import Path; import shutil; [shutil.rmtree(path, ignore_errors=True) for name in ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist") for path in Path(".").rglob(name)]'
