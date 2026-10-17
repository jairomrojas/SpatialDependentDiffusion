.PHONY: help paper figures test lint format verify benchmark clean

# Everything runs from the repository root; code is imported, not installed.
PYTHON ?= python

help:
	@echo "paper      compile the manuscript"
	@echo "figures    regenerate standalone manuscript figures"
	@echo "test       run the scientific and regression suite"
	@echo "lint       check lint and formatting"
	@echo "format     apply the configured formatter"
	@echo "verify     run lint and tests"
	@echo "benchmark  run a bounded simulation performance benchmark"
	@echo "clean      remove LaTeX and Python build caches"

paper:
	cd manuscript && latexmk -pdf main.tex

figures:
	$(PYTHON) -m analysis.manuscript.figure_1_response_rules.run
	$(PYTHON) -m analysis.manuscript.figure_2_nutrient_landscape.run
	$(PYTHON) -m analysis.manuscript.figure_4_macroscopic_observables.run --reuse
	$(PYTHON) -m analysis.manuscript.figure_5_foraging_observables.run --reuse
	$(PYTHON) -m analysis.manuscript.figure_6_information_observables.run --reuse

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

verify: lint test

benchmark:
	$(PYTHON) -m analysis.simulations.performance_benchmark.run

clean:
	cd manuscript && latexmk -c
	$(PYTHON) -c 'from pathlib import Path; import shutil; [shutil.rmtree(path, ignore_errors=True) for name in ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache") for path in Path(".").rglob(name)]'
