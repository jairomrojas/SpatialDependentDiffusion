# Contributing

## Development setup

On the reference workstation, use the global Pixi environment named
`physics`. The `python` shim exposed by Pixi must resolve to that environment:

```bash
pixi global list
python -c "import sys; print(sys.prefix)"
PYTHONPATH=src python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

External collaborators without that workstation environment can create a
virtual environment and install `.[dev]`. The package uses a `src/` layout;
CI installs it in editable mode so imports also exercise the packaged path.
`python run.py ...` remains available in the mandatory global environment via
the repository-local `PYTHONPATH` workflow.

## Scientific change checklist

Changes to equations, response cases, stochastic updates, boundaries, density
estimation, or the finite-volume solver should include:

1. a mathematical identity or limiting case in `tests/`;
2. fixed-seed reproducibility where applicable;
3. a refinement check in \(dt\), grid spacing, \(M\), or bandwidth;
4. conservation/nonnegativity checks for probability densities;
5. an update to `../manuscript/main.tex` when an equation or scientific
   interpretation changes;
6. regenerated artifacts only after all checks pass.

Do not infer a weak physical effect from a case-specific color scale,
total-variation norm, or \(L^2\) norm alone. Use a signed preselected
observable with paired uncertainty, then check response scaling and numerical
refinement.

## Generated files

Raw files below `results/data/`, videos, diagnostic frame sequences, caches,
compiled Numba artifacts, and local virtual environments are ignored by
version control. Share large generated results as a release artifact or data
deposit, keeping NPZ and CSV files together with their figures. Density NPZ
files embed their TOML input and runtime provenance. Only compact figures and
summary metrics intended for collaborators belong in curated `results/`.

## Public release

This repository does not choose a legal license on behalf of its owner. Add a
`LICENSE` file and author/citation metadata before making a public release.
