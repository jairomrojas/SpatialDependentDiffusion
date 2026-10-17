# Contributing

## Development setup

On the reference workstation, use the global Pixi environment named
`physics`. The `python` shim exposed by Pixi must resolve to that environment:

```bash
pixi global list
python -c "import sys; print(sys.prefix)"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

All commands run from the repository root; the code is imported from there,
not installed. External collaborators without that workstation environment can
create a virtual environment and install the dependencies listed in
`pyproject.toml` (see `simulations/population_release/README.md`); CI does the same.

## Scientific change checklist

Changes to equations, response cases, stochastic updates, boundaries, density
estimation, or the finite-volume solver should include:

1. a mathematical identity or limiting case in `model/solvers/tests/`;
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

Everything below `outputs/` (raw archives, videos, diagnostic frame sequences, caches,
compiled Numba artifacts) and local virtual environments are ignored by
version control. Share large generated results as a release artifact or data
deposit, keeping NPZ and CSV files together with their figures. Density NPZ
files embed their TOML input and runtime provenance. Only compact figures and
summary metrics intended for collaborators are curated in
`outputs/simulations/population_release/` and shared deliberately.

## Public release

This repository does not choose a legal license on behalf of its owner. Add a
`LICENSE` file and author/citation metadata before making a public release.
