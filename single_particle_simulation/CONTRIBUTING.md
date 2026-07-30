# Contributing

## Development setup

Use an isolated Python 3.11 or newer environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
```

The package uses a `src/` layout. Install it in editable mode before running
the tests so imports exercise the packaged code path. `python run.py ...`
remains available as a convenience for collaborators who have installed the
runtime dependencies but not the package itself.

## Scientific change checklist

Changes to equations, response cases, stochastic updates, boundaries, density
estimation, or the finite-volume solver should include:

1. a mathematical identity or limiting case in `tests/`;
2. fixed-seed reproducibility where applicable;
3. a refinement check in \(dt\), grid spacing, \(M\), or bandwidth;
4. conservation/nonnegativity checks for probability densities;
5. an update to `MATHEMATICAL_AND_OUTPUT_GUIDE.md` when an output field or
   interpretation changes;
6. regenerated artifacts only after all checks pass.

Do not infer a weak physical effect from a case-specific color scale,
total-variation norm, or \(L^2\) norm alone. Use a signed preselected
observable with paired uncertainty, then check response scaling and numerical
refinement.

## Generated files

`output/`, caches, compiled Numba artifacts, and local virtual environments
are ignored by version control. Share large generated results as a release
artifact or data deposit, keeping the NPZ and CSV files together with the
figures/videos. Density NPZ files embed their TOML input and runtime
provenance.

## Public release

This repository does not choose a legal license on behalf of its owner. Add a
`LICENSE` file and author/citation metadata before making a public release.
