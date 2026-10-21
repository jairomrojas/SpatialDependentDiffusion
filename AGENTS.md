# AGENTS.md — Spatial-Diffusion (cell foraging in a nutrient landscape)

Collaborator paper *Which foraging strategy? Reading microscopic motility rules from a population release* (Rojas, Brenner, El Hady) and its `chem_swimmer` code: microscopic position–heading simulation, a finite-volume density–polarity solver, and a weak-landscape Fourier–Green solution, compared across response Cases 0–9.

## Layout (`~/setup/rules/organizing_projects/organizing_repositories.md`)
- `manuscript/main.tex` — sole mathematical source of truth (REVTeX 4.2 `pre` reprint, a collaborator paper: keep its class; the house PRL note template does not apply). `manuscript/figures/` — Figures 1–6 and the appendix sheet (`figure-remaining-cases.pdf`).
- `model/solvers/chem_swimmer/` — the package (imported, never installed); `model/solvers/tests/` — scientific and regression suite; `model/parameters/` — `default.toml` (canonical experiment), `foraging.toml`, `navigation.toml`.
- `simulations/population_release/run.py` — CLI (`check`, `compare`, `grid`, `video`, `live`); its README is the reproduction guide.
- `analysis/manuscript/<figure>/run.py` (Figures 1, 2, 4, 5, 6, supplementary videos) and `analysis/simulations/<audit>/run.py` (convergence, sampling floor, response scaling, long-time checks, benchmark); index in `analysis/README.md`.
- `outputs/simulations/population_release/` — archives (`data/`), metrics, supplementary figures, videos, `provenance.toml`; ignored by Git. `outputs/analysis/...` — intermediates such as Figure 1's SVG.
- `sources/notes/` — `performance.md`, `repository_audit.md` (dated), `supplementary_video_legends.md`. `CONTRIBUTING.md` — the six-item scientific checklist; read it before changing numerics.

## Commands (from the repository root)
```bash
science_pytest -q --tb=short -- model/solvers/tests                          # scientific + regression suite (~3 min)
ruff check . && ruff format --check .        # CI pins ruff 0.17.1, the same as physics
cd manuscript && science_run latexmk -pdf main.tex                           # paper
science_run python -m simulations.population_release.run check --case all    # seconds
science_run python -m simulations.population_release.run compare             # full numerical comparison (long)
science_run python -m analysis.manuscript.figure_4_macroscopic_observables.run --reuse
```
`make paper|figures|test|lint` wrap the same commands portably for collaborators; on this workstation prefix them with `science_run`.
`python` is the shared `physics` interpreter. The `venv`/`pip` lines in `simulations/population_release/README.md` and CI are for external collaborators and GitHub only; never use them here.
Job purposes for `resources` live in root `science-jobs.json` (keys are module names, as `science_run` labels `python -m` jobs).
`cairosvg` (Figure 1's SVG-to-PDF step) is not in `physics`; adding it is the owner's decision (`pixi global add --environment physics cairosvg`).

## Invariants
- Equations and notation change in `manuscript/main.tex` first; code follows. Every scientific change follows the six-item checklist in `CONTRIBUTING.md` (identity/limit test, fixed seed, refinement in dt/grid/M/bandwidth, conservation and non-negativity, manuscript update, regenerate artifacts last).
- The density cache is validated by a SHA-256 of every numerical `.py` file in `model/solvers/chem_swimmer/` (all but `visualization/`, `cli.py`, `provenance.py`, `result_cache.py`; `result_cache._numerical_source_fingerprint`). Any byte change there, even a comment, invalidates the cached archives; never add files inside the package, and keep tests in `model/solvers/tests/`.
- Weak effects are judged with a signed, preselected observable and paired uncertainty, never from a case-specific colour scale or a norm alone.
- Memory: the microscopic archive (`outputs/simulations/population_release/data/pdf_cases_00_09.npz`, ~3.4 GB compressed) expands to ~4.8 GB in RAM because compressed NPZ cannot be memory-mapped; macro/analytic archives expand to ~2.9 GB each. Measure the peak and declare `science_run --mem` from it; never load two archives in one job without a measured peak.
- Storage: existing NPZ archives stay readable and are not bulk-converted. New array output is HDF5 sliced on read (`/home/inre/setup/rules/computing_and_storing_data/storing_scientific_data.md`); migrating an archive needs a measured read benchmark and the owner's approval. Archives embed their TOML, SHA-256 digest, and runtime versions; keep that provenance in any new format.
- Progress currently goes to `tqdm` (`model/solvers/chem_swimmer/progress.py`), which the `resources` monitor cannot read; long runs should also report through `science_progress.Progress` (optional import, no-op elsewhere; `/home/inre/setup/tools/running_work_safely/safe_job_launcher/progress_reporting.md`).
- Threads: WSL vCPU numbers do not identify P/E-cores; `taskset -c 0-7` only limits oversubscription. Thread and batch choices come from `analysis/simulations/performance_benchmark/` measurements.
- Figures in `manuscript/figures/` are one vector PDF each (`~/setup/rules/writing_papers/making_figures.md`); figure file names stay as they are, because Figure 3's name is written in the fingerprinted `cases.py`.

## Session continuity
Read `HANDOFF.md` (local, untracked), then `git status -sb`. Stage explicit paths only.

## Do / Ask / Never
- **Do:** run the test suite and lint after code changes; regenerate a curated figure only together with its supporting metrics from the same archive.
- **Ask:** before pushing to the shared remote (co-authors see it), choosing a license, publishing data, deleting local archives, or changing `model/parameters/default.toml`.
- **Never:** commit `outputs/`, videos, or frame sequences; pip/conda on this workstation; claim a weak effect without paired uncertainty and refinement; edit a fingerprinted module for cosmetic reasons.
