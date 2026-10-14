# AGENTS.md — Spatial-Diffusion (cell foraging in a nutrient landscape)

Collaborator paper *Which foraging strategy? Reading microscopic motility rules from a population release* (Rojas, Brenner, El Hady) and its `chem_swimmer` package: microscopic position–heading simulation, a finite-volume density–polarity solver, and a weak-landscape Fourier–Green solution, compared across response Cases 0–9.

## Layout
- `manuscript/main.tex` — sole mathematical source of truth (REVTeX 4.2 `pre` reprint, a collaborator paper: keep its class; the house PRL note template does not apply). `manuscript/figures/` — Figures 1–6 and the appendix sheet.
- `simulation/` — installable package (`src/chem_swimmer/`), `config/default.toml` (canonical experiment), `run.py` (`check`, `compare`, `grid`, `video`, `live`), `scripts/` (figure generators, audits, benchmark), `tests/`, curated `results/` (figures, metrics, `provenance.toml`).
- Read `simulation/README.md` and `simulation/CONTRIBUTING.md` before changing numerics.

## Commands
```bash
cd simulation && PYTHONPATH=src science_pytest -q --tb=short --maxfail=1 -- tests   # scientific + regression suite
cd simulation && ruff check . && ruff format --check .   # lint; pyproject pins ruff 0.16.6, physics has 0.17.0: formatting diffs may be version noise
cd manuscript && science_run latexmk -pdf main.tex                                  # paper
science_run python simulation/run.py check --case all                               # from repo root
science_run python simulation/run.py compare                                        # full numerical comparison (long)
science_run python simulation/scripts/generate_figure_4_macroscopic_observables.py --reuse
```
`make paper|figures|test|lint` wrap the same commands portably for collaborators; on this workstation prefix them with `science_run`.
`python` is the shared `physics` interpreter. The `venv`/`pip` instructions in `simulation/README.md` are for external collaborators only; never use them here.
Job purposes for `resources` live in root `science-jobs.json`.

## Invariants
- Equations and notation change in `manuscript/main.tex` first; code follows. Every scientific change follows the six-item checklist in `simulation/CONTRIBUTING.md` (identity/limit test, fixed seed, refinement in dt/grid/M/bandwidth, conservation and non-negativity, manuscript update, regenerate artifacts last).
- Weak effects are judged with a signed, preselected observable and paired uncertainty, never from a case-specific colour scale or a norm alone.
- Memory: the microscopic archive (`results/data/pdf_cases_00_09.npz`, ~3.4 GB compressed) expands to ~4.8 GB in RAM because compressed NPZ cannot be memory-mapped; macro/analytic archives expand to ~2.9 GB each. The shared science.slice caps a job at 6 GiB, so never load two archives in one job and measure the peak before declaring `--mem`.
- Storage: existing NPZ archives stay readable and are not bulk-converted. New array output is HDF5 sliced on read (`/home/inre/setup/rules/writing_scientific_code.md` §13, `scientific-data-storage.md`); migrating an archive needs a measured read benchmark and the owner's approval. Archives embed their TOML, SHA-256 digest, and runtime versions; keep that provenance in any new format.
- Progress currently goes to `tqdm` (`src/chem_swimmer/progress.py`), which the `resources` monitor cannot read; long runs should also report through `science_progress.Progress` (optional import, no-op elsewhere; `/home/inre/setup/tools/safe_job_launcher/progress.md`).
- Threads: WSL vCPU numbers do not identify P/E-cores; `taskset -c 0-7` only limits oversubscription. Thread and batch choices come from `scripts/benchmark_performance.py` measurements.
- Figures in `manuscript/figures/` are vector PDFs for the collaborator paper; the owner's PNG-only rule for internal notes does not override the co-authors' format without the owner's decision.

## Session continuity
No `HANDOFF.md` yet; create one (local, untracked) at the first milestone. Start with `git status -sb`: the worktree has ~90 uncommitted changes, including deleted docs and scripts; preserve them. Stage explicit paths only.

## Do / Ask / Never
- **Do:** run the test suite and lint after code changes; regenerate a curated figure only together with its supporting metrics from the same archive.
- **Ask:** before pushing to the shared remote, choosing a license, publishing data, deleting local archives, or changing `config/default.toml`.
- **Never:** commit `results/data/`, videos, or frame sequences; pip/conda on this workstation; claim a weak effect without paired uncertainty and refinement.
