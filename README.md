# Cell foraging in a nutrient landscape

Research repository for the manuscript *Which foraging strategy? Reading
microscopic motility rules from a population release*.

## Start here

- [`manuscript/main.tex`](manuscript/main.tex) is the sole mathematical source
  of truth; [`manuscript/main.pdf`](manuscript/main.pdf) is the compiled paper.
- [`simulations/population_release/README.md`](simulations/population_release/README.md)
  explains the numerical model and reproduction workflow.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) is the checklist for every scientific change.

## Layout

The folders follow one convention: code that defines the model, code that runs
it, code that measures its output, and what that code produces.

```text
.
├── manuscript/                     paper source, PDF, and Figures 1--6 plus the appendix sheet
├── model/
│   ├── solvers/chem_swimmer/       model, solvers, diagnostics, visualization (imported, never run)
│   ├── solvers/tests/              mathematical and regression checks
│   └── parameters/                 default.toml (canonical paper experiment), foraging.toml, navigation.toml
├── simulations/population_release/ run.py: check, compare, grid, video, live
├── analysis/
│   ├── manuscript/                 one folder per figure generator, and the supplementary videos
│   └── simulations/                one folder per audit (convergence, sampling floor, benchmark, ...)
├── outputs/                        everything the code writes (archives, metrics, figures, videos); not in Git
└── sources/notes/                  performance profile, engineering audit, video legends
```

## Common tasks

Every command runs from the repository root; the code is imported, not installed.

Compile the paper:

```bash
cd manuscript
latexmk -pdf main.tex
```

Test and lint the code:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

`make paper|figures|test|lint|verify|benchmark` wraps the same commands.
See [`simulations/population_release/README.md`](simulations/population_release/README.md)
for the dependencies, thread and batch benchmarking, and the full reproduction workflow.

Regenerate the standalone figures. Figure 3 and the Cases 2--9 appendix sheet
come from `python -m simulations.population_release.run compare` instead.

```bash
# Figure 1: the nine response rules (needs cairosvg for the PDF)
python -m analysis.manuscript.figure_1_response_rules.run
# Figure 2: the nutrient landscape, and why this mode
python -m analysis.manuscript.figure_2_nutrient_landscape.run
# Figure 4: the macroscopic-observable comparison
python -m analysis.manuscript.figure_4_macroscopic_observables.run
# Figure 5: what one release measures and earns
python -m analysis.manuscript.figure_5_foraging_observables.run
# Figure 6: finite-lag navigation and information
python -m analysis.manuscript.figure_6_information_observables.run
# Supplementary Videos 1 and 2 (needs ffmpeg)
python -m analysis.manuscript.supplementary_videos.run
```

Large raw arrays, videos, caches, and frame-by-frame diagnostics are written
below `outputs/` and intentionally excluded from the repository.
