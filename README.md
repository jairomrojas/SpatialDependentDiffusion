# Cell foraging in a nutrient landscape

Research repository for the manuscript *Which foraging strategy? Reading
microscopic motility rules from a population release*.

## Start here

- [`manuscript/main.tex`](manuscript/main.tex) is the sole mathematical source
  of truth; [`manuscript/main.pdf`](manuscript/main.pdf) is the compiled paper.
- [`simulation/README.md`](simulation/README.md) explains the numerical model
  and reproduction workflow.
- [`simulation/results/README.md`](simulation/results/README.md) indexes the
  curated supplementary figures and validation metrics.

## Layout

```text
.
├── manuscript/                 paper source, PDF, figures, and references
│   ├── main.tex
│   ├── main.pdf
│   ├── figures/                generated Figures 1--6
│   └── references/             locally retained source literature
└── simulation/                 installable Python package and numerical work
    ├── config/default.toml     canonical paper experiment
    ├── src/chem_swimmer/       model, solvers, diagnostics, visualization
    ├── scripts/                figure generation and focused audits
    ├── tests/                  mathematical and regression checks
    └── results/                curated supplementary outputs and metrics
```

## Common tasks

Compile the paper:

```bash
cd manuscript
latexmk -pdf main.tex
```

Install and test the simulation:

```bash
cd simulation
pixi global list
PYTHONPATH=src python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

See [`simulation/README.md`](simulation/README.md) for the `venv` fallback,
performance-core pinning, benchmarking, and the full reproduction workflow.

Regenerate the standalone figures from the repository root. Figures 3 and the
Cases 2--9 appendix sheet come from `simulation/run.py compare` instead.

```bash
# Figure 1: the nine response rules
python simulation/scripts/generate_figure_1_response_rules.py
# Figure 2: the nutrient landscape, and why this mode
python simulation/scripts/generate_figure_2_nutrient_landscape.py
# Figure 4: the macroscopic-observable comparison
python simulation/scripts/generate_figure_4_macroscopic_observables.py
# Figure 5: what one release measures and earns
python simulation/scripts/generate_figure_5_foraging_observables.py
# Figure 6: finite-lag navigation and information
python simulation/scripts/generate_figure_6_information_observables.py
# Supplementary Videos 1 and 2 (needs ffmpeg)
python simulation/scripts/generate_manuscript_videos.py
```

Large raw arrays, videos, caches, and frame-by-frame diagnostics are generated
locally and intentionally excluded from the curated repository.
