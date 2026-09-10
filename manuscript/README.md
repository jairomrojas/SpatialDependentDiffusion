# Manuscript

`main.tex` is the canonical scientific document and `main.pdf` is its compiled
form. The body carries six figures:

- `figures/figure-1-response-rules.pdf`: conceptual map of the nine mechanisms;
- `figures/figure-2-nutrient-landscape.pdf`: the single cosine mode the
  comparison runs on, and the three requirements that select it, written by
  `../simulation/scripts/generate_figure_2_nutrient_landscape.py`;
- `figures/figure-3-case-1-validation.pdf`: Case 1 numerical validation;
- `figures/figure-4-macroscopic-observables.pdf`: net polarity, centroid,
  spread and pulse shape along all three paths, written by
  `../simulation/scripts/generate_figure_4_macroscopic_observables.py`;
- `figures/figure-5-foraging-observables.pdf`: the drift/mobility pair read
  off one landscape mode, nutrient uptake, the two relaxation clocks, and the
  divergence between strategies, written by
  `../simulation/scripts/generate_figure_5_foraging_observables.py` from
  `../simulation/config/foraging.toml`;
- `figures/figure-6-information-observables.pdf`: the finite-lag navigation
  channel and the information a cell's motion carries about its state, written
  by `../simulation/scripts/generate_figure_6_information_observables.py` from
  `../simulation/config/navigation.toml`.

A further sheet in the appendix collects the remaining eight mechanisms. The
SVG beside Figure 1 is its editable vector source. Figures 2--6 are generated
by the simulation pipeline. Additional scientific figures are indexed in
[`../simulation/results/README.md`](../simulation/results/README.md).

Compile from this directory with `latexmk -pdf main.tex`. Temporary LaTeX
files should not be committed; `main.pdf` is retained for collaborators who
do not have the REVTeX toolchain installed.
