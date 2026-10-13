# Utility scripts

- `generate_figure_1_response_rules.py`: builds the editable SVG and PDF for
  manuscript Figure 1.
- `generate_figure_2_nutrient_landscape.py`: the landscape the comparison runs
  on, and the three requirements that pick its wavenumber. Writes Figure 2 and
  `results/metrics/nutrient-landscape.csv`.
- `generate_figure_4_macroscopic_observables.py`: net polarity, centroid,
  spread and pulse shape along all three paths (Figure 4).
- `generate_figure_5_foraging_observables.py`: what one release measures and
  earns (Figure 5), from `config/foraging.toml`.
- `generate_figure_6_information_observables.py`: finite-lag navigation and
  information (Figure 6), from `config/navigation.toml`.
- `generate_manuscript_videos.py`: Supplementary Videos 1 and 2, from the
  archives `run.py compare` leaves in `results/data/`. Needs FFmpeg.
- `check_numerical_convergence.py`: refines time and spatial discretizations.
- `check_response_scaling.py`: checks linear-response scaling with paired
  microscopic simulations.
- `check_long_time_validation.py`: repeats the Figure 3 comparison over three
  independent seeds, which turns the paired sampling error into an error bar
  and makes the polar closure's own bias measurable. Writes
  `results/metrics/long-time-validation.csv`.
- `check_sampling_floor.py`: reruns the microscopic ensemble over several
  seeds and splits the micro-minus-theory discrepancy into a systematic part
  and a Monte-Carlo part, so a weak response can be told from a wrong one.
- `check_flux_dissipation.py`: measures the finite-volume face-flux numerical
  diffusivity against theory, comparing Rusanov, central, and the hybrid
  coefficient the solver uses.
- `check_long_time_coefficients.py`: integrates the closed density--polarity
  system directly and confirms the manuscript's long-time coefficients
  `D_eff`, `kappa`, and the centroid drift `V = kappa + delta D_eff'`.
- `inspect_case_magnitudes.py`: compares response magnitudes stored in NPZ
  archives.
- `benchmark_performance.py`: runs a bounded end-to-end benchmark and can
  sweep thread/batch settings without creating manuscript-sized archives.

Run scripts from `simulation/` unless a command explicitly says otherwise.
The Figure 1 generator is the exception: it resolves the repository root from
its own location and may be called from anywhere.
