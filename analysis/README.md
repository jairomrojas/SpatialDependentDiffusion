# Figure generators and audits

One folder per script, each with a `run.py`; run them as modules from the repository root
(`python -m analysis.<group>.<folder>.run`). `R/` below stands for
`outputs/simulations/population_release/`, the shared output store.

- `manuscript/figure_1_response_rules/`: draws manuscript Figure 1 as SVG
  (kept in `outputs/analysis/manuscript/figure_1_response_rules/`) and converts it to the PDF.
- `manuscript/figure_2_nutrient_landscape/`: the landscape the comparison runs
  on, and the three requirements that pick its wavenumber. Writes Figure 2 and
  `R/metrics/nutrient-landscape.csv`.
- `manuscript/figure_4_macroscopic_observables/`: net polarity, centroid,
  spread and pulse shape along all three paths (Figure 4).
- `manuscript/figure_5_foraging_observables/`: what one release measures and
  earns (Figure 5), from `model/parameters/foraging.toml`.
- `manuscript/figure_6_information_observables/`: finite-lag navigation and
  information (Figure 6), from `model/parameters/navigation.toml`.
- `manuscript/supplementary_videos/`: Supplementary Videos 1 and 2, from the
  archives `simulations.population_release.run compare` leaves in `R/data/`. Needs FFmpeg.
- `simulations/numerical_convergence/`: refines time and spatial discretizations.
- `simulations/response_scaling/`: checks linear-response scaling with paired
  microscopic simulations.
- `simulations/long_time_validation/`: repeats the Figure 3 comparison over three
  independent seeds, which turns the paired sampling error into an error bar
  and makes the polar closure's own bias measurable. Writes
  `R/metrics/long-time-validation.csv`.
- `simulations/sampling_floor/`: reruns the microscopic ensemble over several
  seeds and splits the micro-minus-theory discrepancy into a systematic part
  and a Monte-Carlo part, so a weak response can be told from a wrong one.
- `simulations/flux_dissipation/`: measures the finite-volume face-flux numerical
  diffusivity against theory, comparing Rusanov, central, and the hybrid
  coefficient the solver uses.
- `simulations/long_time_coefficients/`: integrates the closed density--polarity
  system directly and confirms the manuscript's long-time coefficients
  `D_eff`, `kappa`, and the centroid drift `V = kappa + delta D_eff'`.
- `simulations/case_magnitudes/`: compares response magnitudes stored in NPZ
  archives.
- `simulations/performance_benchmark/`: runs a bounded end-to-end benchmark and can
  sweep thread/batch settings without creating manuscript-sized archives.

Every script resolves the repository root from its own location.
