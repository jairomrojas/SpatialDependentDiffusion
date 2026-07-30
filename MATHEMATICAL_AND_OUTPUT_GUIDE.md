# Mathematical and output guide

This document explains what `single_particle_simulation` computes, how the
microscopic and macroscopic descriptions are related, and how to interpret
every generated output. It is intended to let a collaborator understand the
project without first reading the implementation.

## 1. The scientific question

The code studies a single active particle moving in a two-dimensional unit
disk. The particle senses a prescribed chemical field \(c(\mathbf x)\). Its
translational diffusion, rotational diffusion, propulsion speed, and tumble
rate can each respond either to the local concentration or to the directional
signal

\[
S(\mathbf x,\theta)
=\mathbf e(\theta)\mathbin{\cdot}\nabla c(\mathbf x),
\qquad
\mathbf e(\theta)=(\cos\theta,\sin\theta).
\]

The main question is whether the microscopic response mechanism can be
inferred from the evolving spatial probability density.

The program answers this in two ways:

1. `--micro` simulates many stochastic particles and estimates their density.
2. `--macro` solves the closed drift-diffusion equation derived in
   `Theory.pdf`.

Running `python run.py compare` performs both calculations, uses common scales
for fair visual comparison, and creates quantitative inference diagnostics.

## 2. Dimensionless variables

The disk radius \(R\) is the length unit and \(R^2/D_{T0}\) is the time unit:

\[
\mathbf x=\frac{\mathbf r_{\rm physical}}{R},
\qquad
t=\frac{t_{\rm physical}}{R^2/D_{T0}}.
\]

Consequently,

\[
|\mathbf x|\le 1,\qquad D_{T0}=1.
\]

All coordinates, times, velocities, rates, densities, figures, CSV files, and
NPZ arrays produced by the code are dimensionless. The value
`simulation.duration` is already the final dimensionless time; the program
does not convert it again.

## 3. Microscopic stochastic dynamics

For one time increment \(dt\), the code evaluates all coefficients at the
beginning of the step and applies the Itô update

\[
\Delta\mathbf x
=v\,\mathbf e(\theta)\,dt+\sqrt{2D_Tdt}\,\boldsymbol\xi_T,
\]

\[
\Delta\theta=\sqrt{2D_Rdt}\,\xi_R.
\]

A tumble occurs with probability

\[
P(\text{tumble in }dt)=1-\exp(-\lambda dt).
\]

The four kinetic coefficients obey the linear response law

\[
X(\mathbf x,\theta)
=X_0+X^c\,[c(\mathbf x)-c_0]+X^S S(\mathbf x,\theta),
\qquad
X\in\{D_T,D_R,v,\lambda\}.
\]

The response is linear only while every modulation remains small compared
with its corresponding background coefficient. The `check` command scans the
entire disk and all headings before simulation and reports these ratios.
Negative kinetic coefficients are rejected; they are never clipped.

### The nine cases

| Case | Nonzero sensitivity | Microscopic meaning | Contribution to the linear macroscopic mobility \(\kappa\) |
|---:|---|---|---:|
| 0 | none | homogeneous reference particle | \(0\) |
| 1 | \(D_T^c\) | translational diffusion changes with concentration | \(-D_T^c\) |
| 2 | \(D_R^c\) | rotational diffusion changes with concentration | \(0\) |
| 3 | \(v^c\) | speed changes with concentration | \(-v_0v^c/(2\gamma_0)\) |
| 4 | \(\lambda^c\) | tumble rate changes with concentration | \(0\) |
| 5 | \(D_T^S\) | translational diffusion depends on heading relative to the gradient | \(0\) |
| 6 | \(D_R^S\) | rotational diffusion depends on heading relative to the gradient | \(-v_0D_R^S/(2\gamma_0)\) |
| 7 | \(v^S\) | speed depends on heading relative to the gradient | \(v^S/2\) |
| 8 | \(\lambda^S\) | tumble rate depends on heading relative to the gradient | \(-v_0(1-a_1)\lambda^S/(2\gamma_0)\) |

Here

\[
\gamma_0=D_{R0}+\lambda_0(1-a_1)
\]

is the first angular-harmonic relaxation rate.

## 4. Macroscopic equation

The angular distribution is summarized by the density \(\rho\), polarity
\(\mathbf p\), and nematic tensor \(\mathbf Q\):

\[
\rho=\int_0^{2\pi}P\,d\theta,\qquad
\mathbf p=\int_0^{2\pi}\mathbf eP\,d\theta,
\]

\[
\mathbf Q
=\int_0^{2\pi}
\left(\mathbf e\mathbf e^{\mathsf T}-\frac12\mathbf I\right)P\,d\theta.
\]

`Theory.pdf` assumes:

- a small chemical response amplitude \(\epsilon\);
- a small persistence-to-field-length ratio
  \(\delta=\ell_p/L\), where \(\ell_p=v_0/\gamma_0\);
- times long compared with \(1/\gamma_0\).

At first order in \(\epsilon\) and leading order in \(\delta\), the polarity is
adiabatically eliminated and the density obeys

\[
\partial_t\rho
=\nabla\mathbin{\cdot}
\left(D_{\rm eff}\nabla\rho-\kappa\rho\nabla c\right),
\]

\[
D_{\rm eff}=D_{T0}+\frac{v_0^2}{2\gamma_0},
\]

\[
\kappa
=\frac{v^S}{2}-D_T^c
-\frac{v_0}{2\gamma_0}
\left[v^c+D_R^S+(1-a_1)\lambda^S\right].
\]

The no-flux macroscopic boundary condition is

\[
\mathbf n\mathbin{\cdot}
\left(-D_{\rm eff}\nabla\rho+\kappa\rho\nabla c\right)=0.
\]

### Where the three zero cases cancel

Before closing the angular moments, the translational part of the exact
density equation contains

\[
\nabla^2\left[
D_{T0}\rho+D_T^c(c-c_0)\rho
+D_T^S\,\mathbf p\mathbin{\cdot}\nabla c
\right].
\]

This identity makes the Case 5 cancellation precise. In an isotropic,
uniform bulk reference state, \(\mathbf p^{(0)}=0\), so the term proportional
to \(D_T^S\) cannot force the scalar density at first order. Cases 2 and 4 do
not occur in this exact density equation at all; they modify the angular
relaxation equation for \(\mathbf p\). Multiplying an isotropic
\(\mathbf p^{(0)}=0\) by a concentration-dependent relaxation rate also
cannot create a first-order bulk density source.

These are local bulk statements about the reference state used in the
asymptotic expansion. They do not prove that the complete bounded
microscopic process has zero first derivative with respect to those response
coefficients.

The macro solver uses a conservative finite-volume method with
exponential-fitting face fluxes. It preserves total probability, remains
nonnegative under its reported timestep restriction, and preserves the
discrete equilibrium

\[
\rho_{\rm eq}\propto\exp(\kappa c/D_{\rm eff}).
\]

## 5. Why Cases 2, 4, and 5 can look too large in a microscopic movie

The zero entries in the table mean "zero at the order retained by the
closure." They do not all mean the same thing.

- Cases 2 and 4 cannot select a preferred direction in an initially uniform,
  isotropic bulk. Their bulk spatial density is therefore identical to Case 0
  at the retained order. A sliding active wall can nevertheless create a
  kinetic boundary layer because particles keep headings that point into the
  wall; changing \(D_R\) or \(\lambda\) changes how quickly they escape that
  layer.
- Case 5 first reaches the *isotropic bulk* density through
  \(D_T^S\mathbf p\mathbin{\cdot}\nabla c\), which is higher order when the
  bulk reference polarity is zero. At a sliding active wall, however, the
  baseline kinetic distribution has a nonzero orientation-resolved boundary
  layer \(\mathbf p_{\rm wall}^{(0)}\). The product
  \(D_T^S\mathbf p_{\rm wall}^{(0)}\mathbin{\cdot}\nabla c\) is then linear in
  \(D_T^S\). The full microscopic Case 5 response can consequently be
  first-order even though its retained bulk mobility is zero.

The macroscopic no-flux condition is imposed *after* eliminating the angular
boundary layer. It is not an exact replacement for a microscopic sliding,
specular, or diffuse collision law. Comparing micro and macro close to the
wall is therefore a model-discrepancy test as well as a discretization test.

There is also an important estimator effect. A pixelwise magnitude such as

\[
\frac12\int|\rho_k-\rho_0|\,d^2x
\]

is positive even when two true densities are identical. Cases 2 and 4 change
orientation randomization, so paired particle paths can decorrelate rapidly
from Case 0. Their histogram-difference noise can then be larger than the
noise of a weak but real Case 3 response. A case-specific movie range expands
that noise to fill the full color palette and can make it look like a large
physical effect.

For this reason:

- the standard difference movies use one common scale and are suitable for
  magnitude comparison;
- the `detail` movies use a separate scale per case and are only for inspecting
  morphology;
- physical ranking should use the signed
  \(\Delta\langle c\rangle\) observable and its paired standard error in
  `microscopic_mechanism_summary.csv`.

The default full-ensemble validation shows that Case 3 is strongly resolved,
Case 4 is consistent with zero, and Case 5 is resolved but smaller than Case
3. At \(t=0.5\), the absolute signed
\(\Delta\langle c\rangle\) ratios relative to Case 3 are approximately
0.35, 0.21, and 0.26 for Cases 2, 4, and 5. Thus none is larger than Case 3
under the signed common-scale observable, even though a detail movie or
total-variation norm can suggest otherwise. Case 2 falls below the
eight-comparison familywise 95-percent threshold; Case 4 is plainly
unresolved; Case 5 is strongly resolved.

The symmetric response-scaling audit in
`output/data/response_scaling_audit.csv` uses \(M=16384\) and decomposes
\(F(\alpha)-F(0)\) into

\[
F_{\rm odd}(\alpha)=\frac{F(\alpha)-F(-\alpha)}{2},\qquad
F_{\rm even}(\alpha)=\frac{F(\alpha)+F(-\alpha)}{2}.
\]

For the default sliding wall, doubling \(|\alpha|\) from 0.5 to 1 changes the
odd final \(\Delta\langle c\rangle\) as follows:

| Case | odd response at 0.5 | odd response at 1 | interpretation |
|---:|---:|---:|---|
| 2 | \(0.001215\pm0.000448\) | \(0.001775\pm0.000741\) | weak/borderline; not a clean asymptotic scaling law |
| 3 | \(0.003096\pm0.000119\) | \(0.006209\pm0.000233\) | robust linear response |
| 4 | \(-0.002143\pm0.001210\) | \(-0.000973\pm0.001240\) | unresolved and not stable in scale |
| 5 | \(-0.000768\pm0.000066\) | \(-0.001559\pm0.000130\) | robust linear wall-layer response |

The Case 5 even component is consistent with zero at both scales. This is the
decisive evidence that its observed full-disk microscopic signal is dominated
by a linear contribution absent from the bulk closure, rather than by the
quadratic bulk term alone. Specular-wall paired estimates have much larger
variance because heading reflections rapidly decorrelate perturbed paths;
they are retained in the CSV as a sensitivity analysis, not treated as a
high-precision null experiment.

## 6. Automatic numerical resolution

Let \(\ell_{\min}\) be the smallest of the disk radius, every built-in field
width, and an explicitly supplied custom-field length. The target spatial
increment is

\[
\Delta x_{\rm target}
=f\,\ell_{\min},
\]

where \(f=\texttt{numerics.max_spatial_step_fraction}\).

The automatic microscopic timestep is the smallest candidate from:

\[
dt_{\rm adv}
=\frac{\Delta x_{\rm target}}{v_{\max}},
\]

\[
dt_{\rm trans}
=\frac{\Delta x_{\rm target}^2}{4D_{T,\max}},
\]

\[
dt_{\rm rot}
=\frac{\theta_{\rm rms,max}^2}{2D_{R,\max}},
\]

\[
dt_{\rm tumble}
=\frac{(\lambda dt)_{\max}}{\lambda_{\max}}.
\]

The selected value is adjusted slightly so that an integer number of steps
reaches the requested final time exactly. The terminal report prints every
formula, candidate value, and limiting process.

## 7. Ensemble size and density estimation

The microscopic density is estimated from \(M\) particles. For a Gaussian KDE
bandwidth \(bR\), a uniform unit-disk density contributes approximately

\[
N_{\rm kernel}\simeq 2Mb^2
\]

particles to one kernel footprint. The code uses this as an occupancy floor:

\[
M
=2^{\left\lceil
\log_2\left[
\frac{N_{\rm target}}{2b^2}
\right]\right\rceil}.
\]

With the default \(b=0.05\) and \(N_{\rm target}=80\), this gives
\(M=16384\).

This rule is not an optimal-bandwidth calculation and is not a convergence
proof. KDE bias depends on spatial derivatives of the unknown density and is
especially delicate at a boundary. Collaborator-facing conclusions must be
stable when \(M\) is increased and the absolute and difference bandwidths are
varied.

The simulator processes particles in bounded batches. Random increments are
generated once per particle and timestep and shared across Cases 0-8. Saved
positions, headings, and tumble counts are immediately reduced into
histograms and ensemble moments; individual trajectories are not written to
disk.

## 8. Output directory: file-by-file interpretation

All paths below are relative to `single_particle_simulation`.

### `output/figures/chemical_field.png`

The prescribed dimensionless chemical landscape \(c(x,y)\) on the unit disk.
Use it to locate maxima, minima, and gradient directions before interpreting
particle accumulation.

### `output/figures/density_inference_observables.png`

A four-panel comparison derived from the micro and macro density fields:

- signed change in mean chemical exposure
  \(\Delta\langle c\rangle\);
- its time integral;
- the theoretical mobility \(\kappa\) for each case;
- macroscopic half-time, fitted relaxation time, and 90-percent response time.

Solid lines are microscopic and dashed lines are macroscopic. This figure uses
common axes and is appropriate for comparing magnitudes.

### `output/figures/microscopic_mechanism_observables.png`

Aggregate microscopic diagnostics with paired 95-percent Monte Carlo
confidence intervals. It contains:

- signed \(\Delta\langle c\rangle\) for Cases 2, 3, 4, and 5;
- local polarization
  \(\Delta\langle\mathbf e\cdot\nabla c\rangle\);
- displacement variance;
- gradient-aligned nematic order;
- second-harmonic heading memory;
- cumulative tumble counts.

These observables help separate mechanisms that are degenerate in the closed
density equation, especially Cases 1/3 and Cases 6/8.

### `output/data/density_observables_summary.csv`

One row per case containing:

- \(\kappa\) and \(\kappa/D_{\rm eff}\);
- equilibrium changes in mean concentration, centroid, and total variation;
- \(t_{50}\), \(t_{90}\), and the fitted relaxation time;
- angular, field-diffusion, and disk-mixing reference times.

### `output/data/density_observables_timeseries.csv`

One row per method, case, and saved time. Columns contain signed mean chemical
exposure, cumulative exposure, centroid shifts, total variation, and the
\(L^2\) distance from Case 0.

Total variation and \(L^2\) are useful morphology measures but have a positive
Monte Carlo noise floor. Signed observables are safer for deciding whether a
weak response is physically resolved.

### `output/data/microscopic_mechanism_summary.csv`

The concise statistical audit for the microscopic cases. It reports:

- theoretical \(\kappa\);
- whether the linear closed equation predicts zero or nonzero density response;
- final signed \(\Delta\langle c\rangle\);
- its paired standard error and \(z\)-score;
- the 95-percent familywise \(z\)-threshold for all eight case comparisons;
- magnitude relative to Case 3;
- whether the preselected final-time response is resolved at familywise
  95-percent confidence.

This is the first file to inspect when a nominally zero case looks large in a
movie.

### `output/data/microscopic_mechanism_observables.csv`

The full time series for every aggregate microscopic observable. Each value is
followed by the standard error of the paired Case-minus-Case-0 difference.
The raw Case 0 value is retained, so collaborators can reconstruct either
absolute values or differences.

### `output/data/response_scaling_audit.csv`

The perturbative-order and wall-law audit generated by
`scripts/audit_response_scaling.py`. It contains one row per wall law, case,
response magnitude, odd/even/raw component, and spatial observable. The
important fields are:

- `scale_magnitude`: \(|\alpha|\), multiplying the configured sensitivity;
- `component`: positive raw, negative raw, odd, or even response;
- `estimate` and `paired_standard_error`;
- `core_concentration_mass` and `wall_concentration_mass`, which add to the
  mean concentration and localize changes relative to `core_radius`;
- `wall_probability`;
- \(M\), \(dt\), number of steps, core radius, and seed.

Use the odd component to test linear scaling and the even component to detect
quadratic corrections. A component should scale with \(\alpha\) or
\(\alpha^2\), respectively, and remain larger than its paired uncertainty
before assigning it a perturbative order.

### `output/data/numerical_convergence_audit.csv`

The refinement evidence generated by
`scripts/audit_numerical_convergence.py`. Its `micro_timestep` rows report the
final signed \(\Delta\langle c\rangle\) and paired standard error for Cases
0, 2, 3, 4, and 5 at the automatic \(dt\), \(dt/2\), and \(dt/4\). The
`macro_equilibrium_grid` rows report the observed equilibrium
\(\Delta\langle c\rangle\) on grids from \(81^2\) through \(321^2\).

In the saved default audit, Cases 2 and 4 change sign under timestep
refinement and remain smaller than their uncertainty. Cases 3 and 5 retain
their sign and compatible magnitude. The Case 3 macroscopic equilibrium
changes by less than \(6\times10^{-8}\) over the reported grids; the stronger
Case 7 changes by about \(2\times10^{-5}\).

### `output/data/pdf_cases_00_08.npz`

Compressed microscopic particle/KDE data. Important arrays are:

- `time`: saved dimensionless times;
- `coordinates`: cell-center \(x\) and \(y\) coordinates;
- `density`: absolute KDE, shape `(case, time, y, x)`;
- `difference_density`: smoother KDE used for Case-minus-Case-0 movies;
- `case_numbers`, `ensemble_size`, `dt`, and `seed`;
- absolute and difference bandwidths;
- names, values, and paired standard errors of the aggregate microscopic
  observables;
- `method="micro"` and the configuration source;
- `schema_version`, `software_version`, the complete `config_text`, its
  `config_sha256`, and a compact JSON record of Python/NumPy/Numba/platform
  versions.

Every density slice integrates to one over the unit disk.

### `output/data/pdf_macro_cases_00_08.npz`

The corresponding closed-equation density data. It has the same density,
coordinate, time, bandwidth, and metadata structure, but
`method="macro"` and `ensemble_size=0` because no particle sampling is used.
The microscopic-observable arrays are empty because the closed equation no
longer contains heading or tumble information.

### `output/data/msd_all_cases.csv`

Mean-squared displacement data produced by the trajectory-comparison workflow.
It contains dimensionless time and ensemble displacement statistics for the
response cases. It is independent of the PDF movie and is useful for checking
effective diffusion.

### `output/videos/probability_density_cases_00_08.mp4`

Microscopic absolute density \(\rho(x,y,t)\) for Cases 0-8. Every panel and
time uses the fixed display range 0 to 1. Values above 1 are valid and simply
saturate in the rendered movie; the NPZ data are never clipped or normalized
to this range.

### `output/videos/probability_density_macro_cases_00_08.mp4`

The same absolute-density view for the closed macroscopic equation.

### `output/videos/probability_density_difference_cases_00_08.mp4`

Microscopic signed differences

\[
\Delta\rho_k=\rho_k-\rho_0.
\]

All cases, times, and both methods use one synchronized linear range. This is
the difference movie to use for fair magnitude comparisons.

### `output/videos/probability_density_difference_macro_cases_00_08.mp4`

The synchronized macroscopic counterpart. Cases 2, 4, and 5 are exactly zero
in this movie because their retained-order mobility is zero.

### `output/videos/probability_density_difference_detail_cases_00_08.mp4`

Microscopic morphology-focused differences. Each case has its own fixed linear
range over the complete movie. It reveals weak spatial structure but must not
be used to compare response magnitudes across cases.

### `output/videos/probability_density_difference_detail_macro_cases_00_08.mp4`

The corresponding morphology-focused macroscopic movie.

Other optional commands can create `output/videos/case_XX.mp4`, a single
particle trajectory movie for Case `XX`.

## 9. Source-code map

| File | Responsibility |
|---|---|
| `run.py` | Runs the package directly from the repository. |
| `configs/default.toml` | The only routine input file: physics, accuracy, sampling, and rendering. |
| `src/chem_swimmer/config.py` | Typed configuration loading and validation. |
| `fields/` | Chemical field values and gradients. |
| `kinetics.py` | Linear response coefficients and macroscopic \(D_{\rm eff},\kappa\). |
| `numerics.py` | Field-length scan and automatic microscopic timestep. |
| `boundary.py` | Disk intersection and sliding, specular, or diffuse reflection. |
| `simulator.py` | Readable reference stochastic integrator. |
| `accelerated.py` | Numba/LLVM fused kernels and bounded-batch ensemble execution. |
| `density.py` | Particle histograms, KDE, aggregate observables, and NPZ output. |
| `macroscopic.py` | Conservative finite-volume solution of the closed equation. |
| `visualization/density_video.py` | Fast direct-raster absolute and difference movies. |
| `visualization/inference_plot.py` | Density timescales, mechanism observables, CSV reports, and uncertainty. |
| `scripts/diagnose_case_magnitudes.py` | Separates core/wall TV texture and prints the final signed response with its paired \(z\)-score from a microscopic NPZ. |
| `scripts/audit_response_scaling.py` | Reverses/scales Cases 2-5, separates odd/even perturbative components, and compares wall laws. |
| `scripts/audit_numerical_convergence.py` | Writes microscopic \(dt\)-refinement and macroscopic equilibrium grid-refinement evidence. |
| `diagnostics.py` | Parameter and theory-regime terminal reports. |
| `tests/test_mathematics.py` | Mathematical identities, conservation, reproducibility, and engine-equivalence tests. |

## 10. Recommended collaborator workflow

From `single_particle_simulation`, run:

```bash
python run.py check --case all
python run.py compare
```

Then inspect the results in this order:

1. `chemical_field.png`;
2. `microscopic_mechanism_summary.csv`;
3. `density_inference_observables.png`;
4. the two common-scale difference movies;
5. the detail movies only when spatial morphology is the question;
6. the NPZ and time-series CSV files for further analysis.

When changing parameters, verify convergence by reducing \(dt\), increasing
\(M\), or both. A response is physically persuasive when a signed observable
is stable under refinement and is large compared with its paired standard
error. A colorful case-specific panel alone is not evidence of a resolved
effect.

For a compact audit of an existing microscopic NPZ, run:

```bash
python scripts/diagnose_case_magnitudes.py output/data/pdf_cases_00_08.npz
```

For the substantially more expensive perturbative-order and wall-law audit,
run:

```bash
python scripts/audit_response_scaling.py --ensemble 16384
```

For the saved timestep and grid-refinement report, run:

```bash
python scripts/audit_numerical_convergence.py --ensemble 4096
```
