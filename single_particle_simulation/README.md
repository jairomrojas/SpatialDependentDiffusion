# Chemically responsive active swimmer

This project implements the microscopic dynamics of equation (6) in
`Theory.pdf` inside a two-dimensional disk. It supports the baseline particle
and each of the eight canonical response mechanisms, an interactive
single-trajectory view, a single-case MP4, and synchronized 3-by-3 movies of
the evolving spatial probability density for Cases 0-8.

The code keeps a readable Python reference and automatically selects a
Numba/LLVM native kernel for built-in chemical fields. NumPy pre-generates the
common random innovations, Numba compiles the state-dependent time loop to
Arm64 machine code, and independent density replicas run on native CPU
threads. The density movie bypasses Matplotlib and pipes raster frames
directly to FFmpeg.

## Quick start on the Mac

Use Python 3.11 or newer. The command-line-tools Python shipped by macOS may be
older, so use a Homebrew, Pixi, Conda, or `uv` Python.

```bash
cd single_particle_simulation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
chem-swimmer check --case all
chem-swimmer live --case 8
chem-swimmer video --case 8
chem-swimmer compare
```

Without installing the package, use the same environment and run:

```bash
python run.py check --case all
python run.py live --case 8
python run.py video --case 8
python run.py compare
```

Every long stage prints a terminal progress meter with elapsed time, rate, and
estimated time remaining. Add `--no-progress` for batch logs.

FFmpeg is needed only for MP4 output:

```bash
brew install ffmpeg
```

On Apple Silicon, `codec = "auto"` selects the hardware
`h264_videotoolbox` encoder when available.

## The one file most users edit

All routine choices live in
[`configs/default.toml`](configs/default.toml):

- duration, automatic/manual time step, random seed, and saved-data interval;
- spatial and temporal accuracy targets for automatic resolution;
- disk radius and wall law;
- initial position and heading;
- `v0`, `D_T0`, `D_R0`, `lambda0`, and `a1`;
- all eight response sensitivities;
- chemical-field shape;
- video frame rate, trail length, and quality.
- probability-density grid, bandwidth, automatically selected ensemble size,
  time slices, and video resolution;
- native engine, thread count, bounded-memory batch sizes, and progress display.

Running `--case N` activates only sensitivity `N` and sets the other seven to
zero. `--case 0` is the unresponsive baseline. Individual trajectories start
at `initial.position` with `initial.heading`; density ensembles use uniform
positions and isotropic headings by default, as assumed by the closed
equation. `compare` runs Cases 0-8 with the same
standard-normal Wiener innovations and the same tumble-test uniforms, making
mechanism-to-mechanism differences easier to interpret. The mechanisms still
scale those common innovations by their own state-dependent diffusivities and
rates, as equation (6) requires.

Before a long simulation, run:

```bash
python run.py check --case all
```

The default profile is perturbative and designed to make the case PDFs
distinguishable:

```bash
python run.py check --case all
python run.py compare
```

This profile keeps every local kinetic modulation at or below 15%, has
`ell_p/ell_min` close to 0.15, and uses `M=16384`.

It scans the disk and headings, verifies that the linear response remains
physical, reports the ranges of all kinetic coefficients, checks
every timestep constraint, and prints the manuscript's `gamma0`, `D_eff`,
and `kappa`. Zero `v0`, `D_R0`, and `lambda0` are valid simulation inputs. When
`gamma0 = 0`, the heading does not relax, so the simulator still runs but the
long-time isotropic `D_eff` and `kappa` diagnostics are reported as unavailable.

## Radius/diffusion scaling and automatic resolution

Every dynamical value in the TOML file is already dimensionless. If starting
from dimensional measurements, convert them once using

```text
x = r_physical/R,                     disk: |x| <= 1,
t = t_physical/(R^2/D_T0),
v = v_physical R/D_T0,
D_T = D_T_physical/D_T0,
D_R = D_R_physical R^2/D_T0,
lambda = lambda_physical R^2/D_T0.
```

Thus the computational disk radius and baseline translational diffusivity are
both exactly one, and `simulation.duration` is directly the final `t`.
Trajectory arrays, MSD data, density data, diagnostics, and every plot use
only these variables; dimensional duplicates are not saved.

With `simulation.dt = "auto"`, the code scans all requested mechanisms over
the disk and chooses the largest step satisfying the configured limits for
advection, translational diffusion, rotational diffusion, and tumbling. It
then makes the step divide `duration` exactly. A numeric manual `dt` is
accepted only if it passes the same parameter-derived checks.

The swimmer is a point process and the disk is not a PDE mesh. Spatial
resolution enters in two places:

- the resolution-controlled disk scan used to bound attainable velocities and rates;
- the grid used only to render the chemical field.

For built-in Gaussian fields, the smallest relevant length is the minimum of
`R` and every Gaussian width. For a linear field it is `R`. A custom field
must supply `numerics.characteristic_length`. The scan and rendered grid both
use at least `numerics.scan_points_per_length` points across that length.
Unsafe manual grid choices fail before simulation.

## Changing the shape of c(r)

The `[field]` table selects one of four field types.

### Rotated elliptical Gaussian

```toml
[field]
kind = "gaussian"
baseline = 1.0
amplitude = 1.0
center = [0.030, 0.020]
sigma = [0.032, 0.044]
angle = 0.35
```

### Constant gradient

```toml
[field]
kind = "linear"
baseline = 2.0
gradient = [6.0, 0.0]
center = [0.0, 0.0]
```

### Several Gaussian sources

```toml
[field]
kind = "multi_gaussian"
baseline = 1.0

[[field.sources]]
amplitude = 0.8
center = [-0.030, 0.020]
sigma = [0.020, 0.036]
angle = 0.0

[[field.sources]]
amplitude = 1.2
center = [0.040, -0.020]
sigma = [0.028, 0.016]
angle = 0.7
```

### Completely custom field

Set `kind = "user"` and edit
[`user_field.py`](src/chem_swimmer/fields/user_field.py). The included
finite-difference gradient means that changing only `value()` is enough.
Replacing it with an analytic `gradient()` makes long simulations faster.

## Output

Unless `--output` is supplied, videos are written to:

- `output/videos/case_08.mp4` for one case;
- `output/videos/probability_density_cases_00_08.mp4` for the microscopic
  Cases 0-8 PDF;
- `output/videos/probability_density_difference_cases_00_08.mp4` for the
  microscopic signed differences `p_case - p_Case0` on one common scale;
- `output/videos/probability_density_macro_cases_00_08.mp4` and
  `output/videos/probability_density_difference_macro_cases_00_08.mp4` for
  the corresponding closed-equation results.
- `output/videos/probability_density_difference_detail_cases_00_08.mp4` and
  `output/videos/probability_density_difference_detail_macro_cases_00_08.mp4`
  for morphology-focused views with a separate range for each case.
- `output/figures/chemical_field.png` for the dimensionless field `c(x,y)`.
- `output/figures/density_inference_observables.png` plus summary and
  time-series CSV files for common-scale transport observables and relaxation
  times.
- `output/figures/microscopic_mechanism_observables.png` plus summary and
  time-series CSV files for signed response, angular moments, tumble counts,
  displacement moments, and paired Monte Carlo uncertainty.
- `output/data/response_scaling_audit.csv` for the optional sign-reversal,
  perturbative-order, core/wall, and wall-law audit.
- `output/data/numerical_convergence_audit.csv` for microscopic timestep and
  macroscopic equilibrium-grid refinement.

The numerical probability density is saved to
`output/data/pdf_cases_00_08.npz` for `--micro` and
`output/data/pdf_macro_cases_00_08.npz` for `--macro`. Each contains
dimensionless `time`,
`coordinates`, the absolute `density`, the smoother `difference_density`, both
bandwidth fields, `M`, `dt`, the method, seed, case numbers, and the
configuration source.
Both density arrays are defined on the unit disk and integrate to one for
every case and time.
The microscopic NPZ additionally stores aggregate angular, tumble, and
displacement observables with paired Case-minus-Case-0 standard errors.
Individual particle trajectories are never saved by `compare`.
Both NPZ files also embed a schema version, software/runtime versions, the
complete TOML input text, and its SHA-256 digest, so the archived result does
not depend on an absolute configuration path remaining valid.

For a collaborator-oriented derivation and a file-by-file interpretation of
all outputs, see
[`../MATHEMATICAL_AND_OUTPUT_GUIDE.md`](../MATHEMATICAL_AND_OUTPUT_GUIDE.md).
The independent equation audit and the focused investigation of Cases 2-5 are
recorded in
[`../VALIDATION_REPORT.md`](../VALIDATION_REPORT.md).

## Microscopic and macroscopic density strategies

With no strategy flag, both density strategies run:

```bash
python run.py compare
```

To run only one strategy:

```bash
python run.py compare --micro
python run.py compare --macro
```

`--micro` is the particle simulation followed by the paired KDE. It uses the
configured or automatically selected particle count `M` and preserves the full
stochastic dynamics of equation (6).

`--macro` directly solves the closed dimensionless equation (16) from
`Theory.pdf`,

```text
partial_t p = div(D_eff grad(p) - kappa p grad(c)),
J = -D_eff grad(p) + kappa p grad(c),       n dot J = 0,
D_eff = D_T0 + v0^2/(2 gamma0),
gamma0 = D_R0 + lambda0(1-a1).
```

It uses the case-specific mobility `kappa` from equation (18). Consequently,
Cases 2, 4, and 5 equal Case 0 at this order, exactly as Table 2 predicts.
There is no sampling count `M` in this method. The default uniform spatial
initial state is represented with exactly unit mass on the finite-volume disk.
Set `density.initial_position_distribution = "fixed"` to start both methods
from `initial.position`, while recognizing that a localized density violates
the closure ordering during its transient.

The macro solver uses a conservative cell-centered finite-volume method,
zeroes the flux through every disk-boundary face, and uses exponential-fitting
face fluxes. This preserves nonnegativity under the reported timestep and
exactly preserves the discrete equilibrium proportional to
`exp(kappa c / D_eff)`. Its grid is automatically enlarged, when necessary, to
resolve both the narrowest field length and the reported KDE bandwidth. The terminal
table reports the diffusion, chemotactic-drift, and combined finite-volume
timestep formulas before integration.

The closure requires `gamma0 > 0`, weak spatial responses, a persistence length
small compared with the field length, and times long compared with
`1/gamma0`. Inputs with `gamma0 = 0` remain valid for `--micro`, but are
rejected by `--macro` because the manuscript's orientational closure is then
undefined.

`density.ensemble_size = "auto"` chooses

```text
M = next_power_of_two(
    min_samples_per_kernel / (2 bandwidth_fraction^2)
)
```

which gives `M = 16384` for the defaults. This supplies about 82 effective
particles per Gaussian-kernel footprint even for a nearly uniform density.
A manual `--ensemble` below the quality requirement is rejected. To trade
spatial detail for speed, increase `density.bandwidth_fraction`; to demand
less statistical smoothness, reduce `density.min_samples_per_kernel`.

`density.difference_bandwidth_fraction = "auto"` uses up to twice the
absolute-PDF bandwidth while retaining at least three kernel widths across the
narrowest resolved field length. A numeric value makes this choice explicit.
For `default.toml`, `b=0.05`, `b_delta=0.15`, and `M=16384`, giving
about 82 and 737 uniform-disk samples per kernel footprint, respectively.

```bash
python run.py compare --micro
python run.py compare --micro --threads 8
```

Case 0 uses only `v0`, `D_T0`, `D_R0`, `lambda0`, and `a1`; every chemical
response coefficient is zero. It still evolves inside the disk with the
selected wall law. This differs from the older “free-space baseline,” which
was an analytic MSD reference with no wall.

The saved and displayed quantity is the dimensionless density `p(x,y,t)`
itself. The movie uses one fixed colorbar from 0 to 1 for every case and time;
it does not divide by a frame-wise or global maximum. Density values above 1
remain unchanged in the NPZ data and simply saturate in the display. The full
simulated interval is compressed into a movie no longer than
`density.max_video_seconds = 10`.

The companion difference movie makes small linear-response changes visible
without altering the absolute-density movie. Its blue-white-red panels show
`delta p_k=p_k-p_0`. The main micro and macro difference movies use one
synchronized symmetric physical-density range across every case, method, and
time, so color intensity compares magnitudes fairly. The separate `detail`
movies give each case its own synchronized range to expose morphology. Their
range is a display choice, not an effect-size estimator: a theory-zero case can
receive a large detail range when finite-M pixel noise is larger than a weak
coherent response. Use the main common-scale movies and the integrated
observables for magnitude comparisons. By default, ranges use the 99.5th
percentile of `|delta p|`; set `difference_case_percentile=100` to use exact
maxima.
The difference KDE is computed directly from the paired raw histograms using
`density.difference_bandwidth_fraction`; it is not a cosmetic video blur.
The macro solution is passed through the identical boundary-corrected KDE, so
both methods represent the same measured spatial resolution. When both methods
run, their common and detail ranges are computed jointly.

Pixelwise total variation and the detail ranges have a positive Monte Carlo
noise floor, so they must not be used alone to rank weak mechanisms. The
signed `mean_concentration` response and its paired standard error in
`output/data/microscopic_mechanism_summary.csv` provide the statistical audit.
Cases 2, 4, and 5 are zero only at the order retained by the closed equation;
any microscopic residual must be tested for convergence and significance.
For the default sliding wall, the symmetric high-\(M\) audit shows that Case
5 has a resolved linear boundary-layer response even though its retained bulk
mobility is zero. Cases 2 and 4 are substantially less stable and must not be
ranked from a case-specific color scale or a nonnegative pixel norm.

Run the perturbative order and wall-law audit with:

```bash
python scripts/audit_response_scaling.py --ensemble 16384
```

At the level of equation (16), the eight sensitivities are not individually
identifiable. Density evolution contains only `D_eff` and

```text
kappa = v_S/2 - D_T_c
        - v0 [v_c + D_R_S + (1-a1) lambda_S] / (2 gamma0).
```

Thus Cases 0, 2, 4, and 5 are identical in the closed theory; Cases 1 and 3
are nearly identical for the default coefficients; and so are Cases 6 and 8.
The inference report writes `kappa`, `kappa/D_eff`, field-exposure and centroid
shifts, total variation, and the 50%, 90%, and fitted relaxation times. Any
repeatable micro-macro departure within a macroscopically degenerate group is
the useful signature of microscopic dynamics; isolated pixel extrema are not.

## M1 Pro performance

The default `[performance]` section is tuned for this machine:

```toml
[performance]
engine = "auto"
threads = 0
msd_batch_size = 8
progress = true
```

`engine = "auto"` uses Numba for Gaussian, linear, and multi-Gaussian fields,
and automatically falls back to the readable Python integrator for
`user_field.py`. Setting `engine = "numba"` makes a fallback an error;
`engine = "python"` is useful for debugging and verification. `threads = 0`
uses all detected cores.

Measured in the supplied Apple-Silicon environment after JIT warm-up:

| Workload | Python reference | Numba | Speed-up |
|---|---:|---:|---:|
| Eight 30-second trajectories | 2.268 s | 0.0079 s | 288x |
| Eight-case MSD, 8 replicas, 3 seconds | 1.790 s | 0.0026 s | about 680x |

The first Numba call also compiles and caches the kernel, taking roughly one
second here. Video rendering then becomes the dominant cost and has its own
frame-by-frame ETA. See [PERFORMANCE_M1.md](docs/PERFORMANCE_M1.md) for the
profiling evidence, design decisions, and primary technical sources.

## Project tree

```text
single_particle_simulation/
├── configs/default.toml       parameters and field shape
├── docs/
│   ├── MATHEMATICS.md         equation-to-code map
│   └── NUMERICS.md            discretization and accuracy
├── src/chem_swimmer/
│   ├── fields/                c(r) and grad(c)
│   ├── visualization/         live display and MP4 rendering
│   ├── simulator.py           equation (6) time stepping
│   ├── statistics.py          paired ensemble MSD and CSV
│   ├── density.py             Cases 0-8 spatial PDF estimator
│   ├── accelerated.py         Numba/LLVM Arm64 kernels
│   ├── numerics.py            R scaling and automatic resolution
│   ├── progress.py            terminal rate and ETA
│   ├── kinetics.py            equation (2), gamma0, D_eff, kappa
│   ├── boundary.py            disk wall laws
│   ├── angles.py              S1 and tumble kernel
│   ├── diagnostics.py         parameter-range checks
│   └── cli.py                 four user commands
├── tests/                     deterministic mathematical checks
├── pyproject.toml
└── run.py
```

See [MATHEMATICS.md](docs/MATHEMATICS.md) before changing the dynamics and
[NUMERICS.md](docs/NUMERICS.md) before increasing the time step.
