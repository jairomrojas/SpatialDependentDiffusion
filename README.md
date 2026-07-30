# Spatially dependent diffusion of an active particle

This repository contains a reproducible microscopic–macroscopic comparison
for a chemically responsive active particle in a two-dimensional disk.
`Theory.pdf` derives the model, while `single_particle_simulation` implements
the stochastic particle dynamics, the closed drift–diffusion approximation,
diagnostics, validation studies, and publication-ready outputs.

Start with these documents:

1. [`MATHEMATICAL_AND_OUTPUT_GUIDE.md`](MATHEMATICAL_AND_OUTPUT_GUIDE.md) is
   the didactic guide to the equations, approximations, source files, and
   every generated output.
2. [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) records the independent
   equation audit, the focused investigation of Cases 2–5, numerical
   refinement evidence, corrections made during review, and remaining model
   limitations.
3. [`Theory.pdf`](Theory.pdf) is the mathematical manuscript.
4. [`single_particle_simulation/README.md`](single_particle_simulation/README.md)
   is the practical installation and command reference.

## Main scientific conclusion

The linear macroscopic mobility is

\[
\kappa=\frac{v^S}{2}-D_T^c-
\frac{v_0}{2\gamma_0}
\left[v^c+D_R^S+(1-a_1)\lambda^S\right],
\qquad
\gamma_0=D_{R0}+\lambda_0(1-a_1).
\]

Cases 2, 4, and 5 therefore have zero *bulk mobility at the retained
first-order closure*, whereas Case 3 does not. This does not require their
finite-sample microscopic density movies to be identically blank. Cases 2
and 4 are dominated by unresolved Monte Carlo texture in the present common
signed observable. Case 5 has a reproducible linear full-disk response caused
by the sliding-wall angular boundary layer, which is absent from the scalar
bulk closure. The detailed evidence is in the validation report.

## Reproduce the default analysis

Use Python 3.11 or newer and FFmpeg if MP4 output is required:

```bash
cd single_particle_simulation
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python run.py check --case all
python run.py compare
python -m unittest discover -s tests -q
```

Routine model and numerical choices are centralized in
[`single_particle_simulation/configs/default.toml`](single_particle_simulation/configs/default.toml).
Generated figures, videos, data archives, and audit tables are under
`single_particle_simulation/output/`.

## Repository map

```text
.
├── README.md
├── Theory.pdf
├── MATHEMATICAL_AND_OUTPUT_GUIDE.md
├── VALIDATION_REPORT.md
└── single_particle_simulation/
    ├── configs/                 model and numerical configuration
    ├── src/chem_swimmer/        installable simulation package
    ├── scripts/                 reproducible focused audits
    ├── tests/                   mathematical and regression tests
    ├── output/                  generated data and visualizations
    ├── README.md                command and implementation reference
    └── CONTRIBUTING.md          collaborator workflow
```

The code and output schema are versioned as `0.2.0` and `2`, respectively.
Each density archive embeds the complete configuration, its SHA-256 digest,
and Python/NumPy/Numba runtime versions. Before public redistribution, the
repository owner should add the intended license and citation metadata.
