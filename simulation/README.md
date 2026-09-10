# Spatially Dependent Diffusion of an Active Particle

[![tests](https://github.com/jairomrojas/SpatialDependentDiffusion/actions/workflows/tests.yml/badge.svg)](https://github.com/jairomrojas/SpatialDependentDiffusion/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Schema 2](https://img.shields.io/badge/schema-2-green.svg)](#)

This repository contains a reproducible microscopic–macroscopic numerical framework for a chemically responsive active Brownian / run-and-tumble particle in a two-dimensional disk. The mathematical derivation is provided in [`docs/theory.pdf`](docs/theory.pdf), while the installable Python package (`chem_swimmer`) implements stochastic particle dynamics, closed drift–diffusion approximations, diagnostics, validation audits, and publication-ready outputs.

---

## 📚 Documentation Map

Start with these key resources in the [`docs/`](docs/) directory:

1. **[`docs/mathematical_guide.md`](docs/mathematical_guide.md)** — Didactic guide to equations, approximations, source files, and generated outputs.
2. **[`docs/validation_report.md`](docs/validation_report.md)** — Independent equation audit, focused analysis of Cases 2–5, numerical refinement evidence, and boundary layer physics.
3. **[`docs/theory.pdf`](docs/theory.pdf)** — Mathematical manuscript detailing derivations.
4. **[`docs/mathematics.md`](docs/mathematics.md)** — In-depth breakdown of continuous equations and moment closures.
5. **[`docs/numerics.md`](docs/numerics.md)** — Numerical scheme design, Scharfetter–Gummel discretization, and stability limits.
6. **[`docs/performance.md`](docs/performance.md)** — Benchmarks and optimization guide (Numba LLVM fused kernels & SIMD parallelism).

---

## 🔬 Main Scientific Conclusions

The linear macroscopic chemotactic mobility derived from angular moment closure is:

$$\kappa = \frac{v^S}{2} - D_T^c - \frac{v_0}{2\gamma_0} \left[ v^c + D_R^S + (1 - a_1)\lambda^S \right], \qquad \gamma_0 = D_{R0} + \lambda_0(1 - a_1)$$

- **Cases 2 ($D_R^c$), 4 ($\lambda^c$), and 5 ($D_T^S$)** have **zero bulk mobility at the retained first-order closure**, whereas Case 3 ($v^c$) has non-zero mobility.
- **Cases 2 & 4** are dominated by unresolved Monte Carlo sampling noise under common signed observables ($\Delta\langle c\rangle$).
- **Case 5** exhibits a reproducible linear full-disk response driven by an **angular boundary layer at the sliding wall** ($\mathbf{p}_{\text{wall}}^{(0)} \neq 0$), which is absent from the scalar bulk closure.

---

## 🚀 Quick Start

### 1. Installation

Requires Python 3.11 or newer:

```bash
# Clone the repository
git clone https://github.com/jairomrojas/SpatialDependentDiffusion.git
cd SpatialDependentDiffusion

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with development dependencies
python -m pip install -e ".[dev]"
```

### 2. Execution Commands

You can run commands either via the installed CLI (`chem-swimmer`) or directly via `run.py`:

```bash
# Check kinetic modulation physical bounds for all cases
python run.py check --case all

# Perform a full micro vs. macro comparison (Cases 0–8)
python run.py compare

# Run live trajectory preview or render single-case MP4 video
python run.py live --case 8
python run.py video --case 8

# Run test suite
PYTHONPATH=src python3 -m unittest discover -s tests
```

> **Note**: FFmpeg (`brew install ffmpeg`) is required only when rendering MP4 video output.

---

## ⚙️ Configuration & Output Structure

Routine model choices are centralized in [`configs/default.toml`](configs/default.toml):
- Physical parameters ($v_0, D_{T0}, D_{R0}, \lambda_0, a_1$) and chemical field shapes.
- Spatial and temporal resolution criteria ($dt$, grid bounds, target RMS steps).
- Sampling configuration (ensemble size $M$, KDE bandwidths, random seeds).

Generated artifacts are written to `output/`:
- `output/figures/`: Publication figures (`chemical_field.png`, `density_inference_observables.png`, etc.).
- `output/videos/`: Raster MP4 videos for absolute/difference probability densities.
- `output/data/`: Compressed `.npz` archives and detailed `.csv` statistical audit logs.

---

## 📁 Repository Layout

```text
.
├── README.md                    # Master documentation & entry point
├── CONTRIBUTING.md              # Contributor workflow & standards
├── pyproject.toml               # Python package configuration (chem-swimmer 0.2.0)
├── run.py                       # Top-level executable CLI script
├── configs/                     # Central model and numerical configuration
│   └── default.toml
├── docs/                        # Centralized documentation hub
│   ├── mathematical_guide.md
│   ├── validation_report.md
│   ├── mathematics.md
│   ├── numerics.md
│   ├── performance.md
│   └── theory.pdf
├── scripts/                     # Reproducible focused audit scripts
│   ├── audit_numerical_convergence.py
│   ├── audit_response_scaling.py
│   └── diagnose_case_magnitudes.py
├── src/                         # Core Python package source
│   └── chem_swimmer/
│       ├── accelerated.py       # Numba LLVM fused kernel engine
│       ├── boundary.py          # Disk collision laws (sliding, specular, diffuse)
│       ├── macroscopic.py       # Scharfetter–Gummel finite-volume solver
│       ├── density.py           # Kernel density estimation & histograms
│       └── ...
├── tests/                       # Mathematical & regression test suite
│   └── test_mathematics.py
└── output/                      # Generated data, figures, and videos (ignored)
```

---

## 📜 Versioning & Reproducibility

- **Code Version**: `0.2.0`
- **Output Schema**: `2`
- Every generated density archive embeds the exact TOML input configuration, its SHA-256 digest, and runtime environment specifications (Python, NumPy, Numba versions).
