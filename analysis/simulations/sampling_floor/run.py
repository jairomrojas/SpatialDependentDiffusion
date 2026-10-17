"""Separate Monte-Carlo noise from systematic closure error in the micro test.

This is the audit behind the per-case reading of Fig. 3 and the appendix
sheets: it answers "is the polar closure wrong for this rule, or is the
particle estimate simply not resolving it?"

Runs the microscopic ensemble with several independent seeds at fixed M and
decomposes the micro-minus-theory discrepancy of the response field into
  bias    = || mean_over_seeds(micro) - theory ||     (systematic)
  scatter = mean_over_seeds || micro - mean ||        (Monte-Carlo)
If the disagreement is pure sampling noise, bias ~ scatter/sqrt(n_seeds).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataclasses import replace

import numpy as np

import chem_swimmer.density as D
import chem_swimmer.polar_fourier as PF
from chem_swimmer.config import load_config
from chem_swimmer.density import PDF_CASES
from chem_swimmer.fields import create_field


def response_density(res, case):
    """The plotted response field: difference_density[case] minus baseline."""
    idx = {int(c): j for j, c in enumerate(res.case_numbers)}
    return res.difference_density[idx[case], -1].astype(float) - res.difference_density[
        idx[0], -1
    ].astype(float)


def disk_norm_factory(coordinates):
    x = np.asarray(coordinates)
    dx = x[1] - x[0]
    grid_x, grid_y = np.meshgrid(x, x)
    mask = grid_x * grid_x + grid_y * grid_y <= 1.0

    def nrm(a):
        return float(np.sqrt((np.where(mask, a, 0.0) ** 2).sum()) * dx)

    return nrm, grid_x, grid_y, dx, mask


M = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 6

base = load_config(ROOT / "config" / "default.toml")
field = create_field(base.field)
t0 = time.time()
ref, _ = PF.compute_polar_fourier_pdf(base, field)
print(f"fourier reference {time.time() - t0:.0f}s", flush=True)
nrm, *_ = disk_norm_factory(ref.coordinates)
theory = {c: response_density(ref, c) for c in PDF_CASES[1:]}

samples = {c: [] for c in PDF_CASES[1:]}
for s in range(NSEED):
    cfg = replace(
        base,
        simulation=replace(base.simulation, seed=base.simulation.seed + 1000 * s),
        density=replace(base.density, ensemble_size=M, min_samples_per_kernel=5),
    )
    t0 = time.time()
    mic = D.compute_position_pdf(cfg, ensemble_size=M)
    print(f"  seed {s}: {time.time() - t0:.0f}s", flush=True)
    for c in PDF_CASES[1:]:
        samples[c].append(response_density(mic, c))

print(f"\nM={M}, {NSEED} seeds")
print(
    f"{'case':>4} {'||theory||':>11} {'bias':>11} {'scatter':>11} "
    f"{'bias/(sc/sqrt n)':>17} {'SNR=th/sc':>10}"
)
for c in PDF_CASES[1:]:
    arr = np.array(samples[c])
    mean = arr.mean(axis=0)
    S = nrm(theory[c])
    bias = nrm(mean - theory[c])
    scatter = float(np.mean([nrm(a - mean) for a in arr]))
    print(
        f"{c:>4} {S:>11.3e} {bias:>11.3e} {scatter:>11.3e} "
        f"{bias / (scatter / np.sqrt(NSEED)):>17.2f} {S / scatter:>10.2f}"
    )
