"""Figure 3's comparison, repeated over independent seeds.

Figure 3 reports one seed at t*Gamma_R = 1.78.  Repeating it turns the paired
sampling error into a measured seed-to-seed scatter, which is what separates
"the closure is wrong for this rule" from "the ensemble is not resolving it".

That distinction matters most for Cases 2 and 4, whose density response is
O(t^3): their signal grows as t^3 while the paired Monte-Carlo noise grows only
as t^(3/2), so the reporting time was chosen to put them above the floor,

    SNR ~ sqrt(M) t^(3/2)

and the CSV records where each case ended up.  The well-resolved cases, whose
statistical error is far below their systematic one, then expose the polar
closure's own error -- the neglected second harmonic, of predicted size
(v k / D_R2)^2.

Reported slices are kept at three: the comparison reads only the final one,
and the accumulation cost is proportional to slices.

    python scripts/check_long_time_validation.py [duration] [M] [seeds]
"""

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from chem_swimmer.config import load_config
from chem_swimmer.density import PDF_CASES, compute_position_pdf
from chem_swimmer.fields import create_field
from chem_swimmer.macroscopic import compute_macroscopic_pdf
from chem_swimmer.polar_fourier import compute_polar_fourier_pdf

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0e-3
ENSEMBLE = int(sys.argv[2]) if len(sys.argv) > 2 else 1_048_576
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
OUTPUT = ROOT / "results" / "metrics" / "long-time-validation.csv"


def response(result, case):
    """The plotted response field: case minus baseline, at the final slice."""
    index = {int(c): j for j, c in enumerate(result.case_numbers)}
    return (
        result.difference_density[index[case], -1]
        - result.difference_density[index[0], -1]
    )


def main() -> None:
    base = load_config(ROOT / "config" / "default.toml")
    config = replace(
        base,
        simulation=replace(base.simulation, duration=DURATION),
        density=replace(
            base.density,
            ensemble_size=ENSEMBLE,
            time_slices=3,
            min_samples_per_kernel=5,
        ),
    )
    field = create_field(config.field)
    gamma1 = config.kinetics.D_R0 + config.kinetics.lambda0 * (1.0 - 0.30)
    gamma2 = 4.0 * config.kinetics.D_R0 + config.kinetics.lambda0 * (1.0 - 0.30**4)
    print(
        f"t = {DURATION:.3e}   t*gamma1 = {DURATION * gamma1:.2f}   "
        f"M = {ENSEMBLE:,}   seeds = {SEEDS}",
        flush=True,
    )

    start = time.time()
    fourier, diagnostics = compute_polar_fourier_pdf(config, field)
    print(
        f"fourier {time.time() - start:5.0f}s  "
        f"wall-free ratio {diagnostics.wall_free_ratio:.3f}",
        flush=True,
    )
    start = time.time()
    macro = compute_macroscopic_pdf(config, field=field)
    print(f"macro   {time.time() - start:5.0f}s", flush=True)

    coordinates = fourier.coordinates
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    spacing = coordinates[1] - coordinates[0]
    mask = grid_x * grid_x + grid_y * grid_y <= 1.0

    def norm(values):
        return float(np.sqrt((np.where(mask, values, 0.0) ** 2).sum()) * spacing)

    theory_fourier = {c: response(fourier, c) for c in PDF_CASES[1:]}
    theory_macro = {c: response(macro, c) for c in PDF_CASES[1:]}

    samples = {c: [] for c in PDF_CASES[1:]}
    for seed_index in range(SEEDS):
        seeded = replace(
            config,
            simulation=replace(
                config.simulation, seed=config.simulation.seed + 1000 * seed_index
            ),
        )
        start = time.time()
        micro = compute_position_pdf(seeded, ensemble_size=ENSEMBLE)
        print(f"  micro seed {seed_index}: {time.time() - start:5.0f}s", flush=True)
        for case in PDF_CASES[1:]:
            samples[case].append(response(micro, case))

    # The closure error the retained harmonics cannot see, for reference.  On a
    # single-mode landscape the wavenumber under test is the landscape's own;
    # otherwise the finest scale the reported field resolves is the best proxy.
    speed = config.kinetics.v0
    wavevector = getattr(field, "wavevector", None)
    wavenumber = (
        float(np.hypot(*wavevector))
        if wavevector is not None
        else 1.0 / config.density.difference_bandwidth_fraction
    )
    predicted_closure = (speed * wavenumber / gamma2) ** 2

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "theory_l2",
                "bias_l2",
                "seed_scatter_l2",
                "bias_over_standard_error",
                "signal_to_noise_fourier",
                "signal_to_noise_macro",
                "relative_bias",
                "statistical_error",
                "fourier_minus_macro_relative",
            ]
        )
        print(
            f"\n{'case':>4}{'SNR':>8}{'rel.bias':>10}{'stat.err':>10}"
            f"{'bias/SE':>9}{'F-vs-macro':>12}"
        )
        for case in PDF_CASES[1:]:
            stack = np.array(samples[case])
            mean = stack.mean(axis=0)
            signal = norm(theory_fourier[case])
            bias = norm(mean - theory_fourier[case])
            scatter = float(np.mean([norm(one - mean) for one in stack]))
            signal_macro = norm(theory_macro[case])
            solver_gap = norm(theory_fourier[case] - theory_macro[case]) / max(
                signal, 1e-300
            )
            z = bias / (scatter / np.sqrt(SEEDS))
            writer.writerow(
                [
                    case,
                    f"{signal:.12g}",
                    f"{bias:.12g}",
                    f"{scatter:.12g}",
                    f"{z:.12g}",
                    f"{signal / scatter:.12g}",
                    f"{signal_macro / scatter:.12g}",
                    f"{bias / signal:.12g}",
                    f"{scatter / signal:.12g}",
                    f"{solver_gap:.12g}",
                ]
            )
            print(
                f"{case:>4}{signal / scatter:>8.1f}{bias / signal * 100:>9.1f}%"
                f"{scatter / signal * 100:>9.1f}%{z:>9.2f}{solver_gap:>12.4f}"
            )
    print(
        f"\npredicted nematic closure error (v k / gamma2)^2 = {predicted_closure:.3f}"
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
