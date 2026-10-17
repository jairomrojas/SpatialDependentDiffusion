"""Figure 5: what the population knows.

Three panels for the finite-lag navigation channel of the manuscript's
"What the population knows" section.  (a) Each rule's projected velocity,
divided by its own pulse drift, collapses onto one of three universal lag
shapes and saturates at one.  (b) The drift and the second angular harmonic
return two independent combinations of the steering and turning sensitivities,
separating Cases 6, 8 and 9, which the population fields merge into a single
mobility.  (c) The information the realized displacement carries about the
starting state: the two translational-diffusivity rules dominate it although
they never navigate, and the rules that navigate best sit at the baseline.

All three paths run on a constant nutrient gradient, where the observable is
unambiguous: particles (windows of the random walk), the kinetic hierarchy
(angular moments of the backward generator, carried past polar order), and the
closed first-order prediction.

    python scripts/generate_figure_6_information_observables.py [M]
    python scripts/generate_figure_6_information_observables.py --reuse
"""

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

import numpy as np

from model.solvers.chem_swimmer import information as nav
from model.solvers.chem_swimmer.config import load_config
from model.solvers.chem_swimmer.density import PDF_CASES
from model.solvers.chem_swimmer.fields import create_field
from model.solvers.chem_swimmer.visualization.information_plot import (
    render_information_report,
)

CONFIG = ROOT / "model" / "parameters" / "navigation.toml"
ARCHIVE = (
    ROOT
    / "outputs"
    / "simulations"
    / "population_release"
    / "data"
    / "navigation-observables.npz"
)
METRICS = (
    ROOT
    / "outputs"
    / "simulations"
    / "population_release"
    / "metrics"
    / "navigation-observables.csv"
)
FIGURE = ROOT / "manuscript" / "figures" / "figure-6-information-observables.pdf"
SUPPLEMENT = (
    ROOT
    / "outputs"
    / "simulations"
    / "population_release"
    / "figures"
    / "supplementary"
    / "figure-s18-information-observables.pdf"
)

FIELDS = (
    "lags",
    "projected_velocity",
    "alignment_harmonic",
    "level_harmonic",
    "information",
    "projected_velocity_error",
    "alignment_harmonic_error",
    "level_harmonic_error",
)


def _store(result: nav.NavigationObservables) -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARCHIVE, **{name: getattr(result, name) for name in FIELDS})


def _load() -> nav.NavigationObservables:
    with np.load(ARCHIVE, allow_pickle=False) as saved:
        values = {name: saved[name] for name in FIELDS}
    return nav.NavigationObservables(
        method="particles", case_numbers=PDF_CASES, **values
    )


def main() -> int:
    arguments = sys.argv[1:]
    reuse = "--reuse" in arguments
    sizes = [int(value) for value in arguments if not value.startswith("--")]

    config = load_config(CONFIG)
    field = create_field(config.field)
    gradient = float(np.linalg.norm(nav.uniform_gradient(config, field)))
    gamma1, gamma2, gamma3 = nav._gamma_rates(config)

    if reuse and ARCHIVE.exists():
        particles = _load()
        print(f"reusing {ARCHIVE.name}")
    else:
        start = time.time()
        particles = nav.particle_navigation(
            config,
            field=field,
            ensemble_size=sizes[0] if sizes else None,
            show_progress=True,
        )
        print(f"random walk {time.time() - start:.0f}s")
        _store(particles)

    lags = particles.lags
    start = time.time()
    kinetic = nav.harmonic_navigation(config, lags, field=field)
    print(f"kinetic hierarchy {time.time() - start:.0f}s")
    prediction = nav.navigation_prediction(config, lags, field=field)

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    with METRICS.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "pulse_drift",
                "drift_particles",
                "drift_kinetic",
                "drift_prediction",
                "drift_error",
                "alignment_particles",
                "alignment_kinetic",
                "alignment_prediction",
                "level_particles",
                "level_kinetic",
                "level_prediction",
                "information_particles",
                "information_kinetic",
                "information_prediction",
            ]
        )
        last = -1
        finite = np.isfinite(particles.information[0]).nonzero()[0][-1]
        for case_number in PDF_CASES:
            row = [case_number, f"{nav.pulse_drift(config, case_number):.6g}"]
            for name, index in (
                ("projected_velocity", last),
                ("alignment_harmonic", last),
                ("level_harmonic", last),
                ("information", finite),
            ):
                for result in (particles, kinetic, prediction):
                    row.append(f"{result.response(name)[case_number][index]:.6g}")
                if name == "projected_velocity":
                    row.append(
                        f"{particles.projected_velocity_error[case_number][last]:.6g}"
                    )
            writer.writerow(row)

    print(f"\ngradient |grad eta| = {gradient:.4g}")
    print(f"gamma1 = {gamma1:.6g}, gamma2 = {gamma2:.6g}, gamma3 = {gamma3:.6g}")
    print(f"lag range gamma1 tau = [{gamma1 * lags[1]:.3g}, {gamma1 * lags[-1]:.3g}]")
    print(
        f"\n{'case':>5} {'V_j':>11} {'A/V g particles':>17} {'kinetic':>10} "
        f"{'prediction':>12}"
    )
    for case_number in PDF_CASES:
        drift = nav.pulse_drift(config, case_number) * gradient
        if abs(drift) < 1.0e-14:
            continue
        values = [
            result.response("projected_velocity")[case_number][-1] / drift
            for result in (particles, kinetic, prediction)
        ]
        print(
            f"{case_number:>5} {drift:>11.4g} {values[0]:>17.4f} "
            f"{values[1]:>10.4f} {values[2]:>12.4f}"
        )

    for path in (FIGURE, SUPPLEMENT):
        render_information_report(
            config,
            gradient,
            particles=particles,
            kinetic=kinetic,
            prediction=prediction,
            figure_path=path,
            also_png=False,
        )
        print(f"wrote {path}")
    print(f"wrote {METRICS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
