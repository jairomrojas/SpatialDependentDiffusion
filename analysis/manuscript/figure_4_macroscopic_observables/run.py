"""Figure 3: macroscopic observables along all three paths.

Runs the same release three ways -- the microscopic random walk, the
conservative polar PDE, and the closed-form moment prediction of
``chem_swimmer.observables`` -- and reduces each to the net polarity, the
centroid, and the second moment.  Writes the manuscript figure, a compact CSV
of the final-slice comparison, and a small reusable archive of the particle
observables so the figure can be restyled without repeating the ensemble.

    python scripts/generate_figure_4_macroscopic_observables.py [M] [duration]
    python scripts/generate_figure_4_macroscopic_observables.py --reuse
"""

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from chem_swimmer import observables as ob
from chem_swimmer.config import load_config
from chem_swimmer.density import PDF_CASES, compute_position_pdf
from chem_swimmer.fields import create_field
from chem_swimmer.visualization.observable_plot import (
    render_macroscopic_observable_report,
)

ARCHIVE = ROOT / "results" / "data" / "macroscopic-observables-particles.npz"
METRICS = ROOT / "results" / "metrics" / "macroscopic-observables.csv"
FIGURE = ROOT.parent / "manuscript" / "figures" / "figure-4-macroscopic-observables.pdf"
SUPPLEMENT = (
    ROOT
    / "results"
    / "figures"
    / "supplementary"
    / "figure-s16-macroscopic-observables.pdf"
)


def _arguments() -> tuple[bool, int, float]:
    arguments = [value for value in sys.argv[1:] if value != "--reuse"]
    reuse = "--reuse" in sys.argv[1:]
    ensemble = int(arguments[0]) if arguments else 1_048_576
    duration = float(arguments[1]) if len(arguments) > 1 else 0.0
    return reuse, ensemble, duration


def _store(series: ob.TransportObservables, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=series.times,
        case_numbers=np.asarray(series.case_numbers, dtype=np.int64),
        net_polarity=series.net_polarity,
        centroid=series.centroid,
        spread=series.spread,
        shape=series.shape,
        mean_squared_displacement=series.mean_squared_displacement,
        net_polarity_error=series.net_polarity_error,
        centroid_error=series.centroid_error,
        spread_error=series.spread_error,
        shape_error=series.shape_error,
    )


def _load(path: Path) -> ob.TransportObservables:
    with np.load(path, allow_pickle=False) as saved:
        return ob.TransportObservables(
            method="particles",
            times=saved["times"],
            case_numbers=tuple(int(case) for case in saved["case_numbers"]),
            net_polarity=saved["net_polarity"],
            centroid=saved["centroid"],
            spread=saved["spread"],
            shape=saved["shape"],
            mean_squared_displacement=saved["mean_squared_displacement"],
            net_polarity_error=saved["net_polarity_error"],
            centroid_error=saved["centroid_error"],
            spread_error=saved["spread_error"],
            shape_error=saved["shape_error"],
        )


def _write_metrics(series: dict[str, ob.TransportObservables], gamma1: float) -> None:
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    particles = series["particles"]
    errors = {
        "net_polarity": particles.net_polarity_error,
        "centroid": particles.centroid_error,
        "spread": particles.spread_error,
        "shape": particles.shape_error,
    }
    with METRICS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "case",
                "observable",
                "component",
                "t_times_gamma1",
                "prediction",
                "polar_pde",
                "particles",
                "particle_standard_error",
                "particle_minus_prediction_z",
            )
        )
        for case_index, case_number in enumerate(PDF_CASES):
            for name in ("net_polarity", "centroid", "spread", "shape"):
                response = {
                    method: data.response(name)[case_index, -1]
                    for method, data in series.items()
                }
                error = errors[name][case_index, -1]
                components = ("x", "y") if np.ndim(response["particles"]) else ("",)
                for axis, component in enumerate(components):

                    def pick(values, axis=axis, component=component):
                        return float(values[axis] if component else values)

                    predicted = pick(response["prediction"])
                    measured = pick(response["particles"])
                    deviation = pick(error)
                    writer.writerow(
                        (
                            case_number,
                            name,
                            component,
                            f"{particles.times[-1] * gamma1:.6g}",
                            f"{predicted:.8g}",
                            f"{pick(response['polar PDE']):.8g}",
                            f"{measured:.8g}",
                            f"{deviation:.8g}",
                            f"{(measured - predicted) / deviation:.4g}"
                            if deviation > 0.0
                            else "",
                        )
                    )


def main() -> None:
    reuse, ensemble, duration = _arguments()
    base = load_config(ROOT / "config" / "default.toml")
    config = replace(
        base,
        density=replace(base.density, ensemble_size=ensemble, min_samples_per_kernel=5),
    )
    if duration > 0.0:
        config = replace(
            config, simulation=replace(config.simulation, duration=duration)
        )
    field = create_field(config.field)
    gamma1 = ob.polarity_relaxation_rate(config)
    print(
        f"M = {ensemble:,}   t_max = {config.simulation.duration:.4g}   "
        f"t*gamma1 = {config.simulation.duration * gamma1:.3f}",
        flush=True,
    )

    if reuse and ARCHIVE.exists():
        particles = _load(ARCHIVE)
        print(f"reusing {ARCHIVE.name}", flush=True)
    else:
        start = time.time()
        result = compute_position_pdf(
            config, ensemble_size=ensemble, show_progress=True
        )
        particles = ob.particle_observables(result, config.initial.position)
        _store(particles, ARCHIVE)
        print(f"random walk {time.time() - start:.0f}s", flush=True)

    start = time.time()
    pde = ob.finite_volume_observables(config, field)
    print(f"polar PDE {time.time() - start:.0f}s", flush=True)
    start = time.time()
    prediction = ob.predicted_observables(config, field, particles.times)
    print(f"prediction {time.time() - start:.0f}s", flush=True)

    pde = pde.interpolated(particles.times)

    series = {"particles": particles, "polar PDE": pde, "prediction": prediction}
    _write_metrics(series, gamma1)
    render_macroscopic_observable_report(series, config, FIGURE, also_png=False)
    render_macroscopic_observable_report(series, config, SUPPLEMENT)
    print(f"wrote {METRICS}\nwrote {FIGURE}\nwrote {SUPPLEMENT}")


if __name__ == "__main__":
    main()
