"""Figure 4: what the population gains.

Three panels.  (a) The uptake of every responding rule, divided by its own
predicted gain ``kappa A^2 / 2 D_eff``, collapses onto the single curve
``1 - exp(-t/tau_q)`` along all three paths.  (b) On one landscape mode the
uptake saturates on the landscape time while the spread is still growing toward
the domain time, two decades later.  (c) The divergence between a rule's
population and the null forager's grows as the square of the density response,
which fixes how many cells a detection needs.

    python scripts/generate_figure_5_foraging_observables.py [M]
    python scripts/generate_figure_5_foraging_observables.py --reuse
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
from chem_swimmer.moment import build_moment_plan
from chem_swimmer.visualization.foraging_plot import render_foraging_report

ARCHIVE = ROOT / "results" / "data" / "foraging-observables.npz"
METRICS = ROOT / "results" / "metrics" / "foraging-observables.csv"
FIGURE = ROOT.parent / "manuscript" / "figures" / "figure-5-foraging-observables.pdf"
SUPPLEMENT = (
    ROOT
    / "results"
    / "figures"
    / "supplementary"
    / "figure-s17-foraging-observables.pdf"
)
# Panel (b) runs past the disk mixing time; panel (c) uses the paper's own
# landscape, where the nine-rule hierarchy lives.
TWO_CLOCK_MIXING_TIMES = 1.3
DISCRIMINATION_DURATION = 4.0e-4
DISCRIMINATION_STEP = 5.0e-7


def _require_clean_flux(config, field, label: str) -> None:
    """Refuse a grid on which the solver would add numerical diffusivity.

    An unresolved run makes the cell Peclet number exceed two, and the
    dissipation the flux then adds carries the speed responses -- it survives
    the case-minus-baseline subtraction and imitates exactly the uptake this
    figure measures.
    """

    plan = build_moment_plan(config, field)[0]
    if plan.artificial_diffusivity > 0.0:
        raise RuntimeError(
            f"{label}: cell Peclet {plan.cell_peclet:.2f} would add numerical "
            f"diffusivity {plan.artificial_diffusivity:.3g}; refine the grid"
        )
    return plan


def _uptake_series(config, field, times=None):
    solver = ob.finite_volume_observables(config, field)
    grid = times if times is not None else solver.times
    solver = solver.interpolated(grid)
    closed = ob.periodic_uptake_prediction(config, field, grid)
    return solver, closed


def main() -> None:
    reuse = "--reuse" in sys.argv[1:]
    arguments = [value for value in sys.argv[1:] if value != "--reuse"]
    base = load_config(ROOT / "config" / "foraging.toml")
    field = create_field(base.field)
    ensemble = int(arguments[0]) if arguments else base.density.ensemble_size
    wavenumber = field.wavenumber
    harvest = ob.landscape_uptake_time(base, wavenumber)
    mixing = ob.domain_mixing_time(base)
    print(
        f"q = {wavenumber:.2f}   q v0/gamma1 = "
        f"{wavenumber * base.kinetics.v0 / ob.polarity_relaxation_rate(base):.3f}   "
        f"tau_landscape = {harvest:.3e}   tau_domain = {mixing:.3e}   "
        f"ratio = {mixing / harvest:.0f}",
        flush=True,
    )
    _require_clean_flux(base, field, "three-path comparison")

    # --- panel (a): three paths on one landscape mode ---
    if reuse and ARCHIVE.exists():
        with np.load(ARCHIVE, allow_pickle=False) as saved:
            times = saved["times"]
            particle_uptake = saved["particle_uptake"]
            particle_error = saved["particle_error"]
        print(f"reusing {ARCHIVE.name}", flush=True)
    else:
        start = time.time()
        result = compute_position_pdf(base, ensemble_size=ensemble, show_progress=True)
        particles = ob.particle_observables(result, base.initial.position)
        times = particles.times
        particle_uptake = particles.uptake
        index = result.microscopic_observables.names.index("mean_eta")
        particle_error = result.microscopic_observables.paired_standard_errors[index]
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ARCHIVE,
            times=times,
            particle_uptake=particle_uptake,
            particle_error=particle_error,
            ensemble_size=np.asarray(ensemble, dtype=np.int64),
        )
        print(f"random walk {time.time() - start:.0f}s", flush=True)

    solver, closed = _uptake_series(base, field, times)

    # --- panel (b): the two clocks, in the continuum ---
    start = time.time()
    long_config = replace(
        base,
        simulation=replace(
            base.simulation, duration=TWO_CLOCK_MIXING_TIMES * mixing, dt="auto"
        ),
        density=replace(base.density, time_slices=301),
    )
    _require_clean_flux(long_config, field, "two-clock run")
    clocks = ob.finite_volume_observables(long_config, field)
    print(f"two-clock run {time.time() - start:.0f}s", flush=True)

    # --- panel (c): distinguishability on the paper's landscape ---
    start = time.time()
    paper = load_config(ROOT / "config" / "default.toml")
    paper = replace(
        paper,
        simulation=replace(
            paper.simulation,
            duration=DISCRIMINATION_DURATION,
            dt=DISCRIMINATION_STEP,
        ),
        density=replace(paper.density, time_slices=241),
    )
    paper_field = create_field(paper.field)
    _require_clean_flux(paper, paper_field, "discrimination run")
    evidence = ob.finite_volume_observables(paper, paper_field)
    print(f"discrimination run {time.time() - start:.0f}s", flush=True)

    _write_metrics(
        base,
        field,
        times,
        particle_uptake,
        particle_error,
        solver,
        closed,
        clocks,
        evidence,
        paper,
        harvest,
        mixing,
        ensemble,
    )
    render_foraging_report(
        base,
        field,
        times=times,
        particle_uptake=particle_uptake,
        particle_error=particle_error,
        solver_uptake=solver.uptake,
        closed_uptake=closed,
        clocks=clocks,
        evidence=evidence,
        evidence_config=paper,
        figure_path=FIGURE,
        also_png=False,
    )
    render_foraging_report(
        base,
        field,
        times=times,
        particle_uptake=particle_uptake,
        particle_error=particle_error,
        solver_uptake=solver.uptake,
        closed_uptake=closed,
        clocks=clocks,
        evidence=evidence,
        evidence_config=paper,
        figure_path=SUPPLEMENT,
    )
    print(f"wrote {METRICS}\nwrote {FIGURE}\nwrote {SUPPLEMENT}")


def _write_metrics(
    config,
    field,
    times,
    particle_uptake,
    particle_error,
    solver,
    closed,
    clocks,
    evidence,
    paper,
    harvest,
    mixing,
    ensemble,
) -> None:
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    with METRICS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "case",
                "kappa",
                "predicted_gain",
                "closed_form_gain",
                "polar_pde_gain",
                "particle_gain",
                "particle_standard_error",
                "t_over_tau_landscape",
                "predicted_discrimination_exponent",
                "fitted_discrimination_exponent",
            )
        )
        expected = {0: None, 1: 2, 2: 6, 3: 4, 4: 6, 5: 4, 6: 4, 7: 2, 8: 4, 9: 4}
        window = slice(20, 140)
        evidence_times = evidence.times
        for case_index, case_number in enumerate(PDF_CASES):
            mobility = ob.chemotactic_mobility(config, case_number)
            divergence = evidence.discrimination[case_index]
            positive = divergence[window] > 0.0
            slope = (
                float(
                    np.polyfit(
                        np.log(evidence_times[window][positive]),
                        np.log(divergence[window][positive]),
                        1,
                    )[0]
                )
                if positive.sum() > 3
                else float("nan")
            )
            writer.writerow(
                (
                    case_number,
                    f"{mobility:.8g}",
                    f"{ob.uptake_gain(config, field, case_number):.8g}",
                    f"{closed[case_index, -1] - closed[0, -1]:.8g}",
                    f"{solver.uptake[case_index, -1] - solver.uptake[0, -1]:.8g}",
                    f"{particle_uptake[case_index, -1] - particle_uptake[0, -1]:.8g}",
                    f"{particle_error[case_index, -1]:.8g}",
                    f"{times[-1] / harvest:.6g}",
                    expected[case_number] if expected[case_number] else "",
                    f"{slope:.4g}" if np.isfinite(slope) else "",
                )
            )
    del paper, clocks, mixing, ensemble


if __name__ == "__main__":
    main()
