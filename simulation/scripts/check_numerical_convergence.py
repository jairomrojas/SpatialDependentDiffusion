"""Generate microscopic timestep and macroscopic grid convergence evidence."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chem_swimmer.accelerated import state_batch_numba
from chem_swimmer.config import load_config
from chem_swimmer.density import PDF_CASES, smooth_density_stack
from chem_swimmer.fields import create_field
from chem_swimmer.kinetics import macroscopic_coefficients
from chem_swimmer.numerics import build_resolution_plan

MICRO_CASES = (0, 2, 3, 4, 5)


def _micro_timestep_rows(
    config,
    *,
    ensemble_size: int,
    refinement_factors: tuple[int, ...],
    batch_size: int,
) -> list[dict[str, object]]:
    field = create_field(config.field)
    plan = build_resolution_plan(config, field, MICRO_CASES)
    base_steps = round(config.simulation.duration / plan.dt)
    seed_sequence = np.random.SeedSequence(config.simulation.seed)
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(ensemble_size)
    ]
    rows = []
    for factor in refinement_factors:
        steps = base_steps * factor
        dt = config.simulation.duration / steps
        fixed = replace(
            config,
            simulation=replace(config.simulation, dt=dt, save_every=steps),
            numerics=replace(config.numerics, resolved=True),
        )
        sample_chunks = []
        for start in range(0, ensemble_size, batch_size):
            positions, _, _ = state_batch_numba(
                fixed,
                field,
                seeds[start : start + batch_size],
                isotropic_headings=True,
                uniform_positions=True,
                case_numbers=MICRO_CASES,
            )
            final_eta = np.asarray(
                field.value(positions[:, :, -1]),
                dtype=float,
            )
            sample_chunks.append(final_eta - final_eta[:, 0:1])
        samples = np.concatenate(sample_chunks, axis=0)
        means = np.mean(samples, axis=0)
        standard_errors = np.std(samples, axis=0, ddof=1) / np.sqrt(ensemble_size)
        for case_index, case_number in enumerate(MICRO_CASES):
            rows.append(
                {
                    "audit": "micro_timestep",
                    "case": case_number,
                    "resolution": factor,
                    "dt_or_spacing": dt,
                    "steps_or_grid": steps,
                    "estimate": float(means[case_index]),
                    "standard_error": float(standard_errors[case_index]),
                    "ensemble_size": ensemble_size,
                    "bandwidth": "",
                    "seed": config.simulation.seed,
                }
            )
        print(
            f"finished micro dt refinement {factor}x "
            f"(dt={dt:.6g}, steps={steps}, M={ensemble_size})",
            flush=True,
        )
    return rows


def _macro_equilibrium_rows(
    config,
    *,
    grid_sizes: tuple[int, ...],
) -> list[dict[str, object]]:
    field = create_field(config.field)
    coefficients = [
        macroscopic_coefficients(
            config.kinetics,
            config.responses.only_case(case_number),
        )
        for case_number in PDF_CASES
    ]
    effective_diffusivity = coefficients[0][1]
    mobilities = np.asarray([coefficient[2] for coefficient in coefficients])
    rows = []
    for grid_size in grid_sizes:
        spacing = 2.0 / grid_size
        coordinates = -1.0 + (np.arange(grid_size) + 0.5) * spacing
        x_grid, y_grid = np.meshgrid(coordinates, coordinates)
        points = np.stack((x_grid, y_grid), axis=-1)
        mask = x_grid * x_grid + y_grid * y_grid <= 1.0
        eta = np.asarray(field.value(points), dtype=float)
        raw = np.zeros((len(PDF_CASES), grid_size, grid_size), dtype=float)
        for case_index, mobility in enumerate(mobilities):
            exponent = mobility * eta[mask] / effective_diffusivity
            exponent -= np.max(exponent)
            raw[case_index, mask] = np.exp(exponent)
            raw[case_index] /= np.sum(raw[case_index]) * spacing**2
        observed, _ = smooth_density_stack(
            raw,
            replace(config.density, grid_size=grid_size),
        )
        difference = observed.astype(float) - observed[0:1].astype(float)
        exposure = np.sum(difference * eta[None], axis=(-2, -1)) * spacing**2
        for case_number in PDF_CASES:
            rows.append(
                {
                    "audit": "macro_equilibrium_grid",
                    "case": case_number,
                    "resolution": grid_size,
                    "dt_or_spacing": spacing,
                    "steps_or_grid": grid_size,
                    "estimate": float(exposure[case_number]),
                    "standard_error": 0.0,
                    "ensemble_size": 0,
                    "bandwidth": config.density.bandwidth_fraction,
                    "seed": config.simulation.seed,
                }
            )
        print(
            f"finished macro equilibrium grid {grid_size}x{grid_size}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.toml",
    )
    parser.add_argument("--ensemble", type=int, default=4096)
    parser.add_argument("--dt-factors", default="1,2,4")
    parser.add_argument("--grid-sizes", default="81,121,161,241,321")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "data" / "numerical_convergence_audit.csv",
    )
    args = parser.parse_args()
    factors = tuple(sorted({int(value) for value in args.dt_factors.split(",")}))
    grids = tuple(sorted({int(value) for value in args.grid_sizes.split(",")}))
    if args.ensemble < 2 or args.batch_size < 1:
        raise ValueError("--ensemble must be at least 2 and --batch-size positive")
    if not factors or any(factor < 1 for factor in factors):
        raise ValueError("--dt-factors must contain positive integers")
    if not grids or any(grid < 33 for grid in grids):
        raise ValueError("--grid-sizes must contain integers of at least 33")

    config = load_config(args.config)
    rows = _micro_timestep_rows(
        config,
        ensemble_size=args.ensemble,
        refinement_factors=factors,
        batch_size=args.batch_size,
    )
    rows.extend(_macro_equilibrium_rows(config, grid_sizes=grids))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
