"""Audit perturbative order and wall localization of Cases 2--5.

The study reverses and rescales each response coefficient while preserving
common particle seeds. For an observable F(alpha)-F(0),

    odd(alpha)  = [F(alpha)-F(-alpha)]/2
    even(alpha) = [F(alpha)+F(-alpha)]/2

isolate its odd (linear/cubic/...) and even (quadratic/quartic/...) parts.
The calculation uses the manuscript's sliding wall law.

Run from the project directory, for example:

    python scripts/check_response_scaling.py --ensemble 2048
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from math import ceil
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from model.solvers.chem_swimmer.accelerated import (
    numba_available_for,
    state_batch_numba,
)
from model.solvers.chem_swimmer.config import ProjectConfig, load_config
from model.solvers.chem_swimmer.fields import create_field
from model.solvers.chem_swimmer.numerics import build_resolution_plan

CASES = (0, 2, 3, 4, 5)
METRICS = (
    "mean_eta",
    "core_eta_mass",
    "wall_eta_mass",
    "wall_probability",
)


def _safe_fixed_step(
    config: ProjectConfig,
    scales: tuple[float, ...],
) -> tuple[ProjectConfig, int]:
    """Choose one dt safe for every positive and negative requested scale."""

    field = create_field(config.field)
    limits = []
    maximum = max(abs(scale) for scale in scales)
    for sign in (-1.0, 1.0):
        signed_responses = replace(
            config.responses,
            **{
                name: sign * maximum * getattr(config.responses, name)
                for name in config.responses.__dataclass_fields__
            },
        )
        candidate = replace(
            config,
            responses=signed_responses,
            numerics=replace(config.numerics, resolved=False),
        )
        limits.append(build_resolution_plan(candidate, field, CASES).dt_limit)
    steps = max(1, ceil(config.simulation.duration / min(limits)))
    dt = config.simulation.duration / steps
    return (
        replace(
            config,
            simulation=replace(config.simulation, dt=dt, save_every=steps),
            numerics=replace(config.numerics, resolved=True),
        ),
        steps,
    )


def _paired_metrics(
    positions: np.ndarray,
    field,
    core_radius: float,
) -> np.ndarray:
    final_positions = positions[:, :, -1]
    eta = np.asarray(field.value(final_positions), dtype=float)
    radius = np.linalg.norm(final_positions, axis=-1)
    core = radius <= core_radius
    wall = ~core
    samples = np.stack(
        (
            eta,
            eta * core,
            eta * wall,
            wall.astype(float),
        ),
        axis=0,
    )
    return samples - samples[..., 0:1]


def _mean_and_sem(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(samples, axis=1)
    if samples.shape[1] == 1:
        return mean, np.zeros_like(mean)
    sem = np.std(samples, axis=1, ddof=1) / np.sqrt(samples.shape[1])
    return mean, sem


def run_audit(
    config: ProjectConfig,
    *,
    ensemble_size: int,
    scales: tuple[float, ...],
    core_radius: float,
    batch_size: int,
) -> list[dict[str, object]]:
    field = create_field(config.field)
    if not numba_available_for(field):
        raise ValueError("the scaling audit requires a built-in chemical field")
    fixed_config, steps = _safe_fixed_step(config, scales)
    seed_sequence = np.random.SeedSequence(fixed_config.simulation.seed)
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(ensemble_size)
    ]
    signed_scales = tuple(
        signed for scale in scales for signed in (-abs(scale), abs(scale))
    )
    raw: dict[float, np.ndarray] = {}

    for scale in signed_scales:
        chunks = []
        for start in range(0, ensemble_size, batch_size):
            positions, _, _ = state_batch_numba(
                fixed_config,
                field,
                seeds[start : start + batch_size],
                isotropic_headings=True,
                uniform_positions=True,
                case_numbers=CASES,
                response_scale=scale,
            )
            chunks.append(_paired_metrics(positions, field, core_radius))
        samples = np.concatenate(chunks, axis=1)
        raw[scale] = samples
        print(
            f"finished sliding wall, scale={scale:+.3g} "
            f"(M={ensemble_size}, steps={steps})",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    boundary = "sliding"
    for scale in scales:
        magnitude = abs(scale)
        positive_samples = raw[magnitude]
        negative_samples = raw[-magnitude]
        odd_samples = 0.5 * (positive_samples - negative_samples)
        even_samples = 0.5 * (positive_samples + negative_samples)
        positive_mean, positive_sem = _mean_and_sem(positive_samples)
        negative_mean, negative_sem = _mean_and_sem(negative_samples)
        odd, odd_sem = _mean_and_sem(odd_samples)
        even, even_sem = _mean_and_sem(even_samples)
        for metric_index, metric in enumerate(METRICS):
            for case_index, case_number in enumerate(CASES[1:], start=1):
                for component, value, error in (
                    (
                        "positive_raw",
                        positive_mean[metric_index, case_index],
                        positive_sem[metric_index, case_index],
                    ),
                    (
                        "negative_raw",
                        negative_mean[metric_index, case_index],
                        negative_sem[metric_index, case_index],
                    ),
                    (
                        "odd",
                        odd[metric_index, case_index],
                        odd_sem[metric_index, case_index],
                    ),
                    (
                        "even",
                        even[metric_index, case_index],
                        even_sem[metric_index, case_index],
                    ),
                ):
                    rows.append(
                        {
                            "boundary": boundary,
                            "case": case_number,
                            "scale_magnitude": magnitude,
                            "component": component,
                            "metric": metric,
                            "estimate": float(value),
                            "paired_standard_error": float(error),
                            "ensemble_size": ensemble_size,
                            "dt": float(fixed_config.simulation.dt),
                            "steps": steps,
                            "core_radius": core_radius,
                            "seed": fixed_config.simulation.seed,
                        }
                    )
    return rows


def _write_csv(rows: list[dict[str, object]], output: Path) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def _print_summary(rows: list[dict[str, object]]) -> None:
    print("\nOdd/even final mean-eta response")
    print("boundary  case  |alpha|       odd (SE)          even (SE)")
    for boundary in dict.fromkeys(str(row["boundary"]) for row in rows):
        for scale in dict.fromkeys(float(row["scale_magnitude"]) for row in rows):
            for case_number in CASES[1:]:
                selected = {
                    str(row["component"]): row
                    for row in rows
                    if row["boundary"] == boundary
                    and row["case"] == case_number
                    and row["scale_magnitude"] == scale
                    and row["metric"] == "mean_eta"
                    and row["component"] in {"odd", "even"}
                }
                odd = selected["odd"]
                even = selected["even"]
                print(
                    f"{boundary:<9} C{case_number:<3d} {scale:6.3f}  "
                    f"{float(odd['estimate']):+.4e} "
                    f"({float(odd['paired_standard_error']):.2e})  "
                    f"{float(even['estimate']):+.4e} "
                    f"({float(even['paired_standard_error']):.2e})"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "model" / "parameters" / "default.toml",
    )
    parser.add_argument("--ensemble", type=int, default=1024)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--scales", default="0.25,0.5,1.0")
    parser.add_argument("--core-radius", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "simulations"
        / "population_release"
        / "data"
        / "response_scaling_audit.csv",
    )
    args = parser.parse_args()
    scales = tuple(float(value) for value in args.scales.split(","))
    if args.ensemble < 2 or args.batch_size < 1:
        raise ValueError("--ensemble must be at least 2 and --batch-size positive")
    if not scales or any(not 0.0 < abs(scale) <= 1.0 for scale in scales):
        raise ValueError("--scales must contain nonzero magnitudes no larger than 1")
    if not 0.0 < args.core_radius < 1.0:
        raise ValueError("--core-radius must lie in (0, 1)")
    config = load_config(args.config)
    if args.duration is not None:
        config = config.with_overrides(duration=args.duration)
    rows = run_audit(
        config,
        ensemble_size=args.ensemble,
        scales=tuple(sorted({abs(scale) for scale in scales})),
        core_radius=args.core_radius,
        batch_size=args.batch_size,
    )
    output = _write_csv(rows, args.output)
    _print_summary(rows)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
