"""Paired ensemble statistics for diffusion-like spreading."""

from __future__ import annotations

import csv
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import ProjectConfig
from .fields import create_field
from .noise import Innovations
from .numerics import resolve_numerics
from .progress import progress_bar
from .simulator import simulate
from .turning import create_turning_kernel


@dataclass(frozen=True)
class MSDResult:
    """Mean squared displacement and its standard error for cases 1-9."""

    times: NDArray[np.float64]
    case_numbers: tuple[int, ...]
    mean: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    ensemble_size: int


def _replicate_squared_displacements(
    task: tuple[ProjectConfig, int, tuple[int, ...]],
) -> NDArray[np.float64]:
    """One worker task; every case receives the same innovation stream."""

    config, seed, case_numbers = task
    field = create_field(config.field)
    steps = round(config.simulation.duration / config.simulation.dt)
    common_noise = Innovations.generate(steps, seed)
    initial = np.asarray(config.initial.position) / config.domain.radius
    squared_displacements = []

    for case_number in case_numbers:
        trajectory = simulate(
            config,
            case_number,
            field=field,
            innovations=common_noise,
        )
        displacement = trajectory.positions - initial
        squared_displacements.append(np.sum(displacement * displacement, axis=1))

    return np.asarray(squared_displacements)


def _worker_count(requested: int, ensemble_size: int) -> int:
    if requested > 0:
        return min(requested, ensemble_size)
    available = os.cpu_count() or 1
    # More than eight Python workers rarely helps this small sequential kernel
    # on typical multicore systems and increases memory/import overhead.
    return min(ensemble_size, available, 8)


def compute_ensemble_msd(
    config: ProjectConfig,
    *,
    ensemble_size: int | None = None,
    workers: int | None = None,
    case_numbers: tuple[int, ...] = tuple(range(1, 10)),
    show_progress: bool = False,
) -> MSDResult:
    """Compute paired MSD curves from independent replicate noise streams."""

    field = create_field(config.field)
    config, _ = resolve_numerics(config, field, case_numbers)
    if isinstance(config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")

    count = config.statistics.ensemble_size if ensemble_size is None else ensemble_size
    if count < 1:
        raise ValueError("ensemble_size must be at least 1")
    requested_workers = config.statistics.workers if workers is None else workers
    number_workers = _worker_count(requested_workers, count)

    seed_sequence = np.random.SeedSequence(config.simulation.seed)
    replicate_seeds = [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in seed_sequence.spawn(count)
    ]
    tasks = [(config, seed, case_numbers) for seed in replicate_seeds]

    use_numba = False
    if config.performance.engine != "python":
        from .accelerated import numba_available_for

        use_numba = numba_available_for(
            field, create_turning_kernel(config.turning)
        ) and case_numbers == tuple(range(1, 10))
        if config.performance.engine == "numba" and not use_numba:
            raise ValueError(
                "Numba MSD requires cases 1-9, a built-in chemical field, and "
                "the wrapped-normal turning kernel"
            )

    if use_numba:
        from .accelerated import msd_batch_numba

        sum_values: NDArray[np.float64] | None = None
        sum_squares: NDArray[np.float64] | None = None
        bar = progress_bar(
            total=count,
            description="Paired ensemble MSD",
            unit="replicate",
            enabled=show_progress,
        )
        batch_size = min(config.performance.msd_batch_size, count)
        for start in range(0, count, batch_size):
            batch_seeds = replicate_seeds[start : start + batch_size]
            samples = msd_batch_numba(config, field, batch_seeds)
            batch_sum = np.sum(samples, axis=0)
            batch_sum_squares = np.sum(samples * samples, axis=0)
            if sum_values is None:
                sum_values = batch_sum
                sum_squares = batch_sum_squares
            else:
                sum_values += batch_sum
                sum_squares += batch_sum_squares
            bar.update(len(batch_seeds))
        bar.close()
        assert sum_values is not None and sum_squares is not None
        mean = sum_values / count
        if count == 1:
            standard_error = np.zeros_like(mean)
        else:
            sample_variance = np.maximum(
                0.0,
                (sum_squares - count * mean * mean) / (count - 1),
            )
            standard_error = np.sqrt(sample_variance / count)
    elif number_workers == 1:
        task_iterator = progress_bar(
            tasks,
            total=count,
            description="Paired ensemble MSD",
            unit="replicate",
            enabled=show_progress,
        )
        replicates = [_replicate_squared_displacements(task) for task in task_iterator]
    else:
        try:
            executor = ProcessPoolExecutor(max_workers=number_workers)
        except OSError as exc:
            warnings.warn(
                f"parallel workers are unavailable ({exc}); using serial MSD computation",
                RuntimeWarning,
                stacklevel=2,
            )
            task_iterator = progress_bar(
                tasks,
                total=count,
                description="Paired ensemble MSD",
                unit="replicate",
                enabled=show_progress,
            )
            replicates = [
                _replicate_squared_displacements(task) for task in task_iterator
            ]
        else:
            with executor:
                mapped = executor.map(_replicate_squared_displacements, tasks)
                replicates = list(
                    progress_bar(
                        mapped,
                        total=count,
                        description="Paired ensemble MSD",
                        unit="replicate",
                        enabled=show_progress,
                    )
                )

    if not use_numba:
        samples = np.asarray(replicates)
        mean = np.mean(samples, axis=0)
        if count == 1:
            standard_error = np.zeros_like(mean)
        else:
            standard_error = np.std(samples, axis=0, ddof=1) / np.sqrt(count)

    # All trajectories share the same save schedule.
    steps = round(config.simulation.duration / config.simulation.dt)
    save_steps = np.arange(0, steps + 1, config.simulation.save_every, dtype=int)
    if save_steps[-1] != steps:
        save_steps = np.append(save_steps, steps)
    time_scale = config.domain.radius**2 / config.kinetics.D_T0
    times = save_steps * config.simulation.dt / time_scale
    times[-1] = config.simulation.duration / time_scale

    return MSDResult(
        times=np.asarray(times, dtype=float),
        case_numbers=case_numbers,
        mean=mean,
        standard_error=standard_error,
        ensemble_size=count,
    )


def save_msd_csv(result: MSDResult, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    header = ["time"]
    for case_number in result.case_numbers:
        header.extend((f"case_{case_number}_msd", f"case_{case_number}_sem"))

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for time_index, time in enumerate(result.times):
            row: list[float] = [float(time)]
            for case_index in range(len(result.case_numbers)):
                row.extend(
                    (
                        float(result.mean[case_index, time_index]),
                        float(result.standard_error[case_index, time_index]),
                    )
                )
            writer.writerow(row)
    return output


def free_space_baseline_msd(
    config: ProjectConfig, times: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Dimensionless free-space baseline MSD at dimensionless ``times``."""

    base = config.kinetics
    radius = config.domain.radius
    time_scale = radius**2 / base.D_T0
    P_turn = create_turning_kernel(config.turning).P_turn
    gamma1 = (base.D_R0 + base.lambda0 * (1.0 - P_turn)) * time_scale
    speed = base.v0 * radius / base.D_T0
    if gamma1 == 0.0:
        # With a frozen heading the active displacement is ballistic. This is
        # the continuous gamma1 -> 0 limit of the expression below.
        return 4.0 * times + speed**2 * times**2
    z = gamma1 * times
    relaxation_kernel = np.empty_like(times)
    small = np.abs(z) < 1.0e-4
    # Series for [z - (1-exp(-z))]/gamma1^2 avoids catastrophic
    # cancellation when orientation relaxes extremely slowly.
    relaxation_kernel[small] = times[small] ** 2 * (
        0.5 - z[small] / 6.0 + z[small] ** 2 / 24.0 - z[small] ** 3 / 120.0
    )
    relaxation_kernel[~small] = (z[~small] + np.expm1(-z[~small])) / gamma1**2
    active = 2.0 * speed**2 * relaxation_kernel
    return 4.0 * times + active
