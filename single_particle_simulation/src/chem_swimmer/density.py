"""Time-dependent spatial probability densities from paired ensembles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path

import numpy as np
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from ._version import PDF_SCHEMA_VERSION
from .config import DensitySettings, ProjectConfig
from .fields import create_field
from .noise import generate_density_particle_draws
from .numerics import inferred_characteristic_length, resolve_numerics
from .progress import progress_bar
from .provenance import ArtifactProvenance, build_provenance
from .simulator import simulate

PDF_CASES = tuple(range(9))
MICROSCOPIC_OBSERVABLE_NAMES = (
    "mean_concentration",
    "alignment",
    "nematic_alignment",
    "heading_correlation_1",
    "heading_correlation_2",
    "cumulative_tumbles",
    "mean_squared_displacement",
    "mean_quartic_displacement",
)


@dataclass(frozen=True)
class MicroscopicObservables:
    """Ensemble moments reduced from particle states before each batch is freed."""

    names: tuple[str, ...]
    values: NDArray[np.float64]
    paired_standard_errors: NDArray[np.float64]


@dataclass(frozen=True)
class PositionPDFResult:
    """Dimensionless ``p(x,t)`` on the unit disk for Cases 0-8."""

    times: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    density: NDArray[np.float32]
    case_numbers: tuple[int, ...]
    ensemble_size: int
    bandwidth_fraction: float
    difference_density: NDArray[np.float32] | None = None
    difference_bandwidth_fraction: float | None = None
    config_source: str = ""
    seed: int = 0
    dt: float = 0.0
    method: str = "micro"
    microscopic_observables: MicroscopicObservables | None = None
    schema_version: int = PDF_SCHEMA_VERSION
    software_version: str = ""
    config_text: str = ""
    config_sha256: str = ""
    runtime_versions: str = ""


def recommended_ensemble_size(settings: DensitySettings) -> int:
    """Minimum M satisfying the declared uniform-kernel occupancy heuristic.

    This is a variance-screening rule, not a bandwidth optimizer or a
    convergence proof. Scientific conclusions still require explicit
    refinement in both ``M`` and the KDE bandwidth.
    """

    raw = ceil(settings.min_samples_per_kernel / (2.0 * settings.bandwidth_fraction**2))
    return 1 << max(0, raw - 1).bit_length()


def resolve_density_ensemble_size(
    settings: DensitySettings,
    requested: int | None = None,
) -> int:
    minimum = recommended_ensemble_size(settings)
    configured = settings.ensemble_size if requested is None else requested
    if isinstance(configured, str):
        return minimum
    count = int(configured)
    if count < minimum:
        raise ValueError(
            f"M={count} is below the configured occupancy floor for "
            f"density.bandwidth_fraction={settings.bandwidth_fraction:.4g}; "
            f"use M >= {minimum}, ensemble_size='auto', a wider bandwidth, "
            "or a smaller density.min_samples_per_kernel. Passing this floor "
            "does not replace an M/bandwidth convergence study."
        )
    return count


def resolve_difference_bandwidth(
    settings: DensitySettings,
    characteristic_length: float,
) -> float:
    """Choose a smoother difference KDE that still resolves the input field."""

    configured = settings.difference_bandwidth_fraction
    if isinstance(configured, str):
        target = min(
            2.0 * settings.bandwidth_fraction,
            characteristic_length / 3.0,
            0.25,
        )
        return max(settings.bandwidth_fraction, target)
    return float(configured)


def expected_samples_per_kernel(
    ensemble_size: int,
    bandwidth_fraction: float,
) -> float:
    """Uniform-disk estimate in one two-dimensional Gaussian footprint."""

    return 2.0 * ensemble_size * bandwidth_fraction**2


def _density_config(config: ProjectConfig) -> ProjectConfig:
    if isinstance(config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    steps = round(config.simulation.duration / config.simulation.dt)
    interval = max(
        1,
        ceil(steps / (config.density.time_slices - 1)),
    )
    return replace(
        config,
        simulation=replace(config.simulation, save_every=interval),
    )


def _saved_times(config: ProjectConfig) -> NDArray[np.float64]:
    if isinstance(config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    steps = round(config.simulation.duration / config.simulation.dt)
    save_steps = np.arange(
        0,
        steps + 1,
        config.simulation.save_every,
        dtype=np.int64,
    )
    if save_steps[-1] != steps:
        save_steps = np.append(save_steps, steps)
    time_scale = config.domain.radius**2 / config.kinetics.D_T0
    times = save_steps.astype(float) * config.simulation.dt / time_scale
    times[-1] = config.simulation.duration / time_scale
    return times


@njit(cache=True)
def _accumulate_histograms_kernel(
    counts: NDArray[np.uint32],
    positions: NDArray[np.float64],
) -> None:
    """Accumulate a batch without large broadcast temporaries or ``np.add.at``."""

    batch_size, case_count, time_count, _ = positions.shape
    grid_size = counts.shape[-1]
    scale = 0.5 * grid_size
    for particle in range(batch_size):
        for case_index in range(case_count):
            for time_index in range(time_count):
                x_bin = int(
                    (positions[particle, case_index, time_index, 0] + 1.0) * scale
                )
                y_bin = int(
                    (positions[particle, case_index, time_index, 1] + 1.0) * scale
                )
                x_bin = min(grid_size - 1, max(0, x_bin))
                y_bin = min(grid_size - 1, max(0, y_bin))
                counts[case_index, time_index, y_bin, x_bin] += 1


def _accumulate_histograms(
    counts: NDArray[np.uint32],
    positions: NDArray[np.float64],
) -> None:
    _accumulate_histograms_kernel(counts, positions)


def _microscopic_observable_samples(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    cumulative_tumbles: NDArray[np.uint32],
    field,
    radius: float,
) -> tuple[NDArray[np.float64], ...]:
    """Return per-particle moments without retaining complete trajectories."""

    input_unit_positions = positions * radius
    concentration = np.asarray(field.value(input_unit_positions), dtype=float)
    gradients = np.asarray(field.gradient(input_unit_positions), dtype=float) * radius
    cosine = np.cos(headings)
    sine = np.sin(headings)
    alignment = cosine * gradients[..., 0] + sine * gradients[..., 1]
    gradient_norm = np.linalg.norm(gradients, axis=-1)
    nematic_alignment = np.zeros_like(alignment)
    nonzero_gradient = gradient_norm > np.finfo(float).eps
    nematic_alignment[nonzero_gradient] = (
        2.0 * alignment[nonzero_gradient] ** 2
        / gradient_norm[nonzero_gradient]
        - gradient_norm[nonzero_gradient]
    )

    heading_change = headings - headings[:, :, 0, None]
    displacement = positions - positions[:, :, 0:1, :]
    squared_displacement = np.sum(displacement * displacement, axis=-1)
    return (
        concentration,
        alignment,
        nematic_alignment,
        np.cos(heading_change),
        np.cos(2.0 * heading_change),
        cumulative_tumbles.astype(float),
        squared_displacement,
        squared_displacement * squared_displacement,
    )


def _accumulate_microscopic_observables(
    sums: NDArray[np.float64],
    paired_difference_squares: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    cumulative_tumbles: NDArray[np.uint32],
    field,
    radius: float,
) -> None:
    samples = _microscopic_observable_samples(
        positions,
        headings,
        cumulative_tumbles,
        field,
        radius,
    )
    for observable_index, values in enumerate(samples):
        sums[observable_index] += np.sum(values, axis=0, dtype=np.float64)
        paired_difference = values - values[:, 0:1]
        paired_difference_squares[observable_index] += np.sum(
            paired_difference * paired_difference,
            axis=0,
            dtype=np.float64,
        )


def _finalize_microscopic_observables(
    sums: NDArray[np.float64],
    paired_difference_squares: NDArray[np.float64],
    ensemble_size: int,
) -> MicroscopicObservables:
    values = sums / ensemble_size
    mean_difference = values - values[:, 0:1]
    standard_errors = np.zeros_like(values)
    if ensemble_size > 1:
        centered_sum_squares = (
            paired_difference_squares - ensemble_size * mean_difference**2
        )
        sample_variance = np.maximum(centered_sum_squares, 0.0) / (
            ensemble_size - 1
        )
        standard_errors = np.sqrt(sample_variance / ensemble_size)
    return MicroscopicObservables(
        names=MICROSCOPIC_OBSERVABLE_NAMES,
        values=values,
        paired_standard_errors=standard_errors,
    )


def _convolve_axis(
    values: NDArray[np.float64],
    kernel: NDArray[np.float64],
    axis: int,
) -> NDArray[np.float64]:
    radius = len(kernel) // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="constant")
    windows = sliding_window_view(padded, len(kernel), axis=axis)
    return np.tensordot(windows, kernel, axes=([-1], [0]))


def _smooth_stack(
    values: NDArray[np.float64],
    kernel: NDArray[np.float64],
) -> NDArray[np.float64]:
    first = _convolve_axis(values, kernel, axis=-1)
    return _convolve_axis(first, kernel, axis=-2)


def smooth_density_stack(
    values: np.ndarray,
    settings: DensitySettings,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Apply the same boundary-corrected Gaussian observation KDE to a stack."""

    grid_size = settings.grid_size
    if values.shape[-2:] != (grid_size, grid_size):
        raise ValueError(
            f"density stack ends in {values.shape[-2:]}, expected "
            f"({grid_size}, {grid_size})"
        )
    cell_width = 2.0 / grid_size
    coordinates = -1.0 + (np.arange(grid_size) + 0.5) * cell_width
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    disk_mask = x_grid * x_grid + y_grid * y_grid <= 1.0

    sigma_bins = settings.bandwidth_fraction / cell_width
    kernel_radius = max(1, ceil(4.0 * sigma_bins))
    offsets = np.arange(-kernel_radius, kernel_radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= np.sum(kernel)

    support = _smooth_stack(disk_mask[None].astype(float), kernel)[0]
    support = np.maximum(support, np.finfo(float).eps)
    cell_area = cell_width * cell_width
    density = np.empty(values.shape, dtype=np.float32)
    flat_values = values.reshape((-1, grid_size, grid_size))
    flat_density = density.reshape((-1, grid_size, grid_size))

    chunk_size = 16
    for start in range(0, len(flat_values), chunk_size):
        stop = min(len(flat_values), start + chunk_size)
        smoothed = _smooth_stack(
            flat_values[start:stop].astype(float),
            kernel,
        )
        smoothed /= support[None]
        smoothed[:, ~disk_mask] = 0.0
        normalization = np.sum(smoothed, axis=(-2, -1)) * cell_area
        if not np.all(np.isfinite(normalization)) or np.any(normalization <= 0.0):
            raise ValueError("density stack cannot be normalized after KDE smoothing")
        smoothed /= normalization[:, None, None]
        flat_density[start:stop] = smoothed.astype(np.float32)
    return density, coordinates


def _density_from_counts(
    counts: NDArray[np.uint32],
    ensemble_size: int,
    settings: DensitySettings,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    density, coordinates = smooth_density_stack(counts, settings)

    # The normalization above uses counts only through a common multiplicative
    # factor, so M cancels. Keep this explicit check tied to the requested M.
    sample_totals = np.sum(counts, axis=(-2, -1))
    if not np.all(sample_totals == ensemble_size):
        raise RuntimeError("the density histogram lost ensemble samples")
    return density, coordinates


def compute_position_pdf(
    config: ProjectConfig,
    *,
    ensemble_size: int | None = None,
    show_progress: bool = False,
) -> PositionPDFResult:
    """Estimate dimensionless ``p(x,t)`` for the baseline and eight cases."""

    field = create_field(config.field)
    config, _ = resolve_numerics(config, field, PDF_CASES)
    density_config = _density_config(config)
    count = resolve_density_ensemble_size(
        density_config.density,
        requested=ensemble_size,
    )
    difference_bandwidth = resolve_difference_bandwidth(
        density_config.density,
        inferred_characteristic_length(density_config, field),
    )
    if show_progress:
        base_samples = expected_samples_per_kernel(
            count,
            density_config.density.bandwidth_fraction,
        )
        difference_samples = expected_samples_per_kernel(
            count,
            difference_bandwidth,
        )
        print(
            "\nProbability-density estimator\n"
            f"  M={count:,}, grid={density_config.density.grid_size}x"
            f"{density_config.density.grid_size}\n"
            f"  absolute PDF : b={density_config.density.bandwidth_fraction:.4g}, "
            f"expected uniform samples/kernel={base_samples:.1f}\n"
            f"  difference PDF: b={difference_bandwidth:.4g}, "
            f"expected uniform samples/kernel={difference_samples:.1f}\n"
            "  The difference KDE is computed directly from the same paired "
            "raw histograms.",
            flush=True,
        )
    times = _saved_times(density_config)
    grid_size = density_config.density.grid_size
    counts = np.zeros(
        (len(PDF_CASES), len(times), grid_size, grid_size),
        dtype=np.uint32,
    )
    observable_sums = np.zeros(
        (len(MICROSCOPIC_OBSERVABLE_NAMES), len(PDF_CASES), len(times)),
        dtype=np.float64,
    )
    observable_paired_squares = np.zeros_like(observable_sums)

    seed_sequence = np.random.SeedSequence(density_config.simulation.seed)
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(count)
    ]

    use_numba = False
    if density_config.performance.engine != "python":
        from .accelerated import numba_available_for

        use_numba = numba_available_for(field)
        if density_config.performance.engine == "numba" and not use_numba:
            raise ValueError(
                "Numba density evolution requires a built-in chemical field"
            )

    bar = progress_bar(
        total=count,
        description="Cases 0-8 probability density",
        unit="particle",
        enabled=show_progress,
    )
    if use_numba:
        from .accelerated import state_batch_numba

        batch_size = min(density_config.density.batch_size, count)
        for start in range(0, count, batch_size):
            batch_seeds = seeds[start : start + batch_size]
            positions, headings, cumulative_tumbles = state_batch_numba(
                density_config,
                field,
                batch_seeds,
                isotropic_headings=(
                    density_config.density.initial_heading_distribution == "isotropic"
                ),
                uniform_positions=(
                    density_config.density.initial_position_distribution == "uniform"
                ),
            )
            _accumulate_histograms(
                counts,
                positions,
            )
            _accumulate_microscopic_observables(
                observable_sums,
                observable_paired_squares,
                positions,
                headings,
                cumulative_tumbles,
                field,
                density_config.domain.radius,
            )
            bar.update(len(batch_seeds))
    else:
        if count > 512:
            raise ValueError(
                "the custom-field Python engine would be impractical for "
                f"M={count}; use a built-in field with Numba or explicitly "
                "lower the density quality target"
            )
        if isinstance(density_config.simulation.dt, str):
            raise TypeError("internal error: automatic dt was not resolved")
        steps = round(density_config.simulation.duration / density_config.simulation.dt)
        for seed in seeds:
            draws = generate_density_particle_draws(
                steps,
                seed,
                initial_position=density_config.initial.position,
                initial_heading=density_config.initial.heading,
                uniform_initial_position=(
                    density_config.density.initial_position_distribution == "uniform"
                ),
                isotropic_initial_heading=(
                    density_config.density.initial_heading_distribution == "isotropic"
                ),
                boundary=density_config.domain.boundary,
            )
            particle_config = replace(
                density_config,
                initial=replace(
                    density_config.initial,
                    position=draws.initial_position,
                    heading=draws.initial_heading,
                ),
            )
            positions = np.empty((1, 9, len(times), 2), dtype=float)
            headings = np.empty((1, 9, len(times)), dtype=float)
            cumulative_tumbles = np.empty(
                (1, 9, len(times)),
                dtype=np.uint32,
            )
            for case_number in PDF_CASES:
                trajectory = simulate(
                    particle_config,
                    case_number,
                    field=field,
                    innovations=draws.innovations,
                )
                positions[0, case_number] = trajectory.positions
                headings[0, case_number] = trajectory.headings
                cumulative_tumbles[0, case_number] = np.cumsum(
                    trajectory.tumble_counts,
                    dtype=np.uint32,
                )
            _accumulate_histograms(
                counts,
                positions,
            )
            _accumulate_microscopic_observables(
                observable_sums,
                observable_paired_squares,
                positions,
                headings,
                cumulative_tumbles,
                field,
                density_config.domain.radius,
            )
            bar.update(1)
    bar.close()

    density, coordinates = _density_from_counts(
        counts,
        count,
        density_config.density,
    )
    if np.isclose(
        difference_bandwidth,
        density_config.density.bandwidth_fraction,
        rtol=0.0,
        atol=1.0e-15,
    ):
        difference_density = density
    else:
        difference_settings = replace(
            density_config.density,
            bandwidth_fraction=difference_bandwidth,
        )
        difference_density, difference_coordinates = _density_from_counts(
            counts,
            count,
            difference_settings,
        )
        if not np.array_equal(coordinates, difference_coordinates):
            raise RuntimeError("density estimators produced different grids")
    if isinstance(density_config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    provenance = build_provenance(str(density_config.source_path))
    return PositionPDFResult(
        times=times,
        coordinates=coordinates,
        density=density,
        case_numbers=PDF_CASES,
        ensemble_size=count,
        bandwidth_fraction=density_config.density.bandwidth_fraction,
        difference_density=difference_density,
        difference_bandwidth_fraction=difference_bandwidth,
        config_source=str(density_config.source_path),
        seed=density_config.simulation.seed,
        dt=float(density_config.simulation.dt),
        method="micro",
        microscopic_observables=_finalize_microscopic_observables(
            observable_sums,
            observable_paired_squares,
            count,
        ),
        schema_version=provenance.schema_version,
        software_version=provenance.software_version,
        config_text=provenance.config_text,
        config_sha256=provenance.config_sha256,
        runtime_versions=provenance.runtime_versions,
    )


def save_position_pdf(
    result: PositionPDFResult,
    output_path: str | Path,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    difference_density = (
        result.density
        if result.difference_density is None
        else result.difference_density
    )
    difference_bandwidth = (
        result.bandwidth_fraction
        if result.difference_bandwidth_fraction is None
        else result.difference_bandwidth_fraction
    )
    observables = result.microscopic_observables
    observable_names = (
        np.asarray((), dtype="U1")
        if observables is None
        else np.asarray(observables.names)
    )
    observable_values = (
        np.empty((0, len(result.case_numbers), len(result.times)), dtype=float)
        if observables is None
        else observables.values
    )
    observable_standard_errors = (
        np.empty_like(observable_values)
        if observables is None
        else observables.paired_standard_errors
    )
    provenance = (
        ArtifactProvenance(
            schema_version=result.schema_version,
            software_version=result.software_version,
            config_text=result.config_text,
            config_sha256=result.config_sha256,
            runtime_versions=result.runtime_versions,
        )
        if result.config_text
        else build_provenance(result.config_source)
    )
    np.savez_compressed(
        output,
        schema_version=np.asarray(provenance.schema_version, dtype=np.int64),
        software_version=np.asarray(provenance.software_version),
        config_text=np.asarray(provenance.config_text),
        config_sha256=np.asarray(provenance.config_sha256),
        runtime_versions=np.asarray(provenance.runtime_versions),
        time=result.times,
        coordinates=result.coordinates,
        density=result.density,
        difference_density=difference_density,
        global_density_max=np.asarray(np.max(result.density)),
        case_numbers=np.asarray(result.case_numbers, dtype=np.int64),
        ensemble_size=np.asarray(result.ensemble_size, dtype=np.int64),
        bandwidth_fraction=np.asarray(result.bandwidth_fraction),
        difference_bandwidth_fraction=np.asarray(difference_bandwidth),
        config_source=np.asarray(result.config_source),
        seed=np.asarray(result.seed, dtype=np.int64),
        dt=np.asarray(result.dt),
        method=np.asarray(result.method),
        microscopic_observable_names=observable_names,
        microscopic_observable_values=observable_values,
        microscopic_observable_paired_standard_errors=observable_standard_errors,
    )
    return output


def load_position_pdf(input_path: str | Path) -> PositionPDFResult:
    """Load a density result written by :func:`save_position_pdf`."""

    path = Path(input_path).expanduser().resolve()
    with np.load(path) as saved:
        schema_version = (
            int(saved["schema_version"]) if "schema_version" in saved.files else 1
        )
        if schema_version > PDF_SCHEMA_VERSION:
            raise ValueError(
                f"{path} uses unsupported density schema {schema_version}; "
                "upgrade chem-swimmer before loading it"
            )
        observable_names = tuple(
            str(name) for name in saved["microscopic_observable_names"]
        )
        observables = None
        if observable_names:
            observables = MicroscopicObservables(
                names=observable_names,
                values=saved["microscopic_observable_values"].astype(float),
                paired_standard_errors=saved[
                    "microscopic_observable_paired_standard_errors"
                ].astype(float),
            )
        return PositionPDFResult(
            times=saved["time"].astype(float),
            coordinates=saved["coordinates"].astype(float),
            density=saved["density"].astype(np.float32),
            difference_density=saved["difference_density"].astype(np.float32),
            case_numbers=tuple(
                int(case_number) for case_number in saved["case_numbers"]
            ),
            ensemble_size=int(saved["ensemble_size"]),
            bandwidth_fraction=float(saved["bandwidth_fraction"]),
            difference_bandwidth_fraction=float(
                saved["difference_bandwidth_fraction"]
            ),
            config_source=str(saved["config_source"]),
            seed=int(saved["seed"]),
            dt=float(saved["dt"]),
            method=str(saved["method"]),
            microscopic_observables=observables,
            schema_version=schema_version,
            software_version=(
                str(saved["software_version"])
                if "software_version" in saved.files
                else ""
            ),
            config_text=(
                str(saved["config_text"]) if "config_text" in saved.files else ""
            ),
            config_sha256=(
                str(saved["config_sha256"])
                if "config_sha256" in saved.files
                else ""
            ),
            runtime_versions=(
                str(saved["runtime_versions"])
                if "runtime_versions" in saved.files
                else ""
            ),
        )
