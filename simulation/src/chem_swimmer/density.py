"""Time-dependent spatial probability densities from paired ensembles."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy.ndimage import convolve1d

from ._version import PDF_SCHEMA_VERSION
from .config import DensitySettings, ProjectConfig
from .fields import ChemicalField, create_field
from .noise import generate_density_particle_draws
from .numerics import inferred_characteristic_length, resolve_numerics
from .progress import progress_bar
from .provenance import ArtifactProvenance, build_provenance
from .simulator import simulate
from .turning import create_turning_kernel

PDF_CASES = tuple(range(10))
MICROSCOPIC_OBSERVABLE_NAMES = (
    "mean_eta",
    "zeta",
    "nematic_zeta",
    "heading_correlation_1",
    "heading_correlation_2",
    "cumulative_tumbles",
    "mean_squared_displacement",
    "mean_quartic_displacement",
    # Transport observables of Sec. "Macroscopic observables" in the paper:
    # the net polarity P = <n>, the centroid X = <r>, and the second moment
    # S = <|r|^2>.  Each is a plain ensemble mean, so the particle estimate
    # needs no spatial binning and no kernel bandwidth.
    "net_polarity_x",
    "net_polarity_y",
    "centroid_x",
    "centroid_y",
    "squared_radius",
    # The traceless part of the same second moment: the pulse shape.  It has no
    # uniform background at all, so the whole signal is the landscape response.
    "shape_xx",
    "shape_xy",
)


@dataclass(frozen=True)
class MicroscopicObservables:
    """Ensemble moments reduced from particle states before each batch is freed."""

    names: tuple[str, ...]
    values: NDArray[np.float64]
    paired_standard_errors: NDArray[np.float64]


@dataclass(frozen=True)
class PositionPDFResult:
    """Dimensionless ``p(x,t)`` on the unit disk for Cases 0-9."""

    times: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    density: NDArray[np.float64]
    case_numbers: tuple[int, ...]
    ensemble_size: int
    bandwidth_fraction: float
    difference_density: NDArray[np.float64] | None = None
    difference_bandwidth_fraction: float | None = None
    config_source: str = ""
    seed: int = 0
    dt: float = 0.0
    method: str = "micro"
    microscopic_observables: MicroscopicObservables | None = None
    polarity_x: NDArray[np.float64] | None = None
    polarity_y: NDArray[np.float64] | None = None
    difference_polarity_x: NDArray[np.float64] | None = None
    difference_polarity_y: NDArray[np.float64] | None = None
    nematic_xx: NDArray[np.float64] | None = None
    nematic_xy: NDArray[np.float64] | None = None
    difference_nematic_xx: NDArray[np.float64] | None = None
    difference_nematic_xy: NDArray[np.float64] | None = None
    nematic_split_half_error_xx: NDArray[np.float64] | None = None
    nematic_split_half_error_xy: NDArray[np.float64] | None = None
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


def resolve_initial_gaussian_sigma(
    settings: DensitySettings,
    characteristic_length: float,
) -> float:
    """Width of the "gaussian" initial_position_distribution.

    Defaults to twice the field's own smallest resolved length scale
    (``numerics.inferred_characteristic_length``), so the initial density
    does not vary on a finer scale than the field itself does --
    analytic.py's closed-form solution requires the density gradient to stay
    small (see that module's docstring), and one characteristic length alone
    measurably undershoots that. Checked directly against macro across a
    sigma sweep on this project's own default config (characteristic
    length 0.5), cosine similarity of the case-vs-Case-0 response field
    (the part that actually needs to match, not the much larger shared
    diffusion background) went sigma=0.5 -> 0.93, sigma=0.8 -> 0.98,
    sigma=1.2 -> 0.99. Twice the characteristic length (sigma=1.0 here) is a
    middle point on that curve: a real, measured improvement over one length
    alone, without widening so far that "concentrated near the center"
    stops being a meaningful description of the initial condition.
    """

    configured = settings.initial_gaussian_sigma
    if isinstance(configured, str):
        return 2.0 * characteristic_length
    return float(configured)


def prescribed_initial_density_grid(
    config: ProjectConfig,
    field: ChemicalField,
    grid_points: NDArray[np.float64],
    mask: NDArray[np.bool_],
    spacing: float,
) -> NDArray[np.float64]:
    """Represent the configured spatial ensemble deterministically on a grid.

    This is the common initial condition for the microscopic observation and
    both closed solvers.  The particle dynamics still draw individual initial
    positions from the configured law; only the reported ensemble density at
    the exactly known first slice is made free of finite-M estimator noise.
    """

    density = np.zeros(mask.shape, dtype=float)
    distribution = config.density.initial_position_distribution
    if distribution == "uniform":
        density[mask] = 1.0 / (np.count_nonzero(mask) * spacing**2)
        return density
    center = np.asarray(config.initial.position, dtype=float)
    if distribution == "gaussian":
        sigma = resolve_initial_gaussian_sigma(
            config.density,
            inferred_characteristic_length(config, field),
        )
        squared_distance = np.sum((grid_points - center) ** 2, axis=-1)
        density = np.where(mask, np.exp(-0.5 * squared_distance / sigma**2), 0.0)
        density /= np.sum(density) * spacing**2
        return density
    grid_size = mask.shape[0]
    column = min(grid_size - 1, max(0, int((center[0] + 1.0) * grid_size / 2.0)))
    row = min(grid_size - 1, max(0, int((center[1] + 1.0) * grid_size / 2.0)))
    if mask[row, column]:
        initial_cell = (row, column)
    else:
        squared_distance = np.sum((grid_points - center) ** 2, axis=-1)
        squared_distance[~mask] = np.inf
        initial_cell = np.unravel_index(np.argmin(squared_distance), mask.shape)
    density[initial_cell] = 1.0 / spacing**2
    return density


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


@njit(cache=True, parallel=True)
def _accumulate_histograms_kernel(
    counts: NDArray[np.uint32],
    positions: NDArray[np.float64],
) -> None:
    """Accumulate a batch without large broadcast temporaries or ``np.add.at``.

    Parallel over cases: ``counts[case]`` are disjoint output slices, so the
    scatter has no write conflicts and needs no atomics or private copies.
    """

    batch_size, case_count, time_count, _ = positions.shape
    grid_size = counts.shape[-1]
    scale = 0.5 * grid_size
    for case_index in prange(case_count):
        for particle in range(batch_size):
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


@njit(cache=True, parallel=True)
def _accumulate_angular_moments_kernel(
    sum_cos: NDArray[np.float64],
    sum_sin: NDArray[np.float64],
    sum_qxx: NDArray[np.float64],
    sum_qxy: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> None:
    """Bin polar and traceless-nematic moments with the density samples.

    Parallel over cases, like :func:`_accumulate_histograms_kernel`.
    """

    batch_size, case_count, time_count, _ = positions.shape
    grid_size = sum_cos.shape[-1]
    scale = 0.5 * grid_size
    for case_index in prange(case_count):
        for particle in range(batch_size):
            for time_index in range(time_count):
                x_bin = int(
                    (positions[particle, case_index, time_index, 0] + 1.0) * scale
                )
                y_bin = int(
                    (positions[particle, case_index, time_index, 1] + 1.0) * scale
                )
                x_bin = min(grid_size - 1, max(0, x_bin))
                y_bin = min(grid_size - 1, max(0, y_bin))
                angle = headings[particle, case_index, time_index]
                cosine = np.cos(angle)
                sine = np.sin(angle)
                sum_cos[case_index, time_index, y_bin, x_bin] += cosine
                sum_sin[case_index, time_index, y_bin, x_bin] += sine
                sum_qxx[case_index, time_index, y_bin, x_bin] += cosine**2 - 0.5
                sum_qxy[case_index, time_index, y_bin, x_bin] += cosine * sine


def _accumulate_angular_moments(
    sum_cos: NDArray[np.float64],
    sum_sin: NDArray[np.float64],
    sum_qxx: NDArray[np.float64],
    sum_qxy: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> None:
    _accumulate_angular_moments_kernel(
        sum_cos,
        sum_sin,
        sum_qxx,
        sum_qxy,
        positions,
        headings,
    )


@njit(cache=True, parallel=True)
def _accumulate_nematic_moments_kernel(
    sum_qxx: NDArray[np.float64],
    sum_qxy: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> None:
    """Bin only Q for the independent split-half error estimator.

    Parallel over cases, like :func:`_accumulate_histograms_kernel`.
    """

    batch_size, case_count, time_count, _ = positions.shape
    grid_size = sum_qxx.shape[-1]
    scale = 0.5 * grid_size
    for case_index in prange(case_count):
        for particle in range(batch_size):
            for time_index in range(time_count):
                x_bin = int(
                    (positions[particle, case_index, time_index, 0] + 1.0) * scale
                )
                y_bin = int(
                    (positions[particle, case_index, time_index, 1] + 1.0) * scale
                )
                x_bin = min(grid_size - 1, max(0, x_bin))
                y_bin = min(grid_size - 1, max(0, y_bin))
                angle = headings[particle, case_index, time_index]
                cosine = np.cos(angle)
                sine = np.sin(angle)
                sum_qxx[case_index, time_index, y_bin, x_bin] += cosine**2 - 0.5
                sum_qxy[case_index, time_index, y_bin, x_bin] += cosine * sine


def _accumulate_nematic_moments(
    sum_qxx: NDArray[np.float64],
    sum_qxy: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> None:
    _accumulate_nematic_moments_kernel(sum_qxx, sum_qxy, positions, headings)


def _observable_field_samples(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    field,
    radius: float,
) -> tuple[NDArray[np.float64], ...]:
    """Chemical-field observables at every saved state, natively when possible."""

    from .accelerated import numba_available_for, observable_fields_numba

    if numba_available_for(field):
        return observable_fields_numba(field, radius, positions, headings)

    input_unit_positions = positions * radius
    # One pass, not two: the separate value/gradient calls evaluated every
    # Gaussian source twice over ~M*cases*slices points.
    raw_eta, raw_gradients = field.value_and_gradient(input_unit_positions)
    eta = np.asarray(raw_eta, dtype=float)
    gradients = np.asarray(raw_gradients, dtype=float) * radius
    zeta = np.cos(headings) * gradients[..., 0] + np.sin(headings) * gradients[..., 1]
    gradient_norm = np.linalg.norm(gradients, axis=-1)
    nematic_zeta = np.zeros_like(zeta)
    nonzero_gradient = gradient_norm > np.finfo(float).eps
    nematic_zeta[nonzero_gradient] = (
        2.0 * zeta[nonzero_gradient] ** 2 / gradient_norm[nonzero_gradient]
        - gradient_norm[nonzero_gradient]
    )
    return eta, zeta, nematic_zeta


def _microscopic_observable_samples(
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    cumulative_tumbles: NDArray[np.uint32],
    field,
    radius: float,
) -> tuple[NDArray[np.float64], ...]:
    """Return per-particle moments without retaining complete trajectories."""

    eta, zeta, nematic_zeta = _observable_field_samples(
        positions, headings, field, radius
    )

    heading_change = headings - headings[:, :, 0, None]
    displacement = positions - positions[:, :, 0:1, :]
    squared_displacement = np.sum(displacement * displacement, axis=-1)
    return (
        eta,
        zeta,
        nematic_zeta,
        np.cos(heading_change),
        np.cos(2.0 * heading_change),
        cumulative_tumbles.astype(float),
        squared_displacement,
        squared_displacement * squared_displacement,
        np.cos(headings),
        np.sin(headings),
        positions[..., 0],
        positions[..., 1],
        np.sum(positions * positions, axis=-1),
        positions[..., 0] ** 2 - positions[..., 1] ** 2,
        positions[..., 0] * positions[..., 1],
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
        sample_variance = np.maximum(centered_sum_squares, 0.0) / (ensemble_size - 1)
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
    """Convolve one spatial axis with zero extension in bounded memory.

    ``sliding_window_view`` plus ``tensordot`` expressed the same operation,
    but exposed a large strided virtual matrix to BLAS.  On the production
    stack that path was both memory-bandwidth heavy and about an order of
    magnitude slower than SciPy's direct compiled one-dimensional loop.
    Request float64 output explicitly because histogram inputs are uint32.
    """

    return convolve1d(
        values,
        kernel,
        axis=axis,
        output=np.float64,
        mode="constant",
        cval=0.0,
        origin=0,
    )


def _smooth_stack(
    values: NDArray[np.float64],
    kernel: NDArray[np.float64],
) -> NDArray[np.float64]:
    first = _convolve_axis(values, kernel, axis=-1)
    return _convolve_axis(first, kernel, axis=-2)


def _kde_kernel_and_support(
    settings: DensitySettings,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]
]:
    """Shared Gaussian KDE kernel and disk-support correction for one bandwidth."""

    grid_size = settings.grid_size
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
    return kernel, support, disk_mask, coordinates


def _smooth_support_corrected(
    values: np.ndarray,
    settings: DensitySettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gaussian-smooth a stack and correct for missing kernel support outside the disk.

    Does not renormalize; the caller decides whether/how to fix a total.
    """

    grid_size = settings.grid_size
    if values.shape[-2:] != (grid_size, grid_size):
        raise ValueError(
            f"density stack ends in {values.shape[-2:]}, expected "
            f"({grid_size}, {grid_size})"
        )
    kernel, support, disk_mask, coordinates = _kde_kernel_and_support(settings)
    smoothed = np.empty(values.shape, dtype=np.float64)
    flat_values = values.reshape((-1, grid_size, grid_size))
    flat_smoothed = smoothed.reshape((-1, grid_size, grid_size))

    chunk_size = 16
    for start in range(0, len(flat_values), chunk_size):
        stop = min(len(flat_values), start + chunk_size)
        chunk = _smooth_stack(flat_values[start:stop].astype(float), kernel)
        chunk /= support[None]
        chunk[:, ~disk_mask] = 0.0
        flat_smoothed[start:stop] = chunk
    return smoothed, coordinates


def smooth_density_stack(
    values: np.ndarray,
    settings: DensitySettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply the same boundary-corrected Gaussian observation KDE to a stack."""

    density, coordinates, _ = _normalized_smoothed(values, settings)
    return density, coordinates


def _normalized_smoothed(
    values: np.ndarray,
    settings: DensitySettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """As :func:`smooth_density_stack`, also returning the per-slice normalization."""

    smoothed, coordinates = _smooth_support_corrected(values, settings)
    cell_area = (2.0 / settings.grid_size) ** 2
    normalization = np.sum(smoothed, axis=(-2, -1)) * cell_area
    if not np.all(np.isfinite(normalization)) or np.any(normalization <= 0.0):
        raise ValueError("density stack cannot be normalized after KDE smoothing")
    density = smoothed / normalization[..., None, None]
    return density, coordinates, normalization


def smooth_weighted_stack(
    values: np.ndarray,
    settings: DensitySettings,
    normalization: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Smooth a signed (possibly negative) weighted-sum stack with a given normalization.

    Used for the polarity numerators ``sum cos(theta)`` and ``sum sin(theta)``,
    which must share the exact normalization applied to the paired density
    ``rho`` (from :func:`_normalized_smoothed`) so that ``p`` and ``rho`` stay
    consistently scaled: e.g. a fully polarized cell gives ``|p| == rho``.
    """

    smoothed, _ = _smooth_support_corrected(values, settings)
    return smoothed / normalization[..., None, None]


def _density_from_counts(
    counts: NDArray[np.uint32],
    ensemble_size: int,
    settings: DensitySettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    density, coordinates, normalization = _normalized_smoothed(counts, settings)

    # The normalization above uses counts only through a common multiplicative
    # factor, so M cancels. Keep this explicit check tied to the requested M.
    sample_totals = np.sum(counts, axis=(-2, -1))
    if not np.all(sample_totals == ensemble_size):
        raise RuntimeError("the density histogram lost ensemble samples")
    return density, coordinates, normalization


def compute_position_pdf(
    config: ProjectConfig,
    *,
    ensemble_size: int | None = None,
    show_progress: bool = False,
) -> PositionPDFResult:
    """Estimate dimensionless ``p(x,t)`` for the baseline and nine cases."""

    field = create_field(config.field)
    config, _ = resolve_numerics(config, field, PDF_CASES)
    density_config = _density_config(config)
    count = resolve_density_ensemble_size(
        density_config.density,
        requested=ensemble_size,
    )
    characteristic_length = inferred_characteristic_length(density_config, field)
    difference_bandwidth = resolve_difference_bandwidth(
        density_config.density,
        characteristic_length,
    )
    initial_gaussian_sigma = resolve_initial_gaussian_sigma(
        density_config.density,
        characteristic_length,
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
    polarity_sum_cos = np.zeros(
        (len(PDF_CASES), len(times), grid_size, grid_size),
        dtype=np.float64,
    )
    polarity_sum_sin = np.zeros_like(polarity_sum_cos)
    nematic_sum_xx = np.zeros_like(polarity_sum_cos)
    nematic_sum_xy = np.zeros_like(polarity_sum_cos)
    # The first independent half of the seeded ensemble is retained so the
    # paper audit can estimate the full-M Monte Carlo field error from the
    # difference between two disjoint, identically processed half samples.
    split_count = count // 2
    other_count = count - split_count
    split_counts = np.zeros_like(counts)
    split_nematic_sum_xx = np.zeros_like(polarity_sum_cos)
    split_nematic_sum_xy = np.zeros_like(polarity_sum_cos)
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

        use_numba = numba_available_for(
            field,
            create_turning_kernel(density_config.turning),
        )
        if density_config.performance.engine == "numba" and not use_numba:
            raise ValueError(
                "Numba density evolution requires a built-in chemical field"
            )

    bar = progress_bar(
        total=count,
        description="Cases 0-9 probability density",
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
                gaussian_positions=(
                    density_config.density.initial_position_distribution == "gaussian"
                ),
                gaussian_sigma=initial_gaussian_sigma,
            )
            _accumulate_histograms(
                counts,
                positions,
            )
            _accumulate_angular_moments(
                polarity_sum_cos,
                polarity_sum_sin,
                nematic_sum_xx,
                nematic_sum_xy,
                positions,
                headings,
            )
            split_stop = min(len(batch_seeds), max(0, split_count - start))
            if split_stop:
                _accumulate_histograms(split_counts, positions[:split_stop])
                _accumulate_nematic_moments(
                    split_nematic_sum_xx,
                    split_nematic_sum_xy,
                    positions[:split_stop],
                    headings[:split_stop],
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
        for particle_index, seed in enumerate(seeds):
            draws = generate_density_particle_draws(
                steps,
                seed,
                initial_position=density_config.initial.position,
                initial_heading=density_config.initial.heading,
                uniform_initial_position=(
                    density_config.density.initial_position_distribution == "uniform"
                ),
                gaussian_initial_position=(
                    density_config.density.initial_position_distribution == "gaussian"
                ),
                gaussian_sigma=initial_gaussian_sigma,
                isotropic_initial_heading=(
                    density_config.density.initial_heading_distribution == "isotropic"
                ),
            )
            particle_config = replace(
                density_config,
                initial=replace(
                    density_config.initial,
                    position=draws.initial_position,
                    heading=draws.initial_heading,
                ),
            )
            positions = np.empty((1, len(PDF_CASES), len(times), 2), dtype=float)
            headings = np.empty((1, len(PDF_CASES), len(times)), dtype=float)
            cumulative_tumbles = np.empty(
                (1, len(PDF_CASES), len(times)),
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
            _accumulate_angular_moments(
                polarity_sum_cos,
                polarity_sum_sin,
                nematic_sum_xx,
                nematic_sum_xy,
                positions,
                headings,
            )
            if particle_index < split_count:
                _accumulate_histograms(split_counts, positions)
                _accumulate_nematic_moments(
                    split_nematic_sum_xx,
                    split_nematic_sum_xy,
                    positions,
                    headings,
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

    density, coordinates, rho_normalization = _density_from_counts(
        counts,
        count,
        density_config.density,
    )
    polarity_x = smooth_weighted_stack(
        polarity_sum_cos,
        density_config.density,
        rho_normalization,
    )
    polarity_y = smooth_weighted_stack(
        polarity_sum_sin,
        density_config.density,
        rho_normalization,
    )
    nematic_xx = smooth_weighted_stack(
        nematic_sum_xx,
        density_config.density,
        rho_normalization,
    )
    nematic_xy = smooth_weighted_stack(
        nematic_sum_xy,
        density_config.density,
        rho_normalization,
    )
    split_density, split_coordinates, split_normalization = _density_from_counts(
        split_counts,
        split_count,
        density_config.density,
    )
    other_counts = counts - split_counts
    other_density, other_coordinates, other_normalization = _density_from_counts(
        other_counts,
        other_count,
        density_config.density,
    )
    if not (
        np.array_equal(coordinates, split_coordinates)
        and np.array_equal(coordinates, other_coordinates)
    ):
        raise RuntimeError("split-half Q estimators produced inconsistent grids")
    split_qxx = smooth_weighted_stack(
        split_nematic_sum_xx,
        density_config.density,
        split_normalization,
    )
    split_qxy = smooth_weighted_stack(
        split_nematic_sum_xy,
        density_config.density,
        split_normalization,
    )
    other_qxx = smooth_weighted_stack(
        nematic_sum_xx - split_nematic_sum_xx,
        density_config.density,
        other_normalization,
    )
    other_qxy = smooth_weighted_stack(
        nematic_sum_xy - split_nematic_sum_xy,
        density_config.density,
        other_normalization,
    )
    # If Var(Q_n)=Sigma/n, this factor turns Q_A-Q_B into an estimator of
    # the full-ensemble field error: sqrt(n_A n_B)/(n_A+n_B)=1/2 for equal halves.
    split_error_factor = np.sqrt(split_count * other_count) / count
    nematic_split_half_error_xx = split_error_factor * (split_qxx - other_qxx)
    nematic_split_half_error_xy = split_error_factor * (split_qxy - other_qxy)
    del split_density, other_density, split_qxx, split_qxy, other_qxx, other_qxy
    if np.isclose(
        difference_bandwidth,
        density_config.density.bandwidth_fraction,
        rtol=0.0,
        atol=1.0e-15,
    ):
        difference_density = density
        difference_polarity_x = polarity_x
        difference_polarity_y = polarity_y
        difference_nematic_xx = nematic_xx
        difference_nematic_xy = nematic_xy
    else:
        difference_settings = replace(
            density_config.density,
            bandwidth_fraction=difference_bandwidth,
        )
        (
            difference_density,
            difference_coordinates,
            difference_rho_normalization,
        ) = _density_from_counts(
            counts,
            count,
            difference_settings,
        )
        if not np.array_equal(coordinates, difference_coordinates):
            raise RuntimeError("density estimators produced different grids")
        # A second, smoother KDE pass for the polarity field, exactly
        # mirroring difference_density: the absolute-density bandwidth retains
        # more spatial detail, while paired differences benefit from stronger
        # suppression of their smaller residual Monte Carlo texture.
        difference_polarity_x = smooth_weighted_stack(
            polarity_sum_cos,
            difference_settings,
            difference_rho_normalization,
        )
        difference_polarity_y = smooth_weighted_stack(
            polarity_sum_sin,
            difference_settings,
            difference_rho_normalization,
        )
        difference_nematic_xx = smooth_weighted_stack(
            nematic_sum_xx,
            difference_settings,
            difference_rho_normalization,
        )
        difference_nematic_xy = smooth_weighted_stack(
            nematic_sum_xy,
            difference_settings,
            difference_rho_normalization,
        )
    # The complete initial ensemble law is prescribed exactly. Replace only
    # the reported t=0 estimator by that known density/polarity, leaving every
    # evolved slice as a direct finite-M microscopic measurement.
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    grid_points = np.stack((x_grid, y_grid), axis=-1)
    disk_mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    initial_grid = prescribed_initial_density_grid(
        density_config,
        field,
        grid_points,
        disk_mask,
        2.0 / grid_size,
    )
    initial_density, initial_coordinates = smooth_density_stack(
        initial_grid[None, None],
        density_config.density,
    )
    if not np.array_equal(coordinates, initial_coordinates):
        raise RuntimeError("initial-density estimator produced an inconsistent grid")
    density[:, 0] = initial_density[0, 0]

    if difference_density is density:
        initial_difference_density = initial_density
    else:
        initial_difference_density, difference_initial_coordinates = (
            smooth_density_stack(
                initial_grid[None, None],
                difference_settings,
            )
        )
        if not np.array_equal(coordinates, difference_initial_coordinates):
            raise RuntimeError(
                "difference initial-density estimator produced an inconsistent grid"
            )
        difference_density[:, 0] = initial_difference_density[0, 0]

    if density_config.density.initial_heading_distribution == "isotropic":
        heading_x = 0.0
        heading_y = 0.0
    else:
        heading_x = float(np.cos(density_config.initial.heading))
        heading_y = float(np.sin(density_config.initial.heading))
    polarity_x[:, 0] = density[:, 0] * heading_x
    polarity_y[:, 0] = density[:, 0] * heading_y
    difference_polarity_x[:, 0] = difference_density[:, 0] * heading_x
    difference_polarity_y[:, 0] = difference_density[:, 0] * heading_y
    # Preserve the prescribed initial angular moment exactly rather than
    # finite-M binning texture. Isotropic headings give Q=0; a fixed heading
    # gives Qxx=rho(cos^2(theta)-1/2), Qxy=rho sin(theta)cos(theta).
    initial_qxx_factor = heading_x**2 - 0.5
    initial_qxy_factor = heading_x * heading_y
    if density_config.density.initial_heading_distribution == "isotropic":
        initial_qxx_factor = 0.0
        initial_qxy_factor = 0.0
    nematic_xx[:, 0] = density[:, 0] * initial_qxx_factor
    nematic_xy[:, 0] = density[:, 0] * initial_qxy_factor
    difference_nematic_xx[:, 0] = difference_density[:, 0] * initial_qxx_factor
    difference_nematic_xy[:, 0] = difference_density[:, 0] * initial_qxy_factor
    nematic_split_half_error_xx[:, 0] = 0.0
    nematic_split_half_error_xy[:, 0] = 0.0
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
        polarity_x=polarity_x,
        polarity_y=polarity_y,
        difference_polarity_x=difference_polarity_x,
        difference_polarity_y=difference_polarity_y,
        nematic_xx=nematic_xx,
        nematic_xy=nematic_xy,
        difference_nematic_xx=difference_nematic_xx,
        difference_nematic_xy=difference_nematic_xy,
        nematic_split_half_error_xx=nematic_split_half_error_xx,
        nematic_split_half_error_xy=nematic_split_half_error_xy,
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
    empty_field = np.empty((0,) + result.density.shape[1:], dtype=np.float64)
    polarity_x = empty_field if result.polarity_x is None else result.polarity_x
    polarity_y = empty_field if result.polarity_y is None else result.polarity_y
    difference_polarity_x = (
        polarity_x
        if result.difference_polarity_x is None
        else result.difference_polarity_x
    )
    difference_polarity_y = (
        polarity_y
        if result.difference_polarity_y is None
        else result.difference_polarity_y
    )
    nematic_xx = empty_field if result.nematic_xx is None else result.nematic_xx
    nematic_xy = empty_field if result.nematic_xy is None else result.nematic_xy
    difference_nematic_xx = (
        nematic_xx
        if result.difference_nematic_xx is None
        else result.difference_nematic_xx
    )
    difference_nematic_xy = (
        nematic_xy
        if result.difference_nematic_xy is None
        else result.difference_nematic_xy
    )
    nematic_split_half_error_xx = (
        empty_field
        if result.nematic_split_half_error_xx is None
        else result.nematic_split_half_error_xx
    )
    nematic_split_half_error_xy = (
        empty_field
        if result.nematic_split_half_error_xy is None
        else result.nematic_split_half_error_xy
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
    temporary: Path | None = None
    try:
        # A production archive can be several gigabytes.  Writing beside the
        # destination and replacing it atomically prevents an interrupted run
        # from destroying the last complete, fingerprinted cache entry.
        with NamedTemporaryFile(
            mode="w+b", suffix=".npz", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
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
                microscopic_observable_paired_standard_errors=(
                    observable_standard_errors
                ),
                polarity_x=polarity_x,
                polarity_y=polarity_y,
                difference_polarity_x=difference_polarity_x,
                difference_polarity_y=difference_polarity_y,
                nematic_xx=nematic_xx,
                nematic_xy=nematic_xy,
                difference_nematic_xx=difference_nematic_xx,
                difference_nematic_xy=difference_nematic_xy,
                nematic_split_half_error_xx=nematic_split_half_error_xx,
                nematic_split_half_error_xy=nematic_split_half_error_xy,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def load_position_pdf(input_path: str | Path) -> PositionPDFResult:
    """Load a density result written by :func:`save_position_pdf`."""

    path = Path(input_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as saved:
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
            density=saved["density"].astype(np.float64, copy=False),
            difference_density=saved["difference_density"].astype(
                np.float64, copy=False
            ),
            case_numbers=tuple(
                int(case_number) for case_number in saved["case_numbers"]
            ),
            ensemble_size=int(saved["ensemble_size"]),
            bandwidth_fraction=float(saved["bandwidth_fraction"]),
            difference_bandwidth_fraction=float(saved["difference_bandwidth_fraction"]),
            config_source=str(saved["config_source"]),
            seed=int(saved["seed"]),
            dt=float(saved["dt"]),
            method=str(saved["method"]),
            microscopic_observables=observables,
            polarity_x=(
                saved["polarity_x"].astype(np.float64, copy=False)
                if "polarity_x" in saved.files and saved["polarity_x"].size
                else None
            ),
            polarity_y=(
                saved["polarity_y"].astype(np.float64, copy=False)
                if "polarity_y" in saved.files and saved["polarity_y"].size
                else None
            ),
            difference_polarity_x=(
                saved["difference_polarity_x"].astype(np.float64, copy=False)
                if "difference_polarity_x" in saved.files
                and saved["difference_polarity_x"].size
                else None
            ),
            difference_polarity_y=(
                saved["difference_polarity_y"].astype(np.float64, copy=False)
                if "difference_polarity_y" in saved.files
                and saved["difference_polarity_y"].size
                else None
            ),
            nematic_xx=(
                saved["nematic_xx"].astype(np.float64, copy=False)
                if "nematic_xx" in saved.files and saved["nematic_xx"].size
                else None
            ),
            nematic_xy=(
                saved["nematic_xy"].astype(np.float64, copy=False)
                if "nematic_xy" in saved.files and saved["nematic_xy"].size
                else None
            ),
            difference_nematic_xx=(
                saved["difference_nematic_xx"].astype(np.float64, copy=False)
                if "difference_nematic_xx" in saved.files
                and saved["difference_nematic_xx"].size
                else None
            ),
            difference_nematic_xy=(
                saved["difference_nematic_xy"].astype(np.float64, copy=False)
                if "difference_nematic_xy" in saved.files
                and saved["difference_nematic_xy"].size
                else None
            ),
            nematic_split_half_error_xx=(
                saved["nematic_split_half_error_xx"].astype(np.float64, copy=False)
                if "nematic_split_half_error_xx" in saved.files
                and saved["nematic_split_half_error_xx"].size
                else None
            ),
            nematic_split_half_error_xy=(
                saved["nematic_split_half_error_xy"].astype(np.float64, copy=False)
                if "nematic_split_half_error_xy" in saved.files
                and saved["nematic_split_half_error_xy"].size
                else None
            ),
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
                str(saved["config_sha256"]) if "config_sha256" in saved.files else ""
            ),
            runtime_versions=(
                str(saved["runtime_versions"])
                if "runtime_versions" in saved.files
                else ""
            ),
        )
