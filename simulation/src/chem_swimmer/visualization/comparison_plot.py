"""Static contour diagnostics for cross-method initial states and time frames."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from ..config import ProjectConfig
from ..density import PositionPDFResult

METHOD_ORDER = ("micro", "moment", "macro", "analytic")
METHOD_LABELS = {
    "micro": "Microscopic",
    "moment": "Polar closure Q=0 (FV)",
    "macro": "Macroscopic dynamic p, Q=0 (FV)",
    "analytic": "Dynamic p Fourier/Green (plane)",
}


def _ordered_results(
    results: dict[str, PositionPDFResult],
) -> list[tuple[str, PositionPDFResult]]:
    if len(results) < 2:
        raise ValueError("cross-method contour plots require at least two methods")
    ordered = [(name, results[name]) for name in METHOD_ORDER if name in results]
    ordered.extend(
        (name, result) for name, result in results.items() if name not in METHOD_ORDER
    )
    case_numbers = ordered[0][1].case_numbers
    for name, result in ordered:
        if result.case_numbers != case_numbers:
            raise ValueError(f"{name} result uses inconsistent case numbers")
    return ordered


def _disk_mesh(
    coordinates: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    return x_grid, y_grid, x_grid * x_grid + y_grid * y_grid <= 1.0


def _masked(
    values: NDArray[np.float64], mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    return np.where(mask, values, np.nan)


def _nearest_time_index(result: PositionPDFResult, target_time: float) -> int:
    return int(np.argmin(np.abs(result.times - target_time)))


def _on_reference_grid(
    values: NDArray[np.float64],
    source_coordinates: NDArray[np.float64],
    reference_coordinates: NDArray[np.float64],
) -> NDArray[np.float64]:
    if np.array_equal(source_coordinates, reference_coordinates):
        return values.astype(float, copy=False)
    x_reference, y_reference = np.meshgrid(reference_coordinates, reference_coordinates)
    points = np.stack((y_reference.ravel(), x_reference.ravel()), axis=-1)
    interpolator = RegularGridInterpolator(
        (source_coordinates, source_coordinates),
        values,
        bounds_error=False,
        fill_value=np.nan,
    )
    return interpolator(points).reshape(x_reference.shape)


def _agreement_metrics(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> tuple[float, float, float]:
    valid = mask & np.isfinite(reference) & np.isfinite(candidate)
    reference_values = reference[valid]
    candidate_values = candidate[valid]
    residual = candidate_values - reference_values
    reference_norm = float(np.linalg.norm(reference_values))
    candidate_norm = float(np.linalg.norm(candidate_values))
    residual_norm = float(np.linalg.norm(residual))
    relative_l2 = residual_norm / max(reference_norm, np.finfo(float).eps)
    denominator = reference_norm * candidate_norm
    cosine = (
        float(reference_values @ candidate_values) / denominator
        if denominator > np.finfo(float).eps
        else float("nan")
    )
    return relative_l2, cosine, float(np.max(np.abs(residual), initial=0.0))


def _decorate_disk_axis(axis, title: str) -> None:
    axis.add_patch(Circle((0.0, 0.0), 1.0, fill=False, color="0.15", linewidth=0.8))
    axis.set(xlim=(-1.03, 1.03), ylim=(-1.03, 1.03), aspect="equal")
    axis.set_title(title, fontsize=9)
    axis.set_xticks((-1.0, 0.0, 1.0))
    axis.set_yticks((-1.0, 0.0, 1.0))


def render_initial_condition_contours(
    results: dict[str, PositionPDFResult],
    config: ProjectConfig,
    output_path: str | Path,
) -> Path:
    """Plot the actual reported ``rho_in`` and ``p_in`` for every method."""

    ordered = _ordered_results(results)
    for name, result in ordered:
        if result.polarity_x is None or result.polarity_y is None:
            raise ValueError(f"{name} result has no polarity field")

    density_max = max(float(np.nanmax(result.density[:, 0])) for _, result in ordered)
    polarity_max = max(
        float(np.nanmax(np.hypot(result.polarity_x[:, 0], result.polarity_y[:, 0])))
        for _, result in ordered
    )
    zero_initial_polarity = polarity_max <= 1.0e-12
    density_norm = Normalize(0.0, max(density_max, np.finfo(float).eps))
    polarity_norm = Normalize(0.0, 1.0 if zero_initial_polarity else polarity_max)

    reference_name = "micro" if "micro" in results else ordered[0][0]
    reference = results[reference_name]
    reference_coordinates = reference.coordinates
    _, _, reference_mask = _disk_mesh(reference_coordinates)
    reference_density = reference.density[0, 0].astype(float)
    reference_polarity_x = reference.polarity_x[0, 0].astype(float)
    reference_polarity_y = reference.polarity_y[0, 0].astype(float)

    figure = Figure(figsize=(3.5 * len(ordered), 6.8))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, len(ordered), squeeze=False)
    figure.subplots_adjust(
        left=0.08,
        right=0.90,
        bottom=0.07,
        top=0.84,
        hspace=0.42,
        wspace=0.24,
    )
    density_contours = None
    polarity_contours = None
    for column, (name, result) in enumerate(ordered):
        coordinates = result.coordinates
        x_grid, y_grid, mask = _disk_mesh(coordinates)
        density = result.density[0, 0].astype(float)
        polarity_x = result.polarity_x[0, 0].astype(float)
        polarity_y = result.polarity_y[0, 0].astype(float)
        magnitude = np.hypot(polarity_x, polarity_y)

        density_contours = axes[0, column].contourf(
            x_grid,
            y_grid,
            _masked(density, mask),
            levels=np.linspace(density_norm.vmin, density_norm.vmax, 25),
            cmap="viridis",
            norm=density_norm,
        )
        density_case_spread = float(
            np.max(np.abs(result.density[:, 0] - result.density[0:1, 0]))
        )
        density_on_reference = _on_reference_grid(
            density, coordinates, reference_coordinates
        )
        density_relative_l2, _, _ = _agreement_metrics(
            reference_density, density_on_reference, reference_mask
        )
        _decorate_disk_axis(
            axes[0, column],
            f"{METHOD_LABELS.get(name, name)}\n"
            f"C spread {density_case_spread:.1e}; L2 {density_relative_l2:.2g}",
        )

        polarity_contours = axes[1, column].contourf(
            x_grid,
            y_grid,
            _masked(magnitude, mask),
            levels=np.linspace(polarity_norm.vmin, polarity_norm.vmax, 25),
            cmap="magma",
            norm=polarity_norm,
        )
        stride = max(1, len(coordinates) // 15)
        sampled_mask = mask[::stride, ::stride]
        sampled_x = np.where(sampled_mask, polarity_x[::stride, ::stride], np.nan)
        sampled_y = np.where(sampled_mask, polarity_y[::stride, ::stride], np.nan)
        if not zero_initial_polarity:
            axes[1, column].quiver(
                x_grid[::stride, ::stride],
                y_grid[::stride, ::stride],
                sampled_x,
                sampled_y,
                color="white",
                angles="xy",
                scale_units="xy",
                scale=max(1.0, 8.0 * polarity_norm.vmax),
                width=0.004,
            )
        polarity_case_spread = max(
            float(np.max(np.abs(result.polarity_x[:, 0] - result.polarity_x[0:1, 0]))),
            float(np.max(np.abs(result.polarity_y[:, 0] - result.polarity_y[0:1, 0]))),
        )
        polarity_x_on_reference = _on_reference_grid(
            polarity_x, coordinates, reference_coordinates
        )
        polarity_y_on_reference = _on_reference_grid(
            polarity_y, coordinates, reference_coordinates
        )
        polarity_residual = np.hypot(
            polarity_x_on_reference - reference_polarity_x,
            polarity_y_on_reference - reference_polarity_y,
        )
        polarity_error = float(np.nanmax(polarity_residual[reference_mask]))
        _decorate_disk_axis(
            axes[1, column],
            r"$|\mathbf{p}_{in}|$ and direction"
            + f"\nC spread {polarity_case_spread:.1e}; error {polarity_error:.1e}",
        )

    axes[0, 0].set_ylabel(r"Initial density $\rho_{in}$")
    axes[1, 0].set_ylabel(r"Initial polarity $\mathbf{p}_{in}$")
    if density_contours is not None:
        figure.colorbar(
            density_contours,
            ax=list(axes[0]),
            fraction=0.025,
            pad=0.015,
            label=r"$\rho_{in}$",
        )
    if polarity_contours is not None:
        polarity_colorbar = figure.colorbar(
            polarity_contours,
            ax=list(axes[1]),
            fraction=0.025,
            pad=0.015,
            label=r"$|\mathbf{p}_{in}|$",
        )
        if zero_initial_polarity:
            polarity_colorbar.set_ticks([0.0])
    figure.suptitle(
        "Initial conditions: "
        f"{config.density.initial_position_distribution} positions, "
        f"{config.density.initial_heading_distribution} headings",
        fontsize=12,
        y=0.97,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    figure.clear()
    return output


def _display_density(
    result: PositionPDFResult,
    case_index: int,
    time_index: int,
) -> NDArray[np.float64]:
    if case_index == 0:
        return result.density[0, time_index].astype(float)
    source = (
        result.difference_density
        if result.difference_density is not None
        else result.density
    )
    return source[case_index, time_index].astype(float) - source[0, time_index].astype(
        float
    )


def render_density_frames_vs_time(
    results: dict[str, PositionPDFResult],
    config: ProjectConfig,
    output_directory: str | Path,
    metrics_path: str | Path,
    *,
    frame_count: int = 5,
) -> tuple[tuple[Path, ...], Path]:
    """Write one method-by-time contour sheet per case and quantitative errors."""

    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    ordered = _ordered_results(results)
    reference_name = "micro" if "micro" in results else ordered[0][0]
    reference = results[reference_name]
    start = max(float(result.times[0]) for _, result in ordered)
    stop = min(float(result.times[-1]) for _, result in ordered)
    target_times = np.linspace(start, stop, frame_count)
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_output = Path(metrics_path).expanduser().resolve()
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with metrics_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "quantity",
                "method",
                "target_time",
                "actual_time",
                "relative_l2_vs_micro",
                "cosine_similarity_vs_micro",
                "max_abs_error_vs_micro",
            ]
        )
        for case_index, case_number in enumerate(reference.case_numbers):
            slices: dict[tuple[str, int], NDArray[np.float64]] = {}
            finite_values: list[NDArray[np.float64]] = []
            for method, result in ordered:
                for time_column, target_time in enumerate(target_times):
                    time_index = _nearest_time_index(result, float(target_time))
                    values = _display_density(result, case_index, time_index)
                    slices[(method, time_column)] = values
                    _, _, mask = _disk_mesh(result.coordinates)
                    finite_values.append(values[mask])
            combined = np.concatenate(finite_values)
            if case_index == 0:
                limit = max(
                    float(
                        np.percentile(
                            combined, config.density.difference_case_percentile
                        )
                    ),
                    np.finfo(float).eps,
                )
                norm = Normalize(0.0, limit)
                cmap = "viridis"
                quantity = "absolute_density"
            else:
                limit = max(
                    float(
                        np.percentile(
                            np.abs(combined), config.density.difference_case_percentile
                        )
                    ),
                    np.finfo(float).eps,
                )
                norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
                cmap = "coolwarm"
                quantity = "case_minus_baseline_density"

            figure = Figure(
                figsize=(2.45 * frame_count, 2.35 * len(ordered)),
                constrained_layout=True,
            )
            FigureCanvasAgg(figure)
            axes = figure.subplots(len(ordered), frame_count, squeeze=False)
            contour = None
            reference_coordinates = reference.coordinates
            _, _, reference_mask = _disk_mesh(reference_coordinates)
            for row, (method, result) in enumerate(ordered):
                x_grid, y_grid, mask = _disk_mesh(result.coordinates)
                for column, target_time in enumerate(target_times):
                    values = slices[(method, column)]
                    contour = axes[row, column].contourf(
                        x_grid,
                        y_grid,
                        _masked(values, mask),
                        levels=np.linspace(norm.vmin, norm.vmax, 25),
                        cmap=cmap,
                        norm=norm,
                        extend="both" if case_index else "max",
                    )
                    reference_index = _nearest_time_index(reference, float(target_time))
                    reference_values = _display_density(
                        reference, case_index, reference_index
                    )
                    candidate_on_reference = _on_reference_grid(
                        values, result.coordinates, reference_coordinates
                    )
                    relative_l2, cosine, max_abs = _agreement_metrics(
                        reference_values, candidate_on_reference, reference_mask
                    )
                    actual_index = _nearest_time_index(result, float(target_time))
                    writer.writerow(
                        [
                            case_number,
                            quantity,
                            method,
                            f"{target_time:.12g}",
                            f"{result.times[actual_index]:.12g}",
                            f"{relative_l2:.12g}",
                            f"{cosine:.12g}",
                            f"{max_abs:.12g}",
                        ]
                    )
                    metric = (
                        "reference"
                        if method == reference_name
                        else f"rel L2={relative_l2:.2g}, cos={cosine:.3f}"
                    )
                    title = (
                        f"t={result.times[actual_index]:.4g}\n{metric}"
                        if row == 0
                        else metric
                    )
                    _decorate_disk_axis(axes[row, column], title)
                    if column == 0:
                        axes[row, column].set_ylabel(METHOD_LABELS.get(method, method))
            if contour is not None:
                label = (
                    r"$\rho$"
                    if case_index == 0
                    else rf"$\rho_{{{case_number}}}-\rho_0$"
                )
                figure.colorbar(
                    contour, ax=list(axes.flat), fraction=0.018, pad=0.01, label=label
                )
            figure.suptitle(
                (
                    "Baseline density frames versus time"
                    if case_index == 0
                    else f"Case {case_number} response frames versus time"
                ),
                fontsize=14,
            )
            output = output_root / f"density_frames_case_{case_number:02d}.png"
            figure.savefig(output, dpi=180, facecolor="white")
            figure.clear()
            written.append(output)
    return tuple(written), metrics_output


def _display_polarity(
    result: PositionPDFResult,
    case_index: int,
    time_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    source_x = (
        result.difference_polarity_x
        if result.difference_polarity_x is not None
        else result.polarity_x
    )
    source_y = (
        result.difference_polarity_y
        if result.difference_polarity_y is not None
        else result.polarity_y
    )
    if source_x is None or source_y is None:
        raise ValueError(f"{result.method} result has no polarity field")
    if case_index == 0:
        return (
            result.polarity_x[0, time_index].astype(float),
            result.polarity_y[0, time_index].astype(float),
        )
    return (
        source_x[case_index, time_index].astype(float)
        - source_x[0, time_index].astype(float),
        source_y[case_index, time_index].astype(float)
        - source_y[0, time_index].astype(float),
    )


def _vector_agreement_metrics(
    reference_x: NDArray[np.float64],
    reference_y: NDArray[np.float64],
    candidate_x: NDArray[np.float64],
    candidate_y: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> tuple[float, float, float]:
    valid = (
        mask
        & np.isfinite(reference_x)
        & np.isfinite(reference_y)
        & np.isfinite(candidate_x)
        & np.isfinite(candidate_y)
    )
    reference = np.concatenate((reference_x[valid], reference_y[valid]))
    candidate = np.concatenate((candidate_x[valid], candidate_y[valid]))
    residual = candidate - reference
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    relative_l2 = float(np.linalg.norm(residual)) / max(
        reference_norm, np.finfo(float).eps
    )
    denominator = reference_norm * candidate_norm
    cosine = (
        float(reference @ candidate) / denominator
        if denominator > np.finfo(float).eps
        else float("nan")
    )
    pointwise_error = np.hypot(candidate_x - reference_x, candidate_y - reference_y)
    return relative_l2, cosine, float(np.nanmax(pointwise_error[valid], initial=0.0))


def render_polarity_frames_vs_time(
    results: dict[str, PositionPDFResult],
    config: ProjectConfig,
    output_directory: str | Path,
    metrics_path: str | Path,
    *,
    frame_count: int = 5,
) -> tuple[tuple[Path, ...], Path]:
    """Write polarity-vector method-by-time sheets and append field errors."""

    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    ordered = _ordered_results(results)
    for name, result in ordered:
        if result.polarity_x is None or result.polarity_y is None:
            raise ValueError(f"{name} result has no polarity field")
    reference_name = "micro" if "micro" in results else ordered[0][0]
    reference = results[reference_name]
    start = max(float(result.times[0]) for _, result in ordered)
    stop = min(float(result.times[-1]) for _, result in ordered)
    target_times = np.linspace(start, stop, frame_count)
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_output = Path(metrics_path).expanduser().resolve()
    if not metrics_output.is_file():
        raise ValueError(
            "density frame metrics must be written before polarity metrics"
        )
    written: list[Path] = []

    with metrics_output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for case_index, case_number in enumerate(reference.case_numbers):
            slices: dict[
                tuple[str, int], tuple[NDArray[np.float64], NDArray[np.float64]]
            ] = {}
            magnitudes: list[NDArray[np.float64]] = []
            for method, result in ordered:
                for time_column, target_time in enumerate(target_times):
                    time_index = _nearest_time_index(result, float(target_time))
                    vector_x, vector_y = _display_polarity(
                        result, case_index, time_index
                    )
                    slices[(method, time_column)] = (vector_x, vector_y)
                    _, _, mask = _disk_mesh(result.coordinates)
                    magnitudes.append(np.hypot(vector_x, vector_y)[mask])
            combined = np.concatenate(magnitudes)
            limit = max(
                float(
                    np.percentile(combined, config.density.difference_case_percentile)
                ),
                np.finfo(float).eps,
            )
            norm = Normalize(0.0, limit)
            quantity = (
                "absolute_polarity"
                if case_index == 0
                else "case_minus_baseline_polarity"
            )
            figure = Figure(
                figsize=(2.45 * frame_count, 2.35 * len(ordered)),
                constrained_layout=True,
            )
            FigureCanvasAgg(figure)
            axes = figure.subplots(len(ordered), frame_count, squeeze=False)
            contour = None
            reference_coordinates = reference.coordinates
            _, _, reference_mask = _disk_mesh(reference_coordinates)
            for row, (method, result) in enumerate(ordered):
                x_grid, y_grid, mask = _disk_mesh(result.coordinates)
                stride = max(1, len(result.coordinates) // 15)
                for column, target_time in enumerate(target_times):
                    vector_x, vector_y = slices[(method, column)]
                    magnitude = np.hypot(vector_x, vector_y)
                    contour = axes[row, column].contourf(
                        x_grid,
                        y_grid,
                        _masked(magnitude, mask),
                        levels=np.linspace(norm.vmin, norm.vmax, 25),
                        cmap="magma",
                        norm=norm,
                        extend="max",
                    )
                    if limit > 1.0e-14:
                        sampled_mask = mask[::stride, ::stride]
                        sampled_x = np.where(
                            sampled_mask, vector_x[::stride, ::stride], np.nan
                        )
                        sampled_y = np.where(
                            sampled_mask, vector_y[::stride, ::stride], np.nan
                        )
                        axes[row, column].quiver(
                            x_grid[::stride, ::stride],
                            y_grid[::stride, ::stride],
                            sampled_x,
                            sampled_y,
                            color="white",
                            angles="xy",
                            scale_units="xy",
                            scale=max(1.0e-12, 10.0 * limit),
                            width=0.004,
                        )

                    reference_index = _nearest_time_index(reference, float(target_time))
                    reference_x, reference_y = _display_polarity(
                        reference, case_index, reference_index
                    )
                    candidate_x = _on_reference_grid(
                        vector_x, result.coordinates, reference_coordinates
                    )
                    candidate_y = _on_reference_grid(
                        vector_y, result.coordinates, reference_coordinates
                    )
                    relative_l2, cosine, max_abs = _vector_agreement_metrics(
                        reference_x,
                        reference_y,
                        candidate_x,
                        candidate_y,
                        reference_mask,
                    )
                    actual_index = _nearest_time_index(result, float(target_time))
                    writer.writerow(
                        [
                            case_number,
                            quantity,
                            method,
                            f"{target_time:.12g}",
                            f"{result.times[actual_index]:.12g}",
                            f"{relative_l2:.12g}",
                            f"{cosine:.12g}",
                            f"{max_abs:.12g}",
                        ]
                    )
                    metric = (
                        "reference"
                        if method == reference_name
                        else f"rel L2={relative_l2:.2g}, cos={cosine:.3f}"
                    )
                    title = (
                        f"t={result.times[actual_index]:.4g}\n{metric}"
                        if row == 0
                        else metric
                    )
                    _decorate_disk_axis(axes[row, column], title)
                    if column == 0:
                        axes[row, column].set_ylabel(METHOD_LABELS.get(method, method))
            if contour is not None:
                label = (
                    r"$|\mathbf{p}|$"
                    if case_index == 0
                    else rf"$|\mathbf{{p}}_{{{case_number}}}-\mathbf{{p}}_0|$"
                )
                figure.colorbar(
                    contour, ax=list(axes.flat), fraction=0.018, pad=0.01, label=label
                )
            figure.suptitle(
                (
                    "Baseline polarity frames versus time"
                    if case_index == 0
                    else f"Case {case_number} polarity-response frames versus time"
                ),
                fontsize=14,
            )
            output = output_root / f"polarity_frames_case_{case_number:02d}.png"
            figure.savefig(output, dpi=180, facecolor="white")
            figure.clear()
            written.append(output)
    return tuple(written), metrics_output
