"""Macroscopic observables and identifiability diagnostics from density movies."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..config import ProjectConfig
from ..density import PDF_CASES, PositionPDFResult, smooth_density_stack
from ..fields import ChemicalField
from ..kinetics import macroscopic_coefficients
from ..numerics import inferred_characteristic_length
from ..turning import create_turning_kernel

CASE_LABELS = tuple(f"C{case_number}" for case_number in PDF_CASES)
FIRST_NEUMANN_DISK_WAVENUMBER = 1.8411837813406593


def render_microscopic_mechanism_report(
    micro: PositionPDFResult,
    config: ProjectConfig,
    figure_path: str | Path,
    summary_path: str | Path,
    timeseries_path: str | Path,
) -> tuple[Path, Path, Path]:
    """Write paired angular and displacement diagnostics for mechanism inference."""

    observables = micro.microscopic_observables
    if observables is None:
        raise ValueError("the microscopic result has no angular observables")
    if observables.values.shape[1:] != (
        len(micro.case_numbers),
        len(micro.times),
    ):
        raise ValueError("microscopic observable dimensions are inconsistent")
    indices = {name: index for index, name in enumerate(observables.names)}
    required = {
        "mean_eta",
        "zeta",
        "nematic_zeta",
        "heading_correlation_2",
        "cumulative_tumbles",
        "mean_squared_displacement",
    }
    missing = required.difference(indices)
    if missing:
        raise ValueError(
            "microscopic result is missing observables: " + ", ".join(sorted(missing))
        )

    timeseries = Path(timeseries_path).expanduser().resolve()
    timeseries.parent.mkdir(parents=True, exist_ok=True)
    with timeseries.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["case", "time"]
        for name in observables.names:
            header.extend((name, f"{name}_paired_standard_error"))
        writer.writerow(header)
        for case_index, case_number in enumerate(micro.case_numbers):
            for time_index, time in enumerate(micro.times):
                row: list[int | str] = [case_number, f"{time:.12g}"]
                for observable_index in range(len(observables.names)):
                    row.extend(
                        (
                            f"{observables.values[observable_index, case_index, time_index]:.12g}",
                            f"{observables.paired_standard_errors[observable_index, case_index, time_index]:.12g}",
                        )
                    )
                writer.writerow(row)

    eta_index = indices["mean_eta"]
    eta = observables.values[eta_index]
    eta_difference = eta - eta[0:1]
    eta_error = observables.paired_standard_errors[eta_index]
    final_case3_scale = abs(eta_difference[3, -1])
    familywise_z_threshold = NormalDist().inv_cdf(
        1.0 - 0.05 / (2.0 * (len(micro.case_numbers) - 1))
    )
    P_turn = create_turning_kernel(config.turning).P_turn
    summary = Path(summary_path).expanduser().resolve()
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "case",
                "kappa",
                "linear_density_prediction",
                "final_delta_mean_eta",
                "final_paired_standard_error",
                "final_z_score",
                "familywise_95_z_threshold",
                "absolute_effect_relative_to_case3",
                "resolved_at_familywise_95_percent",
            )
        )
        for case_index, case_number in enumerate(micro.case_numbers):
            mobility = macroscopic_coefficients(
                config.kinetics,
                config.responses.only_case(case_number),
                P_turn,
            )[2]
            effect = eta_difference[case_index, -1]
            error = eta_error[case_index, -1]
            z_score = effect / error if error > 0.0 else float("nan")
            relative = (
                abs(effect) / final_case3_scale
                if final_case3_scale > 0.0
                else float("nan")
            )
            writer.writerow(
                (
                    case_number,
                    f"{mobility:.12g}",
                    "zero bulk mobility at retained order"
                    if abs(mobility) <= 1.0e-14
                    else "nonzero bulk mobility at retained order",
                    f"{effect:.12g}",
                    f"{error:.12g}",
                    f"{z_score:.12g}",
                    f"{familywise_z_threshold:.12g}",
                    f"{relative:.12g}",
                    bool(
                        np.isfinite(z_score) and abs(z_score) >= familywise_z_threshold
                    ),
                )
            )

    figure_output = Path(figure_path).expanduser().resolve()
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(14.0, 8.5), dpi=150, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    colors = {
        1: colormaps["tab10"](0),
        2: colormaps["tab10"](1),
        3: colormaps["tab10"](2),
        4: colormaps["tab10"](3),
        5: colormaps["tab10"](4),
        6: colormaps["tab10"](5),
        8: colormaps["tab10"](7),
    }

    def plot_paired(
        axis,
        name: str,
        cases: tuple[int, ...],
        *,
        title: str,
        ylabel: str,
    ) -> None:
        observable_index = indices[name]
        values = observables.values[observable_index]
        differences = values - values[0:1]
        standard_errors = observables.paired_standard_errors[observable_index]
        for case_number in cases:
            color = colors[case_number]
            difference = differences[case_number]
            uncertainty = 1.96 * standard_errors[case_number]
            axis.plot(
                micro.times,
                difference,
                color=color,
                linewidth=1.5,
                label=f"C{case_number}",
            )
            axis.fill_between(
                micro.times,
                difference - uncertainty,
                difference + uncertainty,
                color=color,
                alpha=0.13,
                linewidth=0.0,
            )
        axis.axhline(0.0, color="0.55", linewidth=0.8)
        axis.set(title=title, xlabel="t", ylabel=ylabel)
        axis.legend(frameon=False)
        axis.grid(alpha=0.22, linewidth=0.6)

    plot_paired(
        axes[0, 0],
        "mean_eta",
        (2, 3, 4, 5),
        title="Signed response: bulk-closure-zero cases versus C3",
        ylabel=r"$\Delta\langle\eta\rangle$",
    )
    plot_paired(
        axes[0, 1],
        "zeta",
        (1, 3),
        title="Local polarization distinguishes C1 from C3",
        ylabel=r"$\Delta\langle\zeta\rangle$",
    )
    plot_paired(
        axes[0, 2],
        "mean_squared_displacement",
        (1, 3),
        title="Displacement variance: C1 versus C3",
        ylabel=r"$\Delta\langle|\mathbf{x}(t)-\mathbf{x}(0)|^2\rangle$",
    )
    plot_paired(
        axes[1, 0],
        "nematic_zeta",
        (6, 8),
        title="Nematic-alignment test: C6 versus C8",
        ylabel=r"$\Delta\langle|\nabla\eta|\cos 2(\theta-\phi_\eta)\rangle$",
    )
    plot_paired(
        axes[1, 1],
        "heading_correlation_2",
        (6, 8),
        title="Second-harmonic memory: C6 versus C8",
        ylabel=r"$\Delta\langle\cos 2[\theta(t)-\theta(0)]\rangle$",
    )
    plot_paired(
        axes[1, 2],
        "cumulative_tumbles",
        (6, 8),
        title="Cumulative tumbles: C6 versus C8",
        ylabel=r"$\Delta\langle N_{\mathrm{tumble}}(t)\rangle$",
    )
    figure.suptitle(
        "Microscopic mechanism observables (shading: paired 95% Monte Carlo CI)",
        fontsize=15,
    )
    figure.savefig(figure_output, facecolor="white")
    figure.clear()
    return figure_output, summary, timeseries


def _series(
    result: PositionPDFResult,
    field: ChemicalField,
) -> dict[str, np.ndarray]:
    coordinates = result.coordinates
    spacing = 2.0 / len(coordinates)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    points = np.stack((x_grid, y_grid), axis=-1)
    eta = np.asarray(field.value(points), dtype=float)
    density = result.density.astype(float)
    difference = density - density[0:1]
    area = spacing**2
    field_exposure = (
        np.sum(
            difference * eta[None, None],
            axis=(-2, -1),
        )
        * area
    )
    cumulative_exposure = np.zeros_like(field_exposure)
    cumulative_exposure[:, 1:] = np.cumsum(
        0.5
        * (field_exposure[:, 1:] + field_exposure[:, :-1])
        * np.diff(result.times)[None],
        axis=1,
    )
    return {
        "field_exposure": field_exposure,
        "cumulative_field_exposure": cumulative_exposure,
        "centroid_x": np.sum(difference * x_grid[None, None], axis=(-2, -1)) * area,
        "centroid_y": np.sum(difference * y_grid[None, None], axis=(-2, -1)) * area,
        "total_variation": 0.5 * np.sum(np.abs(difference), axis=(-2, -1)) * area,
        "l2": np.sqrt(np.sum(difference**2, axis=(-2, -1)) * area),
    }


def _equilibrium_observables(
    config: ProjectConfig,
    field: ChemicalField,
    result: PositionPDFResult,
    effective_diffusivity: float,
    mobilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = result.coordinates
    grid_size = len(coordinates)
    spacing = 2.0 / grid_size
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    points = np.stack((x_grid, y_grid), axis=-1)
    eta = np.asarray(field.value(points), dtype=float)
    mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    raw = np.zeros((len(PDF_CASES), grid_size, grid_size), dtype=float)
    for case_index, mobility in enumerate(mobilities):
        exponent = mobility * eta[mask] / effective_diffusivity
        exponent -= np.max(exponent)
        raw[case_index, mask] = np.exp(exponent)
        raw[case_index] /= np.sum(raw[case_index]) * spacing**2
    observed, _ = smooth_density_stack(
        raw,
        replace(
            config.density,
            grid_size=grid_size,
            bandwidth_fraction=result.bandwidth_fraction,
        ),
    )
    observed = observed.astype(float)
    difference = observed - observed[0:1]
    area = spacing**2
    exposure = np.sum(difference * eta[None], axis=(-2, -1)) * area
    centroid_x = np.sum(difference * x_grid[None], axis=(-2, -1)) * area
    centroid_y = np.sum(difference * y_grid[None], axis=(-2, -1)) * area
    total_variation = 0.5 * np.sum(np.abs(difference), axis=(-2, -1)) * area
    return exposure, centroid_x, centroid_y, total_variation


def _crossing_time(times: np.ndarray, ratio: np.ndarray, level: float) -> float:
    indices = np.flatnonzero(ratio >= level)
    if not len(indices):
        return float("nan")
    index = int(indices[0])
    if index == 0:
        return float(times[0])
    left_value = ratio[index - 1]
    right_value = ratio[index]
    if right_value == left_value:
        return float(times[index])
    fraction = (level - left_value) / (right_value - left_value)
    return float(times[index - 1] + fraction * (times[index] - times[index - 1]))


def _effective_relaxation_time(times: np.ndarray, ratio: np.ndarray) -> float:
    selected = np.isfinite(ratio) & (ratio >= 0.1) & (ratio <= 0.9) & (times > 0.0)
    if np.count_nonzero(selected) < 3:
        return float("nan")
    selected_times = times[selected]
    transformed = -np.log1p(-np.clip(ratio[selected], 0.0, 1.0 - 1.0e-12))
    slope = float(
        np.dot(selected_times, transformed) / np.dot(selected_times, selected_times)
    )
    return float("nan") if slope <= 0.0 else 1.0 / slope


def _write_timeseries(
    path: Path,
    results: tuple[PositionPDFResult, PositionPDFResult],
    series: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "method",
                "case",
                "time",
                "delta_mean_eta",
                "cumulative_delta_mean_eta",
                "delta_centroid_x",
                "delta_centroid_y",
                "total_variation_from_case0",
                "l2_from_case0",
            )
        )
        for result, values in zip(results, series, strict=True):
            for case_index, case_number in enumerate(result.case_numbers):
                for time_index, time in enumerate(result.times):
                    writer.writerow(
                        (
                            result.method,
                            case_number,
                            f"{time:.12g}",
                            f"{values['field_exposure'][case_index, time_index]:.12g}",
                            f"{values['cumulative_field_exposure'][case_index, time_index]:.12g}",
                            f"{values['centroid_x'][case_index, time_index]:.12g}",
                            f"{values['centroid_y'][case_index, time_index]:.12g}",
                            f"{values['total_variation'][case_index, time_index]:.12g}",
                            f"{values['l2'][case_index, time_index]:.12g}",
                        )
                    )


_METHOD_LINESTYLES = {"micro": "-", "macro": "--", "analytic": ":"}
_METHOD_LINEWIDTHS = {"micro": 1.5, "moment": 1.35, "macro": 1.2, "analytic": 1.6}


def render_density_inference_report(
    results: dict[str, PositionPDFResult],
    field: ChemicalField,
    config: ProjectConfig,
    figure_path: str | Path,
    summary_path: str | Path,
    timeseries_path: str | Path,
) -> tuple[Path, Path, Path]:
    """Write shared-scale observables, relaxation times, and CSV data.

    ``results`` maps a method name (``"micro"``, ``"moment"``, ``"macro"``, ``"analytic"``,
    ...) to its :class:`~chem_swimmer.density.PositionPDFResult`; any subset
    of two or more methods is accepted; all must share the same case numbers.
    Equilibrium and relaxation-time diagnostics are computed from whichever
    of ``"macro"``/``"analytic"`` is present (both reach a smooth,
    noise-free equilibrium; ``"macro"`` is preferred when both are present),
    falling back to ``"micro"`` if neither closed-form method ran.
    """

    if len(results) < 2:
        raise ValueError("render_density_inference_report needs at least two methods")
    case_numbers = next(iter(results.values())).case_numbers
    for name, result in results.items():
        if result.case_numbers != case_numbers:
            raise ValueError(f"{name} result uses inconsistent case numbers")
    series = {name: _series(result, field) for name, result in results.items()}

    reference_name = next(
        (name for name in ("macro", "analytic") if name in results), next(iter(results))
    )
    reference = results[reference_name]
    reference_series = series[reference_name]

    P_turn = create_turning_kernel(config.turning).P_turn
    gamma1, effective_diffusivity, _ = macroscopic_coefficients(
        config.kinetics,
        config.responses.only_case(0),
        P_turn,
    )
    mobilities = np.asarray(
        [
            macroscopic_coefficients(
                config.kinetics,
                config.responses.only_case(case_number),
                P_turn,
            )[2]
            for case_number in PDF_CASES
        ],
        dtype=float,
    )
    (
        equilibrium_exposure,
        equilibrium_centroid_x,
        equilibrium_centroid_y,
        equilibrium_tv,
    ) = _equilibrium_observables(
        config,
        field,
        reference,
        effective_diffusivity,
        mobilities,
    )
    t50 = np.full(len(PDF_CASES), np.nan)
    t90 = np.full(len(PDF_CASES), np.nan)
    relaxation = np.full(len(PDF_CASES), np.nan)
    for case_index in range(1, len(PDF_CASES)):
        target = equilibrium_exposure[case_index]
        if abs(target) <= 1.0e-14:
            continue
        ratio = reference_series["field_exposure"][case_index] / target
        t50[case_index] = _crossing_time(reference.times, ratio, 0.5)
        t90[case_index] = _crossing_time(reference.times, ratio, 0.9)
        relaxation[case_index] = _effective_relaxation_time(reference.times, ratio)

    summary = Path(summary_path).expanduser().resolve()
    summary.parent.mkdir(parents=True, exist_ok=True)
    field_length = inferred_characteristic_length(config, field)
    angular_time = 1.0 / gamma1
    field_diffusion_time = field_length**2 / effective_diffusivity
    disk_mixing_time = 1.0 / (effective_diffusivity * FIRST_NEUMANN_DISK_WAVENUMBER**2)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "case",
                "kappa",
                "kappa_over_D_eff",
                "equilibrium_delta_mean_eta",
                "equilibrium_delta_centroid_x",
                "equilibrium_delta_centroid_y",
                "equilibrium_total_variation",
                f"{reference_name}_t50",
                f"{reference_name}_t90",
                f"{reference_name}_effective_relaxation_time",
                "angular_relaxation_time",
                "field_diffusion_time",
                "disk_mixing_time",
            )
        )
        for case_index, case_number in enumerate(PDF_CASES):
            writer.writerow(
                (
                    case_number,
                    f"{mobilities[case_index]:.12g}",
                    f"{mobilities[case_index] / effective_diffusivity:.12g}",
                    f"{equilibrium_exposure[case_index]:.12g}",
                    f"{equilibrium_centroid_x[case_index]:.12g}",
                    f"{equilibrium_centroid_y[case_index]:.12g}",
                    f"{equilibrium_tv[case_index]:.12g}",
                    f"{t50[case_index]:.12g}",
                    f"{t90[case_index]:.12g}",
                    f"{relaxation[case_index]:.12g}",
                    f"{angular_time:.12g}",
                    f"{field_diffusion_time:.12g}",
                    f"{disk_mixing_time:.12g}",
                )
            )

    timeseries = Path(timeseries_path).expanduser().resolve()
    _write_timeseries(
        timeseries,
        tuple(results.values()),
        tuple(series.values()),
    )

    figure_output = Path(figure_path).expanduser().resolve()
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12.0, 9.0), dpi=150, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    colors = colormaps["tab10"](np.linspace(0.0, 0.8, len(PDF_CASES) - 1))

    for case_index in range(1, len(PDF_CASES)):
        color = colors[case_index - 1]
        for method_index, (name, result) in enumerate(results.items()):
            values = series[name]
            linestyle = _METHOD_LINESTYLES.get(name, "-.")
            linewidth = _METHOD_LINEWIDTHS.get(name, 1.3)
            axes[0, 0].plot(
                result.times,
                values["field_exposure"][case_index],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                label=f"C{case_index}" if method_index == 0 else None,
            )
            axes[0, 1].plot(
                result.times,
                values["cumulative_field_exposure"][case_index],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )

    style_text = ", ".join(
        f"{_METHOD_LINESTYLES.get(name, '-.')} {name}" for name in results
    )
    axes[0, 0].set(
        title=f"Field exposure ({style_text})",
        xlabel="t",
        ylabel=r"$\Delta\langle\eta\rangle$",
    )
    axes[0, 0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0, 0].legend(ncol=4, fontsize=8, frameon=False)
    axes[0, 1].set(
        title="Time-integrated field exposure on one common scale",
        xlabel="t",
        ylabel=r"$\int_0^t\Delta\langle\eta\rangle\,ds$",
    )
    axes[0, 1].axhline(0.0, color="0.6", linewidth=0.8)

    axes[1, 0].bar(
        np.arange(len(PDF_CASES)),
        mobilities,
        color=colormaps["viridis"](np.linspace(0.15, 0.85, len(PDF_CASES))),
    )
    axes[1, 0].axhline(0.0, color="0.3", linewidth=0.8)
    axes[1, 0].set(
        title="What equation (16) can identify",
        xlabel="case",
        ylabel=r"$\kappa$",
        xticks=np.arange(len(PDF_CASES)),
        xticklabels=CASE_LABELS,
    )

    identifiable = np.flatnonzero(np.abs(mobilities) > 1.0e-14)
    width = 0.25
    positions = np.arange(len(identifiable))
    axes[1, 1].bar(
        positions - width,
        t50[identifiable],
        width=width,
        label=r"$t_{50}$",
    )
    axes[1, 1].bar(
        positions,
        relaxation[identifiable],
        width=width,
        label=r"$\tau_{\mathrm{fit}}$",
    )
    axes[1, 1].bar(
        positions + width,
        t90[identifiable],
        width=width,
        label=r"$t_{90}$",
    )
    axes[1, 1].axhline(
        angular_time,
        color="0.25",
        linestyle=":",
        linewidth=1.2,
        label=r"$1/\gamma_1$",
    )
    axes[1, 1].axhline(
        disk_mixing_time,
        color="0.45",
        linestyle="-.",
        linewidth=1.2,
        label="disk mixing",
    )
    axes[1, 1].set(
        title="Dimensionless relaxation times",
        ylabel="time",
        xticks=positions,
        xticklabels=[f"C{case}" for case in identifiable],
        yscale="log",
    )
    axes[1, 1].legend(ncol=2, fontsize=8, frameon=False)
    if np.all(np.isnan(t50[identifiable])) and np.all(np.isnan(t90[identifiable])):
        # t50/t90 are undefined (not a plotting bug) whenever the run's
        # duration ends before the signed observable crosses halfway/90% of
        # its own eventual change -- an empty bar at these x-positions means
        # "not reached by t_final", not "value is zero".
        axes[1, 1].text(
            0.5,
            0.5,
            "t50/t90 not reached within this run's duration\n"
            "(only tau_fit, the exponential fit rate, is shown)",
            transform=axes[1, 1].transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="0.4",
        )

    for axis in axes.flat:
        axis.grid(alpha=0.22, linewidth=0.6)
    figure.suptitle(
        "Density observables and microscopic identifiability",
        fontsize=15,
    )
    figure.savefig(figure_output, facecolor="white")
    figure.clear()
    return figure_output, summary, timeseries
