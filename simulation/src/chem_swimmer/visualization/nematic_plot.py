"""Paper-ready microscopic nematic-field and ``Q=0`` closure audit."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.ticker import PercentFormatter

from ..config import ProjectConfig
from ..density import PositionPDFResult
from ..fields import ChemicalField
from ..nematic import (
    nematic_closure_rows,
    nematic_frobenius,
    reconstruct_density_flux,
)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#6B7280"
LIGHT_GRAY = "#C7CDD4"


def _write_rows(path: str | Path, rows: list[dict[str, object]]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def _panel_label(axis, label: str) -> None:
    artist = axis.text(
        -0.18,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )
    artist.set_in_layout(False)


def _paper_axis(axis) -> None:
    axis.tick_params(labelsize=8, width=0.7, length=3)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def _map_axis(axis, crop: float) -> None:
    axis.set(
        xlabel=r"$x/R$",
        ylabel=r"$y/R$",
        xlim=(-crop, crop),
        ylim=(-crop, crop),
        aspect="equal",
    )
    axis.add_patch(Circle((0.0, 0.0), 1.0, fill=False, color="0.25", lw=0.8, ls="--"))
    _paper_axis(axis)


def _finite_upper(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(float(np.percentile(finite, percentile)), np.finfo(float).eps)


def _director_segments(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    qxx: np.ndarray,
    qxy: np.ndarray,
    mask: np.ndarray,
    stride: int,
    length: float,
) -> list[list[tuple[float, float]]]:
    """Head-tail symmetric line segments along Q's principal director."""

    angle = 0.5 * np.arctan2(qxy, qxx)
    segments: list[list[tuple[float, float]]] = []
    for row in range(0, qxx.shape[0], stride):
        for column in range(0, qxx.shape[1], stride):
            if not mask[row, column]:
                continue
            dx = 0.5 * length * np.cos(angle[row, column])
            dy = 0.5 * length * np.sin(angle[row, column])
            x = x_grid[row, column]
            y = y_grid[row, column]
            segments.append([(x - dx, y - dy), (x + dx, y + dy)])
    return segments


def render_nematic_closure_report(
    result: PositionPDFResult,
    config: ProjectConfig,
    field: ChemicalField,
    figure_path: str | Path,
    summary_path: str | Path,
    timeseries_path: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """Write a publication figure and numerical Q-closure audit.

    ``|Q|`` is the Frobenius norm. Local ratios exclude statistically empty
    KDE tails; ``|Q|/|p|`` also excludes cells whose polarity is effectively
    zero. The split-half curve is a full-M Monte Carlo error-scale estimate,
    not a confidence interval.
    """

    required = (
        result.nematic_xx,
        result.nematic_xy,
        result.polarity_x,
        result.polarity_y,
        result.nematic_split_half_error_xx,
        result.nematic_split_half_error_xy,
    )
    if any(value is None for value in required):
        raise ValueError(
            "the paper Q audit requires microscopic rho, p, Q, and split-half Q fields"
        )

    rows = nematic_closure_rows(result, config, field)
    final_time = float(result.times[-1])
    final_rows = [row for row in rows if float(row["time"]) == final_time]
    summary = _write_rows(summary_path, final_rows)
    timeseries = _write_rows(timeseries_path, rows)

    figure_output = Path(figure_path).expanduser().resolve()
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = figure_output.with_suffix(".pdf")
    figure = Figure(figsize=(12.8, 8.1), dpi=180, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    cases = np.asarray([int(row["case"]) for row in final_rows])
    case_labels = [f"C{case}" for case in cases]

    # (a) Q/rho on its own scale, with the independent split-half error scale.
    q_rho = np.asarray([float(row["q_over_rho_l2"]) for row in final_rows])
    q_rho_error = np.asarray(
        [float(row["q_sampling_error_over_rho_l2"]) for row in final_rows]
    )
    bar_colors = [ORANGE if case == 7 else BLUE for case in cases]
    axes[0, 0].bar(cases, q_rho, color=bar_colors, width=0.72, label="measured")
    axes[0, 0].scatter(
        cases,
        q_rho_error,
        marker="D",
        s=22,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=0.9,
        zorder=3,
        label="split-half MC error scale",
    )
    axes[0, 0].set(
        title=r"Nematic magnitude relative to density",
        xlabel="response case",
        ylabel=r"$|Q|_2/|\rho|_2$",
        xticks=cases,
        xticklabels=case_labels,
    )
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    axes[0, 0].set_ylim(0.0, 1.28 * float(np.max(q_rho)))
    axes[0, 0].set_title(axes[0, 0].get_title(), fontsize=9.5, pad=7)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0, 0].grid(axis="y", alpha=0.2, lw=0.6)

    # (b) Q/p needs its own scale; the weighted 95th percentile shows tails.
    q_p = np.asarray([float(row["q_over_p_l2"]) for row in final_rows])
    q_p_p95 = np.asarray([float(row["q_over_p_weighted_p95"]) for row in final_rows])
    axes[0, 1].bar(cases, q_p, color=bar_colors, width=0.72, label=r"global $L^2$")
    axes[0, 1].scatter(
        cases,
        q_p_p95,
        marker="o",
        s=24,
        color="black",
        zorder=3,
        label="polarity-weighted 95th percentile",
    )
    axes[0, 1].set(
        title=r"Nematic magnitude relative to polarity",
        xlabel="response case",
        ylabel=r"$|Q|_2/|\mathbf{p}|_2$",
        xticks=cases,
        xticklabels=case_labels,
    )
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0, 1].set_ylim(0.0, max(0.08, 1.2 * float(np.max(q_p_p95))))
    axes[0, 1].set_title(axes[0, 1].get_title(), fontsize=9.5, pad=7)
    axes[0, 1].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0, 1].grid(axis="y", alpha=0.2, lw=0.6)

    # (c) Dynamics: baseline versus the only case with a direct Q density current.
    for case_number, color in ((0, GRAY), (7, ORANGE)):
        case_rows = [row for row in rows if int(row["case"]) == case_number]
        times = np.asarray([float(row["time"]) for row in case_rows])
        measured = np.asarray([float(row["q_over_rho_l2"]) for row in case_rows])
        sampling = np.asarray(
            [float(row["q_sampling_error_over_rho_l2"]) for row in case_rows]
        )
        axes[0, 2].plot(
            times,
            measured,
            color=color,
            lw=1.8,
            label=f"C{case_number} measured",
        )
        axes[0, 2].plot(
            times,
            sampling,
            color=color,
            lw=1.1,
            ls="--",
            alpha=0.8,
            label=f"C{case_number} MC error",
        )
    axes[0, 2].set(
        title=r"Time evolution and finite-$M$ error scale",
        xlabel=r"dimensionless time $t$",
        ylabel=r"$|Q|_2/|\rho|_2$",
        xlim=(0.0, final_time),
    )
    axes[0, 2].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    axes[0, 2].set_title(axes[0, 2].get_title(), fontsize=9.5, pad=7)
    axes[0, 2].legend(frameon=False, fontsize=7.2, ncol=2, loc="upper left")
    axes[0, 2].grid(alpha=0.2, lw=0.6)

    case_number = 7 if 7 in result.case_numbers else int(result.case_numbers[-1])
    case_index = result.case_numbers.index(case_number)
    time_index = len(result.times) - 1
    rho = result.density[case_index, time_index].astype(float)
    px = result.polarity_x[case_index, time_index].astype(float)
    py = result.polarity_y[case_index, time_index].astype(float)
    qxx = result.nematic_xx[case_index, time_index].astype(float)
    qxy = result.nematic_xy[case_index, time_index].astype(float)
    q_error_xx = result.nematic_split_half_error_xx[case_index, time_index].astype(
        float
    )
    q_error_xy = result.nematic_split_half_error_xy[case_index, time_index].astype(
        float
    )
    q_norm = nematic_frobenius(qxx, qxy)
    q_error_norm = nematic_frobenius(q_error_xx, q_error_xy)
    p_norm = np.hypot(px, py)
    coordinates = result.coordinates
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    disk = x_grid * x_grid + y_grid * y_grid <= config.domain.radius**2
    populated = disk & (rho >= 0.01 * float(np.max(rho[disk])))
    p99 = float(np.percentile(p_norm[populated], 99.0))
    polar = populated & (p_norm >= 0.05 * p99) & (p_norm > 0.0)
    q_over_rho = np.full_like(rho, np.nan)
    q_over_rho[populated] = q_norm[populated] / rho[populated]
    q_over_p = np.full_like(rho, np.nan)
    q_over_p[polar] = q_norm[polar] / p_norm[polar]
    extent = (
        float(coordinates[0]),
        float(coordinates[-1]),
        float(coordinates[0]),
        float(coordinates[-1]),
    )
    occupied_radius = float(np.max(np.hypot(x_grid[populated], y_grid[populated])))
    crop = min(1.0, max(0.5, occupied_radius + 0.08))
    stride = max(1, len(coordinates) // 18)

    # (d) The Q tensor itself: magnitude as color and head-tail director as segments.
    q_rho_vmax = _finite_upper(q_over_rho)
    image = axes[1, 0].imshow(
        q_over_rho,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=q_rho_vmax,
        interpolation="nearest",
    )
    resolved_director = populated & (q_norm > q_error_norm)
    segments = _director_segments(
        x_grid,
        y_grid,
        qxx,
        qxy,
        resolved_director,
        stride,
        length=0.055,
    )
    if segments:
        axes[1, 0].add_collection(
            LineCollection(segments, colors="white", linewidths=0.8, alpha=0.9)
        )
    axes[1, 0].set_title(
        "Spatial nematic field (Case 7)\n"
        r"color $=|Q|_F/\rho$; head-tail lines $=$ director",
        fontsize=8.5,
    )
    _map_axis(axes[1, 0], crop)
    colorbar = figure.colorbar(image, ax=axes[1, 0], fraction=0.047, pad=0.025)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.formatter = PercentFormatter(1.0, decimals=1)
    colorbar.update_ticks()

    # (e) Q relative to p, with unit polarity arrows to explain the denominator.
    q_p_vmax = _finite_upper(q_over_p)
    image = axes[1, 1].imshow(
        q_over_p,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=q_p_vmax,
        interpolation="nearest",
    )
    arrow_mask = polar[::stride, ::stride]
    sampled_p = p_norm[::stride, ::stride]
    unit_px = np.where(
        arrow_mask,
        px[::stride, ::stride] / np.maximum(sampled_p, np.finfo(float).eps),
        np.nan,
    )
    unit_py = np.where(
        arrow_mask,
        py[::stride, ::stride] / np.maximum(sampled_p, np.finfo(float).eps),
        np.nan,
    )
    axes[1, 1].quiver(
        x_grid[::stride, ::stride],
        y_grid[::stride, ::stride],
        unit_px,
        unit_py,
        color="white",
        alpha=0.85,
        angles="xy",
        scale_units="xy",
        scale=15.0,
        width=0.005,
        headwidth=3.2,
    )
    axes[1, 1].set_title(
        "Nematic-to-polar ratio (Case 7)\n"
        r"color $=|Q|_F/|\mathbf{p}|$; arrows $=\mathbf{p}$ direction",
        fontsize=8.5,
    )
    _map_axis(axes[1, 1], crop)
    colorbar = figure.colorbar(image, ax=axes[1, 1], fraction=0.047, pad=0.025)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.formatter = PercentFormatter(1.0, decimals=0)
    colorbar.update_ticks()

    # (f) The actual omitted term in the exact density current.
    flux = reconstruct_density_flux(result, config, field, case_index, time_index)
    flux_magnitude = np.hypot(flux.q_x, flux.q_y)
    flux_display = np.where(populated, flux_magnitude, np.nan)
    flux_vmax = _finite_upper(flux_display)
    image = axes[1, 2].imshow(
        flux_display,
        origin="lower",
        extent=extent,
        cmap="cividis",
        vmin=0.0,
        vmax=flux_vmax,
        interpolation="nearest",
    )
    sampled_flux = flux_magnitude[::stride, ::stride]
    flux_mask = populated[::stride, ::stride] & (sampled_flux > 0.0)
    unit_jx = np.where(
        flux_mask,
        flux.q_x[::stride, ::stride] / np.maximum(sampled_flux, np.finfo(float).eps),
        np.nan,
    )
    unit_jy = np.where(
        flux_mask,
        flux.q_y[::stride, ::stride] / np.maximum(sampled_flux, np.finfo(float).eps),
        np.nan,
    )
    axes[1, 2].quiver(
        x_grid[::stride, ::stride],
        y_grid[::stride, ::stride],
        unit_jx,
        unit_jy,
        color="white",
        alpha=0.85,
        angles="xy",
        scale_units="xy",
        scale=15.0,
        width=0.005,
        headwidth=3.2,
    )
    case_row = next(row for row in final_rows if int(row["case"]) == case_number)
    axes[1, 2].set_title(
        "Omitted exact density current (Case 7)\n"
        r"$\mathbf{J}_Q=v^{(\zeta)}Q\nabla\eta$; arrows show direction",
        fontsize=8.5,
    )
    axes[1, 2].text(
        0.03,
        0.97,
        "\n".join(
            (
                rf"$\|J_Q\|/\|J_{{exact}}\|={100.0 * float(case_row['q_flux_over_exact_flux']):.3f}\%$",
                rf"$\|J_Q\|/\|J_{{response}}\|={100.0 * float(case_row['q_flux_over_response_flux']):.3f}\%$",
                rf"debiased/response $={100.0 * float(case_row['q_density_flux_noise_debiased_over_response_flux']):.3f}\%$",
                rf"signal/MC error $={float(case_row['q_density_flux_signal_to_sampling_error']):.2f}$",
            )
        ),
        transform=axes[1, 2].transAxes,
        va="top",
        ha="left",
        fontsize=7.4,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9, "pad": 3},
    )
    _map_axis(axes[1, 2], crop)
    colorbar = figure.colorbar(image, ax=axes[1, 2], fraction=0.047, pad=0.025)
    colorbar.set_label(r"$|\mathbf{J}_Q|$", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    for label, axis in zip("abcdef", axes.flat, strict=True):
        _panel_label(axis, f"({label})")
        _paper_axis(axis)

    figure.suptitle(
        "Microscopic test of the polar closure $Q=0$\n"
        rf"$|Q|_F=\sqrt{{2(Q_{{xx}}^2+Q_{{xy}}^2)}}$, "
        rf"$M={result.ensemble_size:,}$, $h/R={result.bandwidth_fraction:g}$, "
        rf"$t={final_time:g}$",
        fontsize=12.5,
    )
    figure.supxlabel(
        "Orange denotes C7 (directional propulsion), the only direct Q contribution "
        "to the density current.\n"
        r"Local maps retain $\rho\geq0.01\rho_{max}$; $|Q|/|\mathbf{p}|$ also requires "
        r"$|\mathbf{p}|\geq0.05p_{99}$. Colors are clipped at the 99th percentile.",
        fontsize=7.5,
    )
    figure.savefig(figure_output, dpi=300, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf_output, facecolor="white", bbox_inches="tight")
    figure.clear()
    return figure_output, pdf_output, summary, timeseries
