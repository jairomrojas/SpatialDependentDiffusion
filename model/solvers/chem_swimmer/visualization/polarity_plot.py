"""Measured polarity field vs. adiabatic theory and the sliding-wall layer.

Equation (16) is a closed equation for the density alone, obtained by
adiabatically eliminating the polarity ``p`` (equation 15). This module
makes the eliminated field visible again: it estimates ``p(x,t)`` directly
from the same particle ensemble used for the density KDE, compares it with
what the adiabatic slaving relation implies from the *measured* density, and
compares the near-wall layer with the closed-form sliding-wall estimate and
derivation in ``boundary_layer.py``.
"""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..boundary_layer import sliding_wall_boundary_layer
from ..config import ProjectConfig
from ..density import PDF_CASES, PositionPDFResult
from ..fields import ChemicalField
from ..kinetics import macroscopic_coefficients
from ..turning import create_turning_kernel

_RADIAL_BINS = 40
_WALL_LAYER_MAX_Y = 0.3


def _radial_profile(
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    bins: int = _RADIAL_BINS,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Azimuthally-averaged profile of ``values`` against radius on ``[0, 1]``."""

    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radius = np.sqrt(x_grid * x_grid + y_grid * y_grid)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    flat_radius = radius[mask]
    flat_values = values[mask]
    bin_index = np.clip(np.digitize(flat_radius, edges) - 1, 0, bins - 1)
    sums = np.bincount(bin_index, weights=flat_values, minlength=bins)
    counts = np.bincount(bin_index, minlength=bins).astype(float)
    with np.errstate(invalid="ignore"):
        profile = sums / counts
    profile[counts == 0.0] = np.nan
    return centers, profile


def _wall_normal_component(
    coordinates: NDArray[np.float64],
    field_x: NDArray[np.float64],
    field_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project a vector field onto the local outward radial (wall-normal) direction."""

    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radius = np.sqrt(x_grid * x_grid + y_grid * y_grid)
    safe_radius = np.where(radius > 1.0e-9, radius, 1.0)
    return (x_grid * field_x + y_grid * field_y) / safe_radius


def _fit_exponential_decay(
    y: NDArray[np.float64],
    magnitude: NDArray[np.float64],
) -> tuple[float, float]:
    """Least-squares fit of ``magnitude ~ a * exp(-rate * y)``; returns ``(a, rate)``."""

    valid = np.isfinite(y) & np.isfinite(magnitude) & (magnitude > 0.0) & (y >= 0.0)
    if np.count_nonzero(valid) < 3:
        return float("nan"), float("nan")
    log_magnitude = np.log(magnitude[valid])
    design = np.stack([np.ones_like(y[valid]), -y[valid]], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, log_magnitude, rcond=None)
    return float(np.exp(coefficients[0])), float(coefficients[1])


def _masked_directional_gradient(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    spacing: float,
    axis: int,
) -> NDArray[np.float64]:
    """Finite-difference derivative along ``axis`` that never reaches across the
    disk mask boundary into the artificial zero-padding used outside it.

    Density arrays are set to exactly 0 outside the disk mask (a storage/
    display convention, not a physical boundary condition), so a plain
    ``np.gradient`` call differences a real interior value against that
    padding at every mask-edge pixel -- an O(1/dx) cliff that has nothing to
    do with the true near-wall density gradient and dwarfs it by an order of
    magnitude. This uses a centered difference where both neighbors along
    ``axis`` are inside the mask, a one-sided difference where only one is,
    and 0 for isolated single-pixel-wide mask rows/columns.
    """

    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (1, 1)
    padded_values = np.pad(values, pad_width, mode="edge")
    padded_mask = np.pad(mask, pad_width, mode="constant", constant_values=False)

    def shifted(array: NDArray, offset: int) -> NDArray:
        length = values.shape[axis]
        indexer: list[slice] = [slice(None)] * array.ndim
        indexer[axis] = slice(1 + offset, 1 + offset + length)
        return array[tuple(indexer)]

    prev_values, next_values = shifted(padded_values, -1), shifted(padded_values, 1)
    prev_valid, next_valid = shifted(padded_mask, -1), shifted(padded_mask, 1)

    central = (next_values - prev_values) / (2.0 * spacing)
    forward = (next_values - values) / spacing
    backward = (values - prev_values) / spacing

    return np.where(
        prev_valid & next_valid,
        central,
        np.where(next_valid, forward, np.where(prev_valid, backward, 0.0)),
    )


def _adiabatic_polarity(
    density: NDArray[np.float64],
    eta_gradient: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    v0: float,
    gamma1: float,
    source_coefficient: float,
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Equation (15): the density-slaved polarity, evaluated from a measured density."""

    cell_width = coordinates[1] - coordinates[0]
    density_gradient_y = _masked_directional_gradient(
        density, mask, cell_width, axis=-2
    )
    density_gradient_x = _masked_directional_gradient(
        density, mask, cell_width, axis=-1
    )
    prefactor = -1.0 / (2.0 * gamma1)
    theory_x = prefactor * (
        v0 * density_gradient_x + source_coefficient * density * eta_gradient[..., 0]
    )
    theory_y = prefactor * (
        v0 * density_gradient_y + source_coefficient * density * eta_gradient[..., 1]
    )
    return theory_x, theory_y


def compute_theoretical_polarity_field(
    pdf: PositionPDFResult,
    config: ProjectConfig,
    field: ChemicalField,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate equation (15) at every case and saved time of ``pdf``.

    ``pdf`` is typically the closed-equation (macro) result, which has no
    polarity of its own: the macro solver only ever tracks the scalar
    density. This reconstructs the polarity the adiabatic slaving relation
    implies from that density, so it can be compared or rendered on equal
    footing with the directly measured microscopic polarity field.
    """

    coordinates = pdf.coordinates
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    points = np.stack((x_grid, y_grid), axis=-1)
    eta_gradient = np.asarray(field.gradient(points), dtype=float)
    P_turn = create_turning_kernel(config.turning).P_turn
    gamma1, _, _ = macroscopic_coefficients(
        config.kinetics, config.responses.zeros(), P_turn
    )
    v0 = config.kinetics.v0
    if not (math.isfinite(gamma1) and gamma1 > 0.0):
        raise ValueError("gamma1 must be finite and positive to evaluate equation (15)")

    theory_x = np.zeros(pdf.density.shape, dtype=np.float64)
    theory_y = np.zeros(pdf.density.shape, dtype=np.float64)
    for case_index, case_number in enumerate(pdf.case_numbers):
        responses = config.responses.only_case(case_number)
        source_coefficient = (
            responses.v_eta
            + responses.D_R_zeta
            + (1.0 - P_turn) * responses.lambda_zeta
            - responses.chi_theta
        )
        for time_index in range(len(pdf.times)):
            density_slice = pdf.density[case_index, time_index].astype(float)
            case_theory_x, case_theory_y = _adiabatic_polarity(
                density_slice,
                eta_gradient,
                coordinates,
                v0,
                gamma1,
                source_coefficient,
                mask,
            )
            theory_x[case_index, time_index] = case_theory_x
            theory_y[case_index, time_index] = case_theory_y

    # Equation (slaved) applies after the fast angular transient.  At the
    # literal initial frame use the prescribed kinetic initial condition,
    # not its post-relaxation replacement.  This makes all three numerical
    # methods start from the same rho_in and p_in while leaving every t>0
    # theoretical frame governed by the manuscript's slaving relation.
    if config.density.initial_heading_distribution == "isotropic":
        theory_x[:, 0] = 0.0
        theory_y[:, 0] = 0.0
    else:
        heading_x = math.cos(config.initial.heading)
        heading_y = math.sin(config.initial.heading)
        theory_x[:, 0] = pdf.density[:, 0] * heading_x
        theory_y[:, 0] = pdf.density[:, 0] * heading_y
    return theory_x, theory_y


def render_polarity_report(
    micro: PositionPDFResult,
    config: ProjectConfig,
    field: ChemicalField,
    figure_path: str | Path,
    summary_path: str | Path,
    cases: tuple[int, ...] = (0, 2, 3, 4, 5),
) -> tuple[Path, Path]:
    """Compare the measured polarity field with adiabatic theory and the wall layer.

    Writes a figure with: the measured vector field for Case 0; the measured
    vs. analytic near-wall polarity and density profiles; a cross-case
    comparison of the near-wall layer for the bulk-closure-zero cases; and
    the measured-vs-adiabatic-theory mismatch as a function of radius.
    """

    if micro.polarity_x is None or micro.polarity_y is None:
        raise ValueError(
            "the microscopic result has no polarity field; recompute with "
            "the current density.py (schema >= 3)"
        )
    if any(case_number not in micro.case_numbers for case_number in cases):
        raise ValueError("requested cases are not all present in the micro result")

    coordinates = micro.coordinates
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    time_index = len(micro.times) - 1

    P_turn = create_turning_kernel(config.turning).P_turn
    gamma1, _, _ = macroscopic_coefficients(
        config.kinetics, config.responses.zeros(), P_turn
    )
    v0 = config.kinetics.v0
    layer = sliding_wall_boundary_layer(config.kinetics, P_turn)

    points = np.stack((x_grid, y_grid), axis=-1)
    eta_gradient = np.asarray(field.gradient(points), dtype=float)

    summary_rows: list[dict[str, float | int | str]] = []
    wall_profiles: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    mismatch_profiles: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    for case_number in cases:
        density_slice = micro.density[case_number, time_index].astype(float)
        polarity_x = micro.polarity_x[case_number, time_index].astype(float)
        polarity_y = micro.polarity_y[case_number, time_index].astype(float)

        # _wall_normal_component projects onto the *outward* radial direction,
        # so a swimmer heading at the wall (accumulating there) contributes
        # positively: this is already |p_wall| directly, no extra sign flip.
        wall_normal = _wall_normal_component(coordinates, polarity_x, polarity_y)
        centers, p_wall_profile = _radial_profile(coordinates, wall_normal, mask)
        _, rho_profile = _radial_profile(coordinates, density_slice, mask)
        distance_from_wall = 1.0 - centers
        wall_profiles[case_number] = (distance_from_wall, p_wall_profile)

        near_wall = distance_from_wall <= _WALL_LAYER_MAX_Y
        amplitude_fit, rate_fit = _fit_exponential_decay(
            distance_from_wall[near_wall],
            p_wall_profile[near_wall],
        )
        bulk = float(np.nanmean(rho_profile[centers <= 0.5]))

        responses = config.responses.only_case(case_number)
        source_coefficient = (
            responses.v_eta
            + responses.D_R_zeta
            + (1.0 - P_turn) * responses.lambda_zeta
        )
        if math.isfinite(gamma1) and gamma1 > 0.0:
            theory_x, theory_y = _adiabatic_polarity(
                density_slice,
                eta_gradient,
                coordinates,
                v0,
                gamma1,
                source_coefficient,
                mask,
            )
            mismatch = np.hypot(polarity_x - theory_x, polarity_y - theory_y)
            mismatch[~mask] = np.nan
            radial_centers, mismatch_profile = _radial_profile(
                coordinates, mismatch, mask
            )
            mismatch_profiles[case_number] = (radial_centers, mismatch_profile)

        summary_rows.append(
            {
                "case": case_number,
                "measured_p_wall_amplitude": amplitude_fit,
                "measured_p_wall_decay_rate": rate_fit,
                "measured_p_wall_width": 1.0 / rate_fit
                if rate_fit > 0.0
                else float("nan"),
                "measured_bulk_density": bulk,
                "analytic_p_wall_over_bulk": layer.wall_polarity_over_bulk_density,
                "analytic_p_wall_amplitude": layer.wall_polarity_over_bulk_density
                * bulk,
                "analytic_decay_rate": layer.decay_rate,
                "analytic_width": layer.width,
                "peclet": layer.peclet,
                "gamma1": layer.gamma1,
                "boundary_law": "sliding",
            }
        )

    summary = Path(summary_path).expanduser().resolve()
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    figure_output = Path(figure_path).expanduser().resolve()
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(14.0, 9.0), dpi=150, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)

    reference_case = 0 if 0 in cases else cases[0]
    density_reference = micro.density[reference_case, time_index].astype(float)
    polarity_reference_x = micro.polarity_x[reference_case, time_index].astype(float)
    polarity_reference_y = micro.polarity_y[reference_case, time_index].astype(float)
    axis = axes[0, 0]
    mesh = axis.pcolormesh(
        x_grid, y_grid, density_reference, cmap="viridis", shading="auto"
    )
    figure.colorbar(mesh, ax=axis, shrink=0.8, label=r"$\rho$")
    stride = max(1, len(coordinates) // 24)
    axis.quiver(
        x_grid[::stride, ::stride],
        y_grid[::stride, ::stride],
        polarity_reference_x[::stride, ::stride],
        polarity_reference_y[::stride, ::stride],
        color="white",
        scale=None,
        width=0.004,
    )
    circle = np.linspace(0.0, 2.0 * np.pi, 200)
    axis.plot(np.cos(circle), np.sin(circle), color="0.9", linewidth=1.0)
    axis.set(
        title=f"Measured $\\rho$ and $\\vec p$: Case {reference_case}, t={micro.times[time_index]:.3g}",
        aspect="equal",
        xlim=(-1.05, 1.05),
        ylim=(-1.05, 1.05),
    )

    axis = axes[0, 1]
    distance_from_wall, p_wall_profile = wall_profiles[reference_case]
    axis.plot(
        distance_from_wall,
        p_wall_profile,
        "o",
        markersize=3.0,
        color=colormaps["tab10"](0),
        label="measured $p_n(y)$",
    )
    if math.isfinite(layer.wall_polarity_over_bulk_density):
        bulk = next(
            row["measured_bulk_density"]
            for row in summary_rows
            if row["case"] == reference_case
        )
        y_fine = np.linspace(0.0, _WALL_LAYER_MAX_Y, 100)
        analytic = (
            layer.wall_polarity_over_bulk_density
            * bulk
            * np.exp(-layer.decay_rate * y_fine)
        )
        axis.plot(
            y_fine,
            analytic,
            "-",
            color="0.2",
            label="analytic $|p_{wall}^{(0)}|e^{-y/\\xi}$",
        )
    axis.set(
        title="Near-wall polarity: measured vs. analytic layer",
        xlabel="$y = 1-r$ (distance from wall)",
        ylabel="$p_n$ (outward radial polarity)",
        xlim=(0.0, _WALL_LAYER_MAX_Y),
    )
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 2]
    centers_rho, rho_profile = _radial_profile(coordinates, density_reference, mask)
    distance_rho = 1.0 - centers_rho
    axis.plot(
        distance_rho,
        rho_profile,
        "o",
        markersize=3.0,
        color=colormaps["tab10"](1),
        label="measured $\\rho(y)$",
    )
    if math.isfinite(layer.decay_rate) and v0 > 0.0:
        bulk = next(
            row["measured_bulk_density"]
            for row in summary_rows
            if row["case"] == reference_case
        )
        p_wall_analytic = layer.wall_polarity_over_bulk_density * bulk
        y_fine = np.linspace(0.0, _WALL_LAYER_MAX_Y, 100)
        rho_analytic = bulk + (
            v0 / (config.kinetics.D_T0 * layer.decay_rate)
        ) * p_wall_analytic * np.exp(-layer.decay_rate * y_fine)
        axis.plot(y_fine, rho_analytic, "-", color="0.2", label="analytic $\\rho(y)$")
    axis.set(
        title="Near-wall density excess (Eq. A)",
        xlabel="$y = 1-r$",
        ylabel=r"$\rho$",
        xlim=(0.0, _WALL_LAYER_MAX_Y),
    )
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    colors = colormaps["tab10"](np.linspace(0.0, 0.8, len(cases)))
    for color, case_number in zip(colors, cases, strict=True):
        distance, profile = wall_profiles[case_number]
        axis.plot(
            distance, profile, color=color, linewidth=1.5, label=f"C{case_number}"
        )
    axis.set(
        title="Bulk-closure-zero cases perturb the same wall layer",
        xlabel="$y=1-r$",
        ylabel="$p_n$ (outward)",
        xlim=(0.0, _WALL_LAYER_MAX_Y),
    )
    axis.legend(frameon=False, fontsize=8, ncol=2)

    axis = axes[1, 1]
    for color, case_number in zip(colors, cases, strict=True):
        if case_number not in mismatch_profiles:
            continue
        radial_centers, mismatch_profile = mismatch_profiles[case_number]
        axis.plot(
            radial_centers,
            mismatch_profile,
            color=color,
            linewidth=1.5,
            label=f"C{case_number}",
        )
    axis.set(
        title=r"$|\vec p_{\rm measured}-\vec p_{\rm adiabatic}|$ vs. radius",
        xlabel="r",
        ylabel="mismatch",
    )
    axis.axvspan(1.0 - _WALL_LAYER_MAX_Y, 1.0, color="0.85", zorder=0)
    axis.legend(frameon=False, fontsize=8, ncol=2)

    axis = axes[1, 2]
    for color, case_number in zip(colors, cases, strict=True):
        wall_series = []
        edge_mask = (x_grid * x_grid + y_grid * y_grid >= 0.92**2) & mask
        for t_index in range(len(micro.times)):
            polarity_x = micro.polarity_x[case_number, t_index].astype(float)
            polarity_y = micro.polarity_y[case_number, t_index].astype(float)
            wall_normal = _wall_normal_component(coordinates, polarity_x, polarity_y)
            wall_series.append(float(np.nanmean(wall_normal[edge_mask])))
        axis.plot(
            micro.times,
            wall_series,
            color=color,
            linewidth=1.5,
            label=f"C{case_number}",
        )
    if math.isfinite(gamma1) and gamma1 > 0.0:
        axis.axvline(
            1.0 / gamma1,
            color="0.4",
            linestyle=":",
            linewidth=1.2,
            label=r"$1/\gamma_1$",
        )
    axis.set(
        title="Edge-annulus polarity vs. time",
        xlabel="t",
        ylabel="$\\langle p_n\\rangle_{r>0.92}$",
    )
    axis.legend(frameon=False, fontsize=7, ncol=2)

    for axis in axes.flat:
        axis.grid(alpha=0.22, linewidth=0.6)
    boundary_note = "sliding-wall analytic overlay shown"
    figure.suptitle(
        f"Polarity field vs. adiabatic theory and the wall layer ({boundary_note})",
        fontsize=14,
    )
    figure.savefig(figure_output, facecolor="white")
    figure.clear()
    return figure_output, summary


def _method_display_polarity(
    result: PositionPDFResult, time_index: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The polarity slice used for cross-method comparison.

    Prefers the smoother difference-bandwidth field (present for a measured
    microscopic result) and falls back to the sharp/theoretical one (macro,
    analytic, or a micro result without a declared difference bandwidth).
    """

    if (
        result.difference_polarity_x is not None
        and result.difference_polarity_y is not None
    ):
        return (
            result.difference_polarity_x[:, time_index].astype(float),
            result.difference_polarity_y[:, time_index].astype(float),
        )
    if result.polarity_x is not None and result.polarity_y is not None:
        return (
            result.polarity_x[:, time_index].astype(float),
            result.polarity_y[:, time_index].astype(float),
        )
    raise ValueError(f"{result.method} result has no polarity field")


def _radial_cos_similarity(
    coordinates: NDArray[np.float64],
    dx_a: NDArray[np.float64],
    dy_a: NDArray[np.float64],
    dx_b: NDArray[np.float64],
    dy_b: NDArray[np.float64],
    mask: NDArray[np.bool_],
    bins: int = _RADIAL_BINS,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Radially-binned direction agreement between two vector-difference fields.

    Returns ``(radius_centers, mean_cos, mean_magnitude_a, mean_magnitude_b)``.
    Points where either field's magnitude is negligible relative to its own
    99th percentile are excluded, since the angle of a near-zero vector is
    dominated by noise (matching the diagnostic used to separate genuine
    directional disagreement from finite-sampling noise).
    """

    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radius = np.sqrt(x_grid * x_grid + y_grid * y_grid)
    magnitude_a = np.hypot(dx_a, dy_a)
    magnitude_b = np.hypot(dx_b, dy_b)
    threshold_a = 0.05 * np.percentile(magnitude_a[mask], 99.0)
    threshold_b = 0.05 * np.percentile(magnitude_b[mask], 99.0)
    valid = mask & (magnitude_a > threshold_a) & (magnitude_b > threshold_b)
    cos_similarity = np.full_like(radius, np.nan)
    cos_similarity[valid] = (dx_a[valid] * dx_b[valid] + dy_a[valid] * dy_b[valid]) / (
        magnitude_a[valid] * magnitude_b[valid]
    )

    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_cos = np.full(bins, np.nan)
    mean_mag_a = np.full(bins, np.nan)
    mean_mag_b = np.full(bins, np.nan)
    bin_index = np.clip(np.digitize(radius[mask], edges) - 1, 0, bins - 1)
    for index in range(bins):
        in_bin = mask.copy()
        in_bin[mask] = bin_index == index
        cos_in_bin = cos_similarity[in_bin]
        cos_in_bin = cos_in_bin[np.isfinite(cos_in_bin)]
        if cos_in_bin.size:
            mean_cos[index] = float(np.mean(cos_in_bin))
        if np.any(in_bin):
            mean_mag_a[index] = float(np.mean(magnitude_a[in_bin]))
            mean_mag_b[index] = float(np.mean(magnitude_b[in_bin]))
    return centers, mean_cos, mean_mag_a, mean_mag_b


def render_polarity_agreement_report(
    results: dict[str, PositionPDFResult],
    figure_path: str | Path,
    summary_path: str | Path,
    cases: tuple[int, ...] = PDF_CASES,
) -> tuple[Path, Path]:
    """Deep pairwise agreement between every pair of methods, for the polarity field.

    For each pair of methods present in ``results`` (e.g. micro-macro,
    micro-analytic, macro-analytic) and each case, bins the direction
    agreement (cosine similarity) and magnitude of
    ``delta p = p_case - p_Case0`` by radius, at the final saved time of
    each method. This is the formal version of the diagnostic that showed
    the r~0.6-0.8 mismatch was finite-M noise (it shrank from ~0 to >0.6 at
    8x the ensemble size); running it here answers "do these actually
    match" quantitatively rather than by eye.

    Using ``delta p`` rather than ``p`` itself matters: every case's
    absolute ``p`` (including Case 0's) contains the same near-wall sliding-
    wall boundary-layer polarity (see ``boundary_layer.py``), a real effect
    that equation (15)'s adiabatic slaving does not predict at all (it
    requires an instantaneous polarity relaxation that only holds away from
    the wall). That shared, un-modeled term dominates the weaker
    case-specific chemotactic response for several mechanisms, so comparing
    absolute ``p`` mostly tests whether a weak response happens to project
    onto an unrelated boundary direction -- not whether the theory is
    correct. Differencing against Case 0 cancels that shared term and
    isolates the actual chemotactic response, which is what equation (15)
    claims to predict.

    Defaults to all ten cases. Case 0 has no response by construction, so
    both ``delta p`` fields are identically 0 and the panel is empty. Cases
    2, 4, and 5 have exactly zero closed-equation mobility, so
    ``macro``/``analytic`` predict ``delta p = 0`` everywhere while
    ``micro`` shows the genuine near-wall boundary-layer polarity documented
    in ``boundary_layer.py`` -- the significance filter therefore
    rejects every point (there is no well-defined "direction" of a null
    vector) and those panels are also empty; this is the expected,
    informative signature of a mechanism with zero bulk mobility, not a
    plotting bug.
    """

    if len(results) < 2:
        raise ValueError("render_polarity_agreement_report needs at least two methods")
    names = list(results)
    case_numbers = results[names[0]].case_numbers
    coordinates = results[names[0]].coordinates
    for name in names:
        if results[name].case_numbers != case_numbers:
            raise ValueError(f"{name} result uses inconsistent case numbers")
        if not np.array_equal(results[name].coordinates, coordinates):
            raise ValueError(f"{name} result uses a different output grid")
        if results[name].polarity_x is None:
            raise ValueError(f"{name} result has no polarity field")

    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    summary = Path(summary_path).expanduser().resolve()
    summary.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    profiles: dict[tuple[str, str, int], tuple[NDArray, NDArray, NDArray, NDArray]] = {}
    for case_number in cases:
        case_index = case_numbers.index(case_number)
        for method_a, method_b in pairs:
            time_a = len(results[method_a].times) - 1
            time_b = len(results[method_b].times) - 1
            px_a, py_a = _method_display_polarity(results[method_a], time_a)
            px_b, py_b = _method_display_polarity(results[method_b], time_b)
            dx_a = px_a[case_index] - px_a[0]
            dy_a = py_a[case_index] - py_a[0]
            dx_b = px_b[case_index] - px_b[0]
            dy_b = py_b[case_index] - py_b[0]
            centers, mean_cos, mag_a, mag_b = _radial_cos_similarity(
                coordinates, dx_a, dy_a, dx_b, dy_b, mask
            )
            profiles[(method_a, method_b, case_number)] = (
                centers,
                mean_cos,
                mag_a,
                mag_b,
            )
            bulk = centers <= 0.5
            wall = centers >= 0.85
            # A case with exactly zero delta p on one side (Case 0, or Cases
            # 2/4/5 for macro/analytic) has no well-defined direction
            # anywhere, so every bin is legitimately all-NaN here.
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                bulk_mean_cos = float(np.nanmean(mean_cos[bulk]))
                wall_mean_cos = float(np.nanmean(mean_cos[wall]))
                overall_mean_cos = float(np.nanmean(mean_cos))
            rows.append(
                {
                    "case": case_number,
                    "method_a": method_a,
                    "method_b": method_b,
                    "bulk_mean_cos": bulk_mean_cos,
                    "wall_mean_cos": wall_mean_cos,
                    "overall_mean_cos": overall_mean_cos,
                    "mean_magnitude_a": float(np.nanmean(mag_a)),
                    "mean_magnitude_b": float(np.nanmean(mag_b)),
                }
            )

    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    figure_output = Path(figure_path).expanduser().resolve()
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    columns = 3
    grid_rows = (len(cases) + columns - 1) // columns
    figure = Figure(
        figsize=(5.0 * columns, 3.6 * grid_rows), dpi=150, constrained_layout=True
    )
    FigureCanvasAgg(figure)
    axes = figure.subplots(grid_rows, columns, squeeze=False)
    pair_colors = colormaps["tab10"](np.linspace(0.0, 0.5, max(1, len(pairs))))

    for panel_index, case_number in enumerate(cases):
        axis = axes[panel_index // columns, panel_index % columns]
        for color, (method_a, method_b) in zip(pair_colors, pairs, strict=True):
            centers, mean_cos, _, _ = profiles[(method_a, method_b, case_number)]
            axis.plot(
                centers,
                mean_cos,
                color=color,
                linewidth=1.6,
                label=f"{method_a}-{method_b}",
            )
        axis.axhline(0.0, color="0.6", linewidth=0.8)
        axis.axhline(1.0, color="0.85", linewidth=0.8, linestyle=":")
        axis.set(
            title=f"Case {case_number}",
            xlabel="r",
            ylabel=r"mean $\cos(\Delta p_a,\Delta p_b)$",
            ylim=(-1.05, 1.05),
        )
        axis.grid(alpha=0.22, linewidth=0.6)
        if panel_index == 0:
            axis.legend(frameon=False, fontsize=8)

    for panel_index in range(len(cases), grid_rows * columns):
        axes[panel_index // columns, panel_index % columns].set_visible(False)

    figure.suptitle(
        "Polarity direction agreement by radius, at the final saved time "
        "(1 = identical direction, 0 = uncorrelated, -1 = opposite)",
        fontsize=13,
    )
    figure.savefig(figure_output, facecolor="white")
    figure.clear()
    return figure_output, summary
