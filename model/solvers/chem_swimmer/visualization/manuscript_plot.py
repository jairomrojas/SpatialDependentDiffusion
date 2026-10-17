"""Publication figures: one three-column sheet per response strategy.

Each PDF places the microscopic random walk, the polar-closed PDE, and the
short-time Fourier/Green solution side by side at the latest wall-free time.
Layout follows APS/PRX figure rules: letter-page physical size so a PDF
reader at page-width is already legible, panel letters that remain at least
2 mm after shrinking to the journal page, annotations inside their own
panel, and no in-map boxes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke
from numpy.typing import NDArray

from ..cases import CASES, STRATEGY_FIGURE_STEMS, get_case
from ..config import ProjectConfig
from ..density import PositionPDFResult, resolve_initial_gaussian_sigma
from ..fields import create_field
from ..numerics import inferred_characteristic_length
from .comparison_plot import (
    _agreement_metrics,
    _disk_mesh,
    _display_density,
    _display_polarity,
    _masked,
    _nearest_time_index,
    _on_reference_grid,
    _vector_agreement_metrics,
)

BLUE = "#0072B2"
ORANGE = "#D55E00"
BLACK = "#111111"
GRAY = "#374151"
NUTRIENT = "#9CA3AF"
_HALO = [withStroke(linewidth=3.6, foreground="white")]

# One name per description, matching the manuscript captions and the legends
# of Figs. 4-6: the microscopic random walk, the finite-volume solution of the
# closed polar system, and the closed-form first-order Fourier solution.
PAPER_COLUMN_LABELS = {
    "micro": "Particles",
    "macro": "Polar PDE",
    "moment": "Polar PDE",
    "analytic": "Fourier",
}
# The nine sensitivities in model/parameters/default.toml are calibrated so that all
# nine responses are comparable at the end of the configured run, where the
# O(t^3) rules have caught up with the O(t) ones.  Reporting there is what
# makes one shared linear scale an honest comparison.
PAPER_REPORT_TIME = 5.0e-3
PAPER_LINE_STYLES = {
    "micro": {
        "color": BLACK,
        "lw": 2.8,
        "ls": "-",
        "zorder": 4,
        "solid_capstyle": "round",
    },
    "macro": {"color": BLUE, "lw": 2.5, "ls": "--", "zorder": 3},
    "moment": {"color": BLUE, "lw": 2.5, "ls": "--", "zorder": 3},
    "analytic": {"color": ORANGE, "lw": 2.7, "ls": "-.", "zorder": 3},
}
_DIFFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "manuscript_delta",
    (
        (0.00, "#2166ac"),
        (0.20, "#67a9cf"),
        (0.40, "#d1e5f0"),
        (0.50, "#f7f7f7"),
        (0.60, "#fddbc7"),
        (0.80, "#ef8a62"),
        (1.00, "#b2182b"),
    ),
)
# Letter-page canvas: opened at "fit page" the type is already 12–18 pt.
# Shrinking to PRX two-column (7.0 in) still leaves >= 8 pt (>= 2 mm).
_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ("Arial", "Helvetica", "DejaVu Sans"),
    "font.size": 14,
    "axes.labelsize": 17,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 1.15,
    "axes.labelpad": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,
    "axes.formatter.use_mathtext": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _paper_methods(
    results: dict[str, PositionPDFResult],
) -> list[tuple[str, PositionPDFResult]]:
    if len(results) < 2:
        raise ValueError("manuscript strategy figures require at least two methods")
    ordered: list[tuple[str, PositionPDFResult]] = []
    seen_labels: set[str] = set()
    if "micro" in results:
        ordered.append(("micro", results["micro"]))
        seen_labels.add(PAPER_COLUMN_LABELS["micro"])
    pde_name = (
        "macro" if "macro" in results else "moment" if "moment" in results else None
    )
    if pde_name is not None:
        ordered.append((pde_name, results[pde_name]))
        seen_labels.add(PAPER_COLUMN_LABELS[pde_name])
    if "analytic" in results:
        ordered.append(("analytic", results["analytic"]))
        seen_labels.add(PAPER_COLUMN_LABELS["analytic"])
    for name, result in results.items():
        label = PAPER_COLUMN_LABELS.get(name, name)
        if name in {item[0] for item in ordered} or label in seen_labels:
            continue
        ordered.append((name, result))
        seen_labels.add(label)
    if len(ordered) < 2:
        raise ValueError("manuscript strategy figures require at least two methods")
    case_numbers = ordered[0][1].case_numbers
    for name, result in ordered:
        if result.case_numbers != case_numbers:
            raise ValueError(f"{name} result uses inconsistent case numbers")
    return ordered


def shared_paper_limits(
    methods: list[tuple[str, PositionPDFResult]],
    cases: tuple[int, ...],
    final_time: float,
    percentile: float,
) -> tuple[float, float]:
    """Return common endpoint limits used by all static and animated media."""

    density_values: list[NDArray[np.float64]] = []
    polarity_values: list[NDArray[np.float64]] = []
    for _, result in methods:
        _, _, mask = _disk_mesh(result.coordinates)
        sampled_mask = mask[::4, ::4]
        time_index = _nearest_time_index(result, final_time)
        density_fields = (
            result.difference_density
            if result.difference_density is not None
            else result.density - result.density[0:1]
        )
        polarity_x = (
            result.difference_polarity_x
            if result.difference_polarity_x is not None
            else result.polarity_x - result.polarity_x[0:1]
        )
        polarity_y = (
            result.difference_polarity_y
            if result.difference_polarity_y is not None
            else result.polarity_y - result.polarity_y[0:1]
        )
        for case_number in cases:
            case_index = result.case_numbers.index(case_number)
            density_values.append(
                np.abs(
                    (
                        density_fields[case_index, time_index, ::4, ::4]
                        - density_fields[0, time_index, ::4, ::4]
                    )[sampled_mask]
                ).ravel()
            )
            polarity_values.append(
                np.hypot(
                    polarity_x[case_index, time_index, ::4, ::4]
                    - polarity_x[0, time_index, ::4, ::4],
                    polarity_y[case_index, time_index, ::4, ::4]
                    - polarity_y[0, time_index, ::4, ::4],
                )[sampled_mask].ravel()
            )
    return (
        _nice_limit(float(np.percentile(np.concatenate(density_values), percentile))),
        _nice_limit(float(np.percentile(np.concatenate(polarity_values), percentile))),
    )


def shared_paper_crop(
    methods: list[tuple[str, PositionPDFResult]],
    cases: tuple[int, ...],
    final_time: float,
) -> float:
    """Use one spatial window for every synchronized paper panel."""

    return max(
        _cloud_crop(methods, methods[0][1].case_numbers.index(case), final_time)
        for case in cases
    )


def _common_final_time(results: list[tuple[str, PositionPDFResult]]) -> float:
    start = max(float(result.times[0]) for _, result in results)
    stop = min(float(result.times[-1]) for _, result in results)
    if stop < start:
        raise ValueError("method time windows do not overlap")
    return stop


def _horizontal_cut(
    values: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    sample_x: NDArray[np.float64],
    y: float = 0.0,
) -> NDArray[np.float64]:
    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator(
        (coordinates, coordinates),
        values,
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack((np.full(sample_x.shape, y), sample_x))
    return interpolator(points)


def _cloud_crop(
    results: list[tuple[str, PositionPDFResult]],
    case_index: int,
    time: float,
) -> float:
    radii: list[float] = []
    for _, result in results:
        index = _nearest_time_index(result, time)
        density = result.density[0, index].astype(float)
        x_grid, y_grid, mask = _disk_mesh(result.coordinates)
        peak = float(np.nanmax(density[mask]))
        occupied = mask & (density >= 0.02 * max(peak, np.finfo(float).eps))
        if case_index > 0:
            response = np.abs(_display_density(result, case_index, index))
            response_peak = float(np.nanmax(response[mask]))
            occupied |= mask & (
                response >= 0.08 * max(response_peak, np.finfo(float).eps)
            )
        if np.any(occupied):
            radii.append(float(np.max(np.hypot(x_grid[occupied], y_grid[occupied]))))
    if not radii:
        return 1.0
    return float(min(1.0, max(0.32, max(radii) + 0.08)))


def _paper_axis(axis, *, direction: str = "in") -> None:
    axis.tick_params(which="both", top=True, right=True, direction=direction, pad=5)
    for spine in axis.spines.values():
        spine.set_linewidth(1.15)
        spine.set_color("0.18")


def _map_ticks(crop: float) -> list[float]:
    if crop <= 0.38:
        candidates = (-0.2, 0.0, 0.2)
    elif crop <= 0.58:
        candidates = (-0.4, 0.0, 0.4)
    else:
        candidates = (-0.6, -0.3, 0.0, 0.3, 0.6)
    ticks = [tick for tick in candidates if abs(tick) <= 0.88 * crop]
    if 0.0 not in ticks:
        ticks.insert(len(ticks) // 2, 0.0)
    return ticks


def _style_map_axis(
    axis,
    crop: float,
    *,
    xlabel: bool,
    ylabel: bool,
    xticklabels: bool,
    yticklabels: bool,
) -> None:
    ticks = _map_ticks(crop)
    axis.set(xlim=(-crop, crop), ylim=(-crop, crop), aspect="equal")
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    if xlabel:
        axis.set_xlabel(r"$x$", labelpad=8)
    else:
        axis.set_xlabel("")
    if ylabel:
        axis.set_ylabel(r"$y$", labelpad=8)
    else:
        axis.set_ylabel("")
    if not xticklabels:
        axis.set_xticklabels([])
    if not yticklabels:
        axis.set_yticklabels([])
    _paper_axis(axis, direction="out")


def _metric_text(relative_l2: float, cosine: float, *, reference: bool) -> str:
    if reference:
        return ""
    cosine_text = "—" if not np.isfinite(cosine) else f"{cosine:.3f}"
    if relative_l2 >= 0.1:
        l2_text = f"{relative_l2:.2f}"
    else:
        l2_text = f"{relative_l2:.2g}"
    return rf"$L^2={l2_text}$" + "\n" + rf"$\mathrm{{cos}}={cosine_text}$"


def _panel_letter(axis, letter: str) -> None:
    """Nature/APS panel letter inside the data, with a white halo instead of a box."""
    tag = axis.text(
        0.045,
        0.955,
        f"({letter})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="0.05",
        zorder=6,
    )
    tag.set_path_effects(_HALO)


# Cases 2 and 4 use the measured t x 6.6 validation point, where their O(t^3)
# density responses clear the sampling floor. The other mechanisms retain the
# earlier times at which their micro/PDE/Fourier agreement is strongest.
_SUMMARY_TARGET_TIMES = {
    2: 0.00330,
    3: 0.000495,
    4: 0.00330,
    5: 0.000430,
    6: 0.000375,
    7: 0.000490,
    8: 0.000480,
    9: 0.000410,
}


def _column_header(axis, heading: str, metric: str) -> None:
    """Reserved header cell: method name on top, metrics stacked below.

    Metrics stay inside this column, so they cannot collide with a neighbor
    or sit on the heatmap.
    """
    axis.set_axis_off()
    if heading:
        axis.text(
            0.5,
            1.0,
            heading,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="0.08",
        )
        metric_y = 0.08
        metric_va = "bottom"
    else:
        metric_y = 0.45
        metric_va = "center"
    if not metric:
        return
    axis.text(
        0.5,
        metric_y,
        metric,
        transform=axis.transAxes,
        ha="center",
        va=metric_va,
        fontsize=13,
        color="0.22",
        linespacing=1.2,
    )


def _map_row(figure, n_methods: int, *, header_ratio: float):
    spec = figure.add_gridspec(
        2,
        n_methods + 1,
        height_ratios=[header_ratio, 1.0],
        width_ratios=[1.0] * n_methods + [0.065],
        hspace=0.05,
        wspace=0.10,
    )
    headers = [figure.add_subplot(spec[0, column]) for column in range(n_methods)]
    axes = [figure.add_subplot(spec[1, column]) for column in range(n_methods)]
    cax = figure.add_subplot(spec[1, -1])
    return headers, axes, cax


def _nice_limit(limit: float) -> float:
    if not np.isfinite(limit) or limit <= 0.0:
        return 1.0
    exponent = int(np.floor(np.log10(limit)))
    frac = limit / 10.0**exponent
    for candidate in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if frac <= candidate + 1e-12:
            return float(candidate * 10.0**exponent)
    return float(10.0 ** (exponent + 1))


def _decade(limit: float) -> int:
    if not np.isfinite(limit) or limit <= 0.0:
        return 0
    exponent = int(np.floor(np.log10(limit)))
    if -1 <= exponent <= 0:
        return 0
    return exponent


def _format_scaled_tick(value: float, scale: float) -> str:
    scaled = value / scale if scale else value
    if abs(scaled) < 1.0e-10:
        return "0"
    rounded = round(float(scaled), 8)
    if abs(rounded - round(rounded)) < 1.0e-8:
        return f"{round(rounded)}"
    return f"{rounded:g}"


def _colorbar(figure, image, axes, symbol: str, limit: float, *, cax=None) -> None:
    if cax is None:
        colorbar = figure.colorbar(
            image,
            ax=axes,
            location="right",
            fraction=0.046,
            pad=0.03,
            aspect=22,
            shrink=0.92,
        )
    else:
        colorbar = figure.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=14, width=1.15, length=5.0, pad=4)
    colorbar.outline.set_linewidth(1.15)
    exponent = _decade(limit)
    scale = 10.0**exponent if exponent else 1.0
    vmin = float(getattr(image.norm, "vmin", 0.0))
    if vmin < 0.0:
        ticks = np.array([-limit, -0.5 * limit, 0.0, 0.5 * limit, limit])
    else:
        ticks = np.array([0.0, 0.5 * limit, limit])
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([_format_scaled_tick(tick, scale) for tick in ticks])
    if exponent != 0:
        colorbar.set_label(
            rf"{symbol}  ($\times 10^{{{exponent}}}$)",
            fontsize=16,
            labelpad=12,
        )
    else:
        colorbar.set_label(symbol, fontsize=16, labelpad=12)


def _landscape_guides(axis, field, crop: float) -> None:
    """Mark where the nutrient is richest and poorest, and nothing else.

    A single cosine mode has straight iso-lines, so a generic contour overlay
    draws a picket fence of parallel lines that reads as an artifact and hides
    the response underneath.  Two families of line carry all the information a
    reader needs: the crests, where the cell finds most nutrient, and the
    troughs, where it finds least.  Any other landscape falls back to ordinary
    contours.
    """

    wavevector = getattr(field, "wavevector", None)
    if wavevector is None:
        span = np.linspace(-crop, crop, 201)
        x_grid, y_grid = np.meshgrid(span, span, indexing="xy")
        values = np.asarray(field.value(np.stack((x_grid, y_grid), axis=-1)), float)
        axis.contour(
            x_grid,
            y_grid,
            values,
            levels=3,
            colors=NUTRIENT,
            linewidths=0.5,
            alpha=0.8,
            zorder=2,
        )
        return

    wavevector = np.asarray(wavevector, dtype=float)
    wavenumber = float(np.hypot(*wavevector))
    unit = wavevector / wavenumber
    period = 2.0 * np.pi / wavenumber
    phase = float(getattr(field, "phase", 0.0))
    reach = crop * np.sqrt(2.0) + period
    # Offsets along q where q.x + phase is a multiple of pi: even multiples are
    # crests, odd ones troughs.
    first = int(np.floor((-reach * wavenumber - phase) / np.pi))
    last = int(np.ceil((reach * wavenumber - phase) / np.pi))
    normal = np.array([-unit[1], unit[0]])
    line = np.linspace(-reach, reach, 2)
    for index in range(first, last + 1):
        offset = (index * np.pi - phase) / wavenumber
        centre = offset * unit
        points = centre[None, :] + line[:, None] * normal[None, :]
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=NUTRIENT,
            lw=0.7,
            ls="-" if index % 2 == 0 else (0, (2.4, 2.0)),
            alpha=0.85,
            zorder=2,
        )


def _has_polarity(results: list[tuple[str, PositionPDFResult]]) -> bool:
    return all(
        result.polarity_x is not None and result.polarity_y is not None
        for _, result in results
    )


def _extent(coordinates: NDArray[np.float64]) -> tuple[float, float, float, float]:
    return (
        float(coordinates[0]),
        float(coordinates[-1]),
        float(coordinates[0]),
        float(coordinates[-1]),
    )


def _render_remaining_cases_summary(
    ordered: list[tuple[str, PositionPDFResult]],
    config: ProjectConfig,
    output_root: Path,
) -> tuple[Path, Path]:
    """Render Cases 2--9 as one space-filling, single-page comparison."""

    if not _has_polarity(ordered):
        raise ValueError("the Cases 2--9 summary requires polarity from every method")
    cases = [case for case in range(2, 10) if case in ordered[0][1].case_numbers]
    if not cases:
        raise ValueError("the Cases 2--9 summary needs at least one remaining case")
    methods = ordered[:3]
    if len(methods) != 3:
        raise ValueError("the Cases 2--9 summary requires exactly three methods")

    final_time = _common_final_time(ordered)
    field = create_field(config.field)
    figure = Figure(figsize=(8.3, 10.6), dpi=180)
    FigureCanvasAgg(figure)
    spec = figure.add_gridspec(
        len(cases),
        8,
        left=0.105,
        right=0.985,
        bottom=0.035,
        top=0.915,
        hspace=0.075,
        wspace=0.055,
        width_ratios=(1, 1, 1, 0.07, 1, 1, 1, 0.07),
    )
    axes = [
        [figure.add_subplot(spec[row, column]) for column in range(8)]
        for row in range(len(cases))
    ]
    headings = [PAPER_COLUMN_LABELS.get(name, name) for name, _ in methods] * 2
    for column, heading in zip((0, 1, 2, 4, 5, 6), headings, strict=True):
        axes[0][column].set_title(heading, fontsize=10.5, fontweight="bold", pad=5)
    figure.text(
        0.31,
        0.986,
        r"Density response  $\rho_i-\rho_0$",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.77,
        0.986,
        r"Polarity response  $|\mathbf{p}_i-\mathbf{p}_0|$",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    for row, case_number in enumerate(cases):
        case_index = ordered[0][1].case_numbers.index(case_number)
        target_time = min(final_time, _SUMMARY_TARGET_TIMES[case_number])
        density_slices: list[tuple[PositionPDFResult, NDArray[np.float64]]] = []
        polarity_slices: list[tuple[PositionPDFResult, NDArray[np.float64]]] = []
        density_values: list[NDArray[np.float64]] = []
        polarity_values: list[NDArray[np.float64]] = []
        for _, result in methods:
            index = _nearest_time_index(result, target_time)
            density = _display_density(result, case_index, index)
            px, py = _display_polarity(result, case_index, index)
            magnitude = np.hypot(px, py)
            _, _, mask = _disk_mesh(result.coordinates)
            density_slices.append((result, density))
            polarity_slices.append((result, magnitude))
            density_values.append(density[mask])
            polarity_values.append(magnitude[mask])
        density_limit = _nice_limit(
            max(
                float(
                    np.percentile(
                        np.abs(np.concatenate(density_values)),
                        config.density.difference_case_percentile,
                    )
                ),
                np.finfo(float).eps,
            )
        )
        polarity_limit = _nice_limit(
            max(
                float(
                    np.percentile(
                        np.concatenate(polarity_values),
                        config.density.difference_case_percentile,
                    )
                ),
                np.finfo(float).eps,
            )
        )
        density_norm = TwoSlopeNorm(vmin=-density_limit, vcenter=0, vmax=density_limit)
        polarity_norm = Normalize(0, polarity_limit)
        crop = _cloud_crop(ordered, case_index, target_time)
        density_image = None
        polarity_image = None

        for column, (result, values) in enumerate(density_slices):
            axis = axes[row][column]
            _, _, mask = _disk_mesh(result.coordinates)
            density_image = axis.imshow(
                _masked(values, mask),
                origin="lower",
                extent=_extent(result.coordinates),
                cmap=_DIFFERENCE_CMAP,
                norm=density_norm,
                interpolation="bilinear",
                rasterized=True,
            )
            _landscape_guides(axis, field, crop)
            axis.set(xlim=(-crop, crop), ylim=(-crop, crop), aspect="equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("0.25")
        for offset, (result, values) in enumerate(polarity_slices):
            axis = axes[row][4 + offset]
            _, _, mask = _disk_mesh(result.coordinates)
            polarity_image = axis.imshow(
                _masked(values, mask),
                origin="lower",
                extent=_extent(result.coordinates),
                cmap="cividis",
                norm=polarity_norm,
                interpolation="bilinear",
                rasterized=True,
            )
            axis.set(xlim=(-crop, crop), ylim=(-crop, crop), aspect="equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("0.25")

        time_exponent = int(np.floor(np.log10(target_time)))
        time_coefficient = target_time / 10.0**time_exponent
        label = (
            f"Case {case_number}  {CASES[case_number].math_label}\n"
            f"$t={time_coefficient:.2f}\\times10^{{{time_exponent}}}$"
        )
        axes[row][0].text(
            -0.08,
            0.5,
            label,
            transform=axes[row][0].transAxes,
            ha="right",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            linespacing=1.25,
        )
        for cax, image, limit in (
            (axes[row][3], density_image, density_limit),
            (axes[row][7], polarity_image, polarity_limit),
        ):
            colorbar = figure.colorbar(image, cax=cax)
            colorbar.set_ticks([])
            colorbar.outline.set_linewidth(0.5)
            exponent = int(np.floor(np.log10(limit)))
            coefficient = limit / 10.0**exponent
            prefix = r"$\pm$" if image is density_image else ""
            colorbar.ax.set_title(
                rf"{prefix}{coefficient:g}$\times10^{{{exponent}}}$",
                fontsize=5.8,
                pad=1.5,
            )

    stem = output_root / "figure-s01-cases-2-9-summary"
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    figure.savefig(pdf_path, facecolor="white", dpi=300)
    figure.savefig(png_path, facecolor="white", dpi=300)
    figure.clear()
    return pdf_path, png_path


def _render_shared_scale_summary(
    ordered: list[tuple[str, PositionPDFResult]],
    config: ProjectConfig,
    output_root: Path,
) -> tuple[Path, Path]:
    """Render Cases 2--9 at one time with one scale per observable."""

    cases = [case for case in range(2, 10) if case in ordered[0][1].case_numbers]
    methods = ordered[:3]
    if len(methods) != 3 or not _has_polarity(methods):
        raise ValueError("the shared-scale summary requires three polar methods")
    final_time = min(_common_final_time(methods), PAPER_REPORT_TIME)
    field = create_field(config.field)
    cached: dict[int, tuple[list, list]] = {}
    for case_number in cases:
        case_index = ordered[0][1].case_numbers.index(case_number)
        density_slices = []
        polarity_slices = []
        for _, result in methods:
            index = _nearest_time_index(result, final_time)
            density = _display_density(result, case_index, index)
            px, py = _display_polarity(result, case_index, index)
            magnitude = np.hypot(px, py)
            _, _, mask = _disk_mesh(result.coordinates)
            density_slices.append((result, density))
            polarity_slices.append((result, magnitude))
        cached[case_number] = (density_slices, polarity_slices)

    percentile = config.density.difference_case_percentile
    synchronized_cases = tuple(range(1, 10))
    density_limit, polarity_limit = shared_paper_limits(
        methods, synchronized_cases, final_time, percentile
    )
    density_norm = TwoSlopeNorm(vmin=-density_limit, vcenter=0.0, vmax=density_limit)
    polarity_norm = Normalize(vmin=0.0, vmax=polarity_limit)
    crop = shared_paper_crop(methods, synchronized_cases, final_time)

    figure = Figure(figsize=(8.3, 10.6), dpi=180)
    FigureCanvasAgg(figure)
    spec = figure.add_gridspec(
        len(cases),
        6,
        left=0.105,
        right=0.985,
        bottom=0.11,
        top=0.89,
        hspace=0.07,
        wspace=0.055,
    )
    axes = [
        [figure.add_subplot(spec[row, column]) for column in range(6)]
        for row in range(len(cases))
    ]
    headings = [PAPER_COLUMN_LABELS.get(name, name) for name, _ in methods] * 2
    for column, heading in enumerate(headings):
        axes[0][column].set_title(heading, fontsize=10.5, fontweight="bold", pad=5)
    exponent = int(np.floor(np.log10(final_time)))
    coefficient = final_time / 10.0**exponent
    figure.text(
        0.545,
        0.992,
        rf"Common reporting time  $t={coefficient:.2f}\times10^{{{exponent}}}$",
        ha="center",
        va="top",
        fontsize=10.5,
    )
    figure.text(
        0.31,
        0.958,
        r"Density response  $\rho_i-\rho_0$",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.77,
        0.958,
        r"Polarity response  $|\mathbf{p}_i-\mathbf{p}_0|$",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    density_image = None
    polarity_image = None
    for row, case_number in enumerate(cases):
        density_slices, polarity_slices = cached[case_number]
        for column, (result, values) in enumerate(density_slices):
            axis = axes[row][column]
            _, _, mask = _disk_mesh(result.coordinates)
            density_image = axis.imshow(
                _masked(values, mask),
                origin="lower",
                extent=_extent(result.coordinates),
                cmap=_DIFFERENCE_CMAP,
                norm=density_norm,
                interpolation="bilinear",
                rasterized=True,
            )
            _landscape_guides(axis, field, crop)
        for offset, (result, values) in enumerate(polarity_slices):
            axis = axes[row][3 + offset]
            _, _, mask = _disk_mesh(result.coordinates)
            polarity_image = axis.imshow(
                _masked(values, mask),
                origin="lower",
                extent=_extent(result.coordinates),
                cmap="cividis",
                norm=polarity_norm,
                interpolation="bilinear",
                rasterized=True,
            )
        for axis in axes[row]:
            axis.set(xlim=(-crop, crop), ylim=(-crop, crop), aspect="equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("0.25")
        axes[row][0].text(
            -0.08,
            0.5,
            f"Case {case_number}  {CASES[case_number].math_label}",
            transform=axes[row][0].transAxes,
            ha="right",
            va="center",
            fontsize=7.8,
            fontweight="bold",
        )

    density_cax = figure.add_axes((0.135, 0.045, 0.35, 0.014))
    polarity_cax = figure.add_axes((0.595, 0.045, 0.35, 0.014))
    density_bar = figure.colorbar(
        density_image, cax=density_cax, orientation="horizontal"
    )
    density_bar.set_ticks((-density_limit, 0, density_limit))
    density_bar.set_label(
        r"Shared linear scale: $\rho_i-\rho_0$", fontsize=8, labelpad=2
    )
    polarity_bar = figure.colorbar(
        polarity_image, cax=polarity_cax, orientation="horizontal"
    )
    polarity_bar.set_ticks((0, polarity_limit / 2, polarity_limit))
    polarity_bar.set_label(
        r"Shared linear scale: $|\mathbf{p}_i-\mathbf{p}_0|$", fontsize=8, labelpad=2
    )
    for colorbar in (density_bar, polarity_bar):
        colorbar.ax.tick_params(labelsize=6.8, length=2.5, width=0.6, pad=1.5)
        colorbar.outline.set_linewidth(0.6)

    stem = output_root / "figure-s01-cases-2-9-summary"
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    figure.savefig(pdf_path, facecolor="white", dpi=300)
    figure.savefig(png_path, facecolor="white", dpi=300)
    figure.clear()
    return pdf_path, png_path


def _render_manuscript_strategy_figures_legacy(
    results: dict[str, PositionPDFResult],
    config: ProjectConfig,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Write one full-page PDF (and PNG preview) per response case 1--9."""

    ordered = _paper_methods(results)
    reference_name, reference = ordered[0]
    final_time = _common_final_time(ordered)
    case_one_time = min(final_time, 5.0e-4)
    field = create_field(config.field)
    show_polarity = _has_polarity(ordered)
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    n_methods = len(ordered)
    n_map_rows = 2 if show_polarity else 1
    reference_coordinates = reference.coordinates
    _, _, reference_mask = _disk_mesh(reference_coordinates)
    letters = "abcdefghijklmnopqrstuvwxyz"

    with mpl.rc_context(_RC):
        for case_index, case_number in enumerate(reference.case_numbers):
            if case_number == 0:
                continue
            # Case 1 remains the main-text validation sheet. Cases 2--9 are
            # rendered together below so the appendix comparison occupies one
            # page instead of eight mostly redundant pages.
            if case_number != 1:
                continue
            get_case(case_number)  # Validate the archived case identifier.
            density_slices: dict[str, NDArray[np.float64]] = {}
            polarity_slices: dict[
                str, tuple[NDArray[np.float64], NDArray[np.float64]]
            ] = {}
            density_values: list[NDArray[np.float64]] = []
            polarity_magnitudes: list[NDArray[np.float64]] = []
            for method, result in ordered:
                time_index = _nearest_time_index(result, case_one_time)
                density = _display_density(result, case_index, time_index)
                density_slices[method] = density
                _, _, mask = _disk_mesh(result.coordinates)
                density_values.append(density[mask])
                if show_polarity:
                    vector = _display_polarity(result, case_index, time_index)
                    polarity_slices[method] = vector
                    polarity_magnitudes.append(np.hypot(vector[0], vector[1])[mask])
            density_limit = _nice_limit(
                max(
                    float(
                        np.percentile(
                            np.abs(np.concatenate(density_values)),
                            config.density.difference_case_percentile,
                        )
                    ),
                    np.finfo(float).eps,
                )
            )
            density_norm = TwoSlopeNorm(
                vmin=-density_limit, vcenter=0.0, vmax=density_limit
            )
            polarity_limit = 0.0
            polarity_norm = None
            if polarity_magnitudes:
                polarity_limit = _nice_limit(
                    max(
                        float(
                            np.percentile(
                                np.concatenate(polarity_magnitudes),
                                config.density.difference_case_percentile,
                            )
                        ),
                        np.finfo(float).eps,
                    )
                )
                polarity_norm = Normalize(0.0, polarity_limit)
            crop = _cloud_crop(ordered, case_index, case_one_time)
            figure = Figure(figsize=(7.4, 5.4), layout="constrained", dpi=180)
            FigureCanvasAgg(figure)
            engine = figure.get_layout_engine()
            if engine is not None:
                engine.set(w_pad=0.03, h_pad=0.03, wspace=0.05, hspace=0.06)

            body = figure.subfigures(n_map_rows, 1, hspace=0.04)
            density_fig = body[0]
            polarity_fig = body[1] if show_polarity else None

            density_headers, density_axes, density_cax = _map_row(
                density_fig, n_methods, header_ratio=0.34
            )
            polarity_headers: list = []
            polarity_axes: list = []
            polarity_cax = None
            if polarity_fig is not None:
                polarity_headers, polarity_axes, polarity_cax = _map_row(
                    polarity_fig, n_methods, header_ratio=0.22
                )

            letter_index = 0
            reference_index = _nearest_time_index(reference, case_one_time)
            reference_density = _display_density(reference, case_index, reference_index)
            density_image = None
            polarity_image = None
            for column, (method, result) in enumerate(ordered):
                axis = density_axes[column]
                values = density_slices[method]
                x_grid, y_grid, mask = _disk_mesh(result.coordinates)
                density_image = axis.imshow(
                    _masked(values, mask),
                    origin="lower",
                    extent=_extent(result.coordinates),
                    cmap=_DIFFERENCE_CMAP,
                    norm=density_norm,
                    interpolation="bilinear",
                    rasterized=True,
                )
                _landscape_guides(axis, field, crop)
                if method == reference_name:
                    metric = ""
                else:
                    candidate = _on_reference_grid(
                        values, result.coordinates, reference_coordinates
                    )
                    relative_l2, cosine, _ = _agreement_metrics(
                        reference_density, candidate, reference_mask
                    )
                    metric = _metric_text(relative_l2, cosine, reference=False)
                _column_header(
                    density_headers[column],
                    PAPER_COLUMN_LABELS.get(method, method),
                    metric,
                )
                _panel_letter(axis, letters[letter_index])
                _style_map_axis(
                    axis,
                    crop,
                    xlabel=not show_polarity,
                    ylabel=column == 0,
                    xticklabels=not show_polarity,
                    yticklabels=column == 0,
                )
                letter_index += 1

            if show_polarity and polarity_fig is not None:
                reference_px, reference_py = _display_polarity(
                    reference, case_index, reference_index
                )
                for column, (method, result) in enumerate(ordered):
                    axis = polarity_axes[column]
                    vector_x, vector_y = polarity_slices[method]
                    magnitude = np.hypot(vector_x, vector_y)
                    x_grid, y_grid, mask = _disk_mesh(result.coordinates)
                    polarity_image = axis.imshow(
                        _masked(magnitude, mask),
                        origin="lower",
                        extent=_extent(result.coordinates),
                        cmap="cividis",
                        norm=polarity_norm,
                        interpolation="bilinear",
                        rasterized=True,
                    )
                    stride = max(1, len(result.coordinates) // 7)
                    if polarity_limit > 1.0e-18:
                        sampled_mag = magnitude[::stride, ::stride]
                        sampled = mask[::stride, ::stride] & (
                            sampled_mag > 0.12 * polarity_limit
                        )
                        safe = np.maximum(sampled_mag, np.finfo(float).eps)
                        axis.quiver(
                            x_grid[::stride, ::stride],
                            y_grid[::stride, ::stride],
                            np.where(
                                sampled, vector_x[::stride, ::stride] / safe, np.nan
                            ),
                            np.where(
                                sampled, vector_y[::stride, ::stride] / safe, np.nan
                            ),
                            color="white",
                            edgecolor="0.15",
                            linewidth=0.35,
                            angles="xy",
                            scale_units="xy",
                            scale=9.5,
                            width=0.0085,
                            headwidth=3.4,
                            headlength=4.2,
                            pivot="mid",
                            minlength=0.0,
                            zorder=3,
                        )
                    if method == reference_name:
                        metric = ""
                    else:
                        candidate_x = _on_reference_grid(
                            vector_x, result.coordinates, reference_coordinates
                        )
                        candidate_y = _on_reference_grid(
                            vector_y, result.coordinates, reference_coordinates
                        )
                        relative_l2, cosine, _ = _vector_agreement_metrics(
                            reference_px,
                            reference_py,
                            candidate_x,
                            candidate_y,
                            reference_mask,
                        )
                        metric = _metric_text(relative_l2, cosine, reference=False)
                    _column_header(polarity_headers[column], "", metric)
                    _panel_letter(axis, letters[letter_index])
                    _style_map_axis(
                        axis,
                        crop,
                        xlabel=True,
                        ylabel=column == 0,
                        xticklabels=True,
                        yticklabels=column == 0,
                    )
                    letter_index += 1

            if density_image is not None:
                _colorbar(
                    density_fig,
                    density_image,
                    density_axes,
                    rf"$\rho_{{{case_number}}}-\rho_0$",
                    density_limit,
                    cax=density_cax,
                )
            if polarity_image is not None and polarity_fig is not None:
                _colorbar(
                    polarity_fig,
                    polarity_image,
                    polarity_axes,
                    rf"$|\mathbf{{p}}_{{{case_number}}}-\mathbf{{p}}_0|$",
                    polarity_limit,
                    cax=polarity_cax,
                )

            stem = output_root / STRATEGY_FIGURE_STEMS[case_number]
            pdf_path = stem.with_suffix(".pdf")
            png_path = stem.with_suffix(".png")
            figure.savefig(pdf_path, facecolor="white", dpi=300)
            figure.savefig(png_path, dpi=300, facecolor="white")
            figure.clear()
            written.append(pdf_path)
            written.append(png_path)

        if len(ordered) == 3 and any(
            case in reference.case_numbers for case in range(2, 10)
        ):
            written.extend(_render_shared_scale_summary(ordered, config, output_root))

    if not written:
        known = ", ".join(str(number) for number in sorted(CASES) if number)
        raise ValueError(
            "manuscript strategy figures need at least one of the response "
            f"cases {known}; the supplied results contain only the baseline"
        )
    return tuple(written)


def _render_setup_and_case_one(
    ordered: list[tuple[str, PositionPDFResult]],
    config: ProjectConfig,
    output_root: Path,
) -> tuple[Path, Path]:
    """Landscape and Case-1 validation in one column-width 2x2 block.

    Panel (a) is the nutrient landscape the whole comparison runs on; panels
    (b)-(d) are the three descriptions, each split into density and polarity.
    Putting the background beside the responses lets a reader check the sign
    of the response against the crest lines without turning the page.
    """

    methods = ordered[:3]
    if len(methods) < 3 or not _has_polarity(methods):
        raise ValueError("the Case-1 figure requires three polar methods")
    case_number = 1
    case_index = methods[0][1].case_numbers.index(case_number)
    time = min(_common_final_time(methods), PAPER_REPORT_TIME)
    field = create_field(config.field)

    slices = []
    for _, result in methods:
        index = _nearest_time_index(result, time)
        density = _display_density(result, case_index, index)
        px, py = _display_polarity(result, case_index, index)
        slices.append((result, density, np.hypot(px, py)))

    percentile = config.density.difference_case_percentile
    synchronized_cases = tuple(range(1, 10))
    density_limit, polarity_limit = shared_paper_limits(
        methods, synchronized_cases, time, percentile
    )
    density_norm = TwoSlopeNorm(vmin=-density_limit, vcenter=0.0, vmax=density_limit)
    polarity_norm = Normalize(0.0, polarity_limit)
    crop = shared_paper_crop(methods, synchronized_cases, time)

    figure = Figure(figsize=(3.4, 2.95), dpi=180)
    FigureCanvasAgg(figure)
    outer = figure.add_gridspec(
        2,
        2,
        left=0.035,
        right=0.985,
        bottom=0.160,
        top=0.900,
        hspace=0.26,
        wspace=0.14,
    )

    # ---- (a) the landscape: map, and the cut that fixes the phase ----
    setup = outer[0, 0].subgridspec(1, 2, wspace=0.06)
    axis = figure.add_subplot(setup[0, 0])
    span = np.linspace(-crop, crop, 320)
    xg, yg = np.meshgrid(span, span, indexing="xy")
    eta = np.asarray(field.value(np.stack((xg, yg), axis=-1)), float)
    axis.imshow(
        eta,
        origin="lower",
        extent=(-crop, crop, -crop, crop),
        cmap="BrBG",
        interpolation="bilinear",
        rasterized=True,
    )
    _landscape_guides(axis, field, crop)
    sigma = resolve_initial_gaussian_sigma(
        config.density, inferred_characteristic_length(config, field)
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 181)
    axis.plot(
        2.0 * sigma * np.cos(angle),
        2.0 * sigma * np.sin(angle),
        color="#d1490b",
        lw=0.9,
        zorder=5,
    )
    axis.set_title(
        r"nutrient  $\eta$", fontsize=6.6, color=BLACK, pad=2.5, loc="left", x=0.0
    )

    # The phase splits the modulation into equal odd and even parts about the
    # release, which is what lets one release excite centroid and spread at
    # once; showing the cut here keeps that choice in front of the reader.
    cut = figure.add_subplot(setup[0, 1])
    wavevector = np.asarray(getattr(field, "wavevector", (1.0, 0.0)), float)
    wavenumber = float(np.hypot(*wavevector))
    unit = wavevector / wavenumber
    distance = np.linspace(-crop, crop, 400)
    points = distance[:, None] * unit[None, :]
    values = np.asarray(field.value(points), float)
    values = values - float(np.mean(values))
    even = 0.5 * (values + values[::-1])
    odd = 0.5 * (values - values[::-1])
    cut.plot(distance, values, color=BLACK, lw=0.9, zorder=3)
    cut.plot(distance, even, color="#1f77b4", lw=0.7, ls=(0, (3.0, 1.6)), zorder=2)
    cut.plot(distance, odd, color="#2ca25f", lw=0.7, ls=(0, (1.2, 1.4)), zorder=2)
    cut.axhline(0.0, color="0.75", lw=0.5, zorder=1)
    cut.set_xlim(-crop, crop)
    cut.set_box_aspect(1.0)  # match the square map beside it
    cut.set_xticks([])
    cut.set_yticks([])
    for spine in cut.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.30")
    cut.set_title(r"cut along $\hat q$", fontsize=6.0, color=GRAY, pad=2.5)
    cut.text(
        0.03,
        0.97,
        "even",
        transform=cut.transAxes,
        fontsize=4.8,
        color="#1f77b4",
        ha="left",
        va="top",
    )
    cut.text(
        0.97,
        0.97,
        "odd",
        transform=cut.transAxes,
        fontsize=4.8,
        color="#2ca25f",
        ha="right",
        va="top",
    )

    # ---- (b)-(d) the three descriptions, each density | polarity ----
    density_image = polarity_image = None
    method_axes = [axis]
    for position, (result, density, magnitude) in enumerate(slices):
        row, column = divmod(position + 1, 2)
        inner = outer[row, column].subgridspec(1, 2, wspace=0.06)
        pair = [figure.add_subplot(inner[0, side]) for side in range(2)]
        _, _, mask = _disk_mesh(result.coordinates)
        density_image = pair[0].imshow(
            _masked(density, mask),
            origin="lower",
            extent=_extent(result.coordinates),
            cmap=_DIFFERENCE_CMAP,
            norm=density_norm,
            interpolation="bilinear",
            rasterized=True,
        )
        _landscape_guides(pair[0], field, crop)
        polarity_image = pair[1].imshow(
            _masked(magnitude, mask),
            origin="lower",
            extent=_extent(result.coordinates),
            cmap="cividis",
            norm=polarity_norm,
            interpolation="bilinear",
            rasterized=True,
        )
        name = PAPER_COLUMN_LABELS.get(methods[position][0], methods[position][0])
        pair[0].set_title(
            name,
            fontsize=6.6,
            color=BLACK,
            pad=2.5,
            fontweight="bold",
            loc="left",
            x=0.0,
        )
        method_axes.extend(pair)

    for a in method_axes:
        a.set(xlim=(-crop, crop), ylim=(-crop, crop), aspect="equal")
        a.set_xticks([])
        a.set_yticks([])
        for spine in a.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("0.30")

    for letter, a in zip(
        "abcd", (method_axes[0], method_axes[1], method_axes[3], method_axes[5])
    ):
        a.annotate(
            f"({letter})",
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0, 11),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color=BLACK,
            ha="left",
            va="center",
            annotation_clip=False,
        )

    density_cax = figure.add_axes((0.115, 0.055, 0.34, 0.026))
    polarity_cax = figure.add_axes((0.565, 0.055, 0.34, 0.026))
    density_bar = figure.colorbar(
        density_image, cax=density_cax, orientation="horizontal"
    )
    density_bar.set_ticks((-density_limit, 0.0, density_limit))
    density_bar.set_label(r"$\rho_1-\rho_0$", fontsize=6, labelpad=1)
    polarity_bar = figure.colorbar(
        polarity_image, cax=polarity_cax, orientation="horizontal"
    )
    polarity_bar.set_ticks((0.0, polarity_limit))
    polarity_bar.set_label(r"$|\mathbf{p}_1-\mathbf{p}_0|$", fontsize=6, labelpad=1)
    for colorbar in (density_bar, polarity_bar):
        colorbar.ax.tick_params(labelsize=5.2, length=1.8, width=0.5, pad=1.0)
        colorbar.outline.set_linewidth(0.5)

    stem = output_root / STRATEGY_FIGURE_STEMS[case_number]
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    figure.savefig(pdf_path, facecolor="white", dpi=300)
    figure.savefig(png_path, facecolor="white", dpi=300)
    figure.clear()
    return pdf_path, png_path


def _render_case_one_shared_style(
    ordered: list[tuple[str, PositionPDFResult]],
    config: ProjectConfig,
    output_root: Path,
) -> tuple[Path, Path]:
    """Render the main validation figure in the summary figure's visual style."""

    methods = ordered[:3]
    if len(methods) < 2 or not _has_polarity(methods):
        raise ValueError("the Case-1 figure requires at least two polar methods")
    method_count = len(methods)
    case_number = 1
    case_index = methods[0][1].case_numbers.index(case_number)
    time = min(_common_final_time(methods), PAPER_REPORT_TIME)
    field = create_field(config.field)
    density_slices = []
    polarity_slices = []
    for _, result in methods:
        index = _nearest_time_index(result, time)
        density = _display_density(result, case_index, index)
        px, py = _display_polarity(result, case_index, index)
        magnitude = np.hypot(px, py)
        density_slices.append((result, density))
        polarity_slices.append((result, magnitude))

    percentile = config.density.difference_case_percentile
    synchronized_cases = tuple(range(1, 10))
    density_limit, polarity_limit = shared_paper_limits(
        methods, synchronized_cases, time, percentile
    )
    density_norm = TwoSlopeNorm(vmin=-density_limit, vcenter=0.0, vmax=density_limit)
    polarity_norm = Normalize(0.0, polarity_limit)
    crop = shared_paper_crop(methods, synchronized_cases, time)

    # One manuscript column wide: methods run down the rows and the two
    # observables across the columns, so each panel keeps a usable size
    # instead of being scaled down by \includegraphics.
    figure = Figure(figsize=(3.4, 4.5), dpi=180)
    FigureCanvasAgg(figure)
    spec = figure.add_gridspec(
        method_count,
        2,
        left=0.150,
        right=0.985,
        bottom=0.135,
        top=0.920,
        hspace=0.055,
        wspace=0.055,
    )
    # axes[row][column] keeps the old meaning -- row 0 density, row 1
    # polarity -- while the transposed grid puts methods down the page.
    axes = [
        [figure.add_subplot(spec[column, row]) for column in range(method_count)]
        for row in range(2)
    ]
    axes[0][0].set_title("Density", fontsize=8.5, fontweight="bold", pad=3)
    axes[1][0].set_title("Polarity", fontsize=8.5, fontweight="bold", pad=3)
    for index, (name, _) in enumerate(methods):
        centre = (
            axes[0][index].get_position().y0 + axes[0][index].get_position().y1
        ) / 2.0
        figure.text(
            0.045,
            centre,
            PAPER_COLUMN_LABELS.get(name, name),
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

    density_image = None
    polarity_image = None
    for column, (result, density) in enumerate(density_slices):
        axis = axes[0][column]
        _, _, mask = _disk_mesh(result.coordinates)
        density_image = axis.imshow(
            _masked(density, mask),
            origin="lower",
            extent=_extent(result.coordinates),
            cmap=_DIFFERENCE_CMAP,
            norm=density_norm,
            interpolation="bilinear",
            rasterized=True,
        )
        _landscape_guides(axis, field, crop)
    for column, (result, magnitude) in enumerate(polarity_slices):
        axis = axes[1][column]
        _, _, mask = _disk_mesh(result.coordinates)
        polarity_image = axis.imshow(
            _masked(magnitude, mask),
            origin="lower",
            extent=_extent(result.coordinates),
            cmap="cividis",
            norm=polarity_norm,
            interpolation="bilinear",
            rasterized=True,
        )
    for row in range(2):
        for column in range(method_count):
            axis = axes[row][column]
            axis.set(xlim=(-crop, crop), ylim=(-crop, crop), aspect="equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("0.25")
            axis.text(
                0.035,
                0.965,
                "abcdef"[method_count * row + column],
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                fontweight="bold",
                color="black" if row == 0 else "white",
            )

    density_cax = figure.add_axes((0.175, 0.062, 0.31, 0.016))
    polarity_cax = figure.add_axes((0.655, 0.062, 0.31, 0.016))
    density_bar = figure.colorbar(
        density_image, cax=density_cax, orientation="horizontal"
    )
    density_bar.set_ticks((-density_limit, 0.0, density_limit))
    density_bar.set_label(r"$\rho_1-\rho_0$", fontsize=7, labelpad=1)
    polarity_bar = figure.colorbar(
        polarity_image, cax=polarity_cax, orientation="horizontal"
    )
    polarity_bar.set_ticks((0.0, polarity_limit / 2.0, polarity_limit))
    polarity_bar.set_label(r"$|\mathbf{p}_1-\mathbf{p}_0|$", fontsize=7, labelpad=1)
    for colorbar in (density_bar, polarity_bar):
        colorbar.ax.tick_params(labelsize=6, length=2.0, width=0.5, pad=1.2)
        colorbar.outline.set_linewidth(0.6)

    stem = output_root / STRATEGY_FIGURE_STEMS[case_number]
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    figure.savefig(pdf_path, facecolor="white", dpi=300)
    figure.savefig(png_path, facecolor="white", dpi=300)
    figure.clear()
    return pdf_path, png_path


def render_manuscript_strategy_figures(
    results: dict[str, PositionPDFResult],
    config: ProjectConfig,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Render the main validation and the shared-scale appendix summary."""

    ordered = _paper_methods(results)
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(_RC):
        render_case_one = (
            _render_setup_and_case_one
            if len(ordered) >= 3
            else _render_case_one_shared_style
        )
        written = list(render_case_one(ordered, config, output_root))
        if len(ordered) == 3:
            written.extend(_render_shared_scale_summary(ordered, config, output_root))
    return tuple(written)
