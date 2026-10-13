"""Time evolution of the macroscopic observables along all three paths.

One sheet compares the random walk, the polar PDE, and the closed-form
prediction for the net polarity, the centroid, and the two parts of the second
moment -- its trace (the spread) and its traceless part (the pulse shape).  The
uniform background is shown apart because it carries no landscape response, and
the nine rules are split into the level-sensing family (1--4) and the
gradient-sensing family (5--9) so that no panel carries more series than a
categorical palette separates reliably.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.ticker import LogLocator
from numpy.typing import NDArray

from ..config import ProjectConfig
from ..observables import (
    POINT_RELEASE_EXPONENTS,
    SHORT_TIME_EXPONENTS,
    TransportObservables,
    effective_diffusivity,
    polarity_relaxation_rate,
)

# Categorical slots 1-5 of the reference palette, validated for adjacent line
# pairs in light mode.  Identity is carried by end labels as well as by hue.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK = "#0b0b0b"
MUTED = "#6b6a66"
FAINT = "#a9a8a3"
GRID = "#dedcd8"
_HALO = [withStroke(linewidth=2.0, foreground="white")]

LEVEL_CASES = (1, 2, 3, 4)
GRADIENT_CASES = (5, 6, 7, 8, 9)
FAMILY_LABELS = ("Cases 1–4: level sensing", "Cases 5–9: gradient sensing")
OBSERVABLES = (
    ("net_polarity", 0, r"average polarity  $|\Delta\vec{P}|$"),
    ("centroid", 1, r"centroid  $|\Delta\vec{X}|$"),
    ("spread", 2, r"spread  $|\Delta S|$"),
    ("shape", 3, r"shape  $|\mathbf{T}|$"),
)
METHOD_STYLE = {
    "particles": {"marker": "o", "ls": "none", "ms": 3.0, "mfc": "none", "mew": 0.9},
    "polar PDE": {"ls": (0, (4.5, 2.0)), "lw": 1.5},
    "prediction": {"ls": "-", "lw": 2.2, "alpha": 0.55},
}
_MARKER_SLICES = 13
_VISIBLE_SIGNIFICANCE = 2.0


def _magnitude(
    series: TransportObservables, name: str, case: int
) -> NDArray[np.float64]:
    index = series.case_numbers.index(case)
    response = series.response(name)[index]
    return np.abs(response) if response.ndim == 1 else np.linalg.norm(response, axis=-1)


def _uncertainty(
    series: TransportObservables, name: str, case: int
) -> NDArray[np.float64] | None:
    errors = getattr(series, f"{name}_error", None)
    if errors is None:
        return None
    index = series.case_numbers.index(case)
    value = errors[index]
    return value if value.ndim == 1 else np.linalg.norm(value, axis=-1)


def _tidy(axis, *, logy: bool = True) -> None:
    axis.set_xscale("log")
    if logy:
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(numticks=5))
        axis.yaxis.set_minor_locator(LogLocator(subs="auto", numticks=12))
    axis.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    axis.tick_params(labelsize=7, color=FAINT, labelcolor=MUTED, length=2.4, pad=1.5)
    axis.tick_params(which="minor", length=1.2, color=FAINT)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def _slope_guides(axis, reference, cases, name, scaled, exponents) -> None:
    """One faint reference slope per predicted exponent present in the panel."""

    column = {"net_polarity": 0, "centroid": 1, "spread": 2, "shape": 3}[name]
    seen: dict[int, float] = {}
    for case in cases:
        exponent = exponents[case][column]
        if exponent is None:
            continue
        magnitude = _magnitude(reference, name, case)
        anchor = max(1, len(scaled) // 6)
        offset = magnitude[anchor] / scaled[anchor] ** exponent
        seen[exponent] = max(seen.get(exponent, 0.0), offset)
    bottom, top = axis.get_ylim()
    first, last = int(0.12 * len(scaled)), int(0.62 * len(scaled))
    window = slice(first, last)
    for exponent, offset in seen.items():
        # Lifted clear of the curve it was read from: a guide invites a slope
        # comparison, and an overlay reads as a fit.
        guide = 2.4 * offset * scaled**exponent
        axis.plot(
            scaled[window],
            guide[window],
            color=FAINT,
            lw=0.7,
            ls=(0, (1, 2)),
            zorder=1,
        )
        if not bottom <= guide[first] <= top:
            continue
        axis.annotate(
            rf"$t^{exponent}$" if exponent > 1 else "$t$",
            (scaled[first], guide[first]),
            xytext=(-2, 1),
            textcoords="offset points",
            color=MUTED,
            fontsize=6.5,
            ha="right",
            va="bottom",
            path_effects=_HALO,
        )


def render_macroscopic_observable_report(
    series: dict[str, TransportObservables],
    config: ProjectConfig,
    figure_path: str | Path,
    *,
    also_png: bool = True,
) -> Path:
    """Write the three-path comparison of the macroscopic observables.

    ``also_png`` writes a raster copy beside the vector one, which is useful for
    the supplementary index but only clutters the manuscript figure directory.
    """

    if not series:
        raise ValueError("no observable series were supplied")
    reference = series.get("prediction", next(iter(series.values())))
    gamma1 = polarity_relaxation_rate(config)
    exponents = (
        POINT_RELEASE_EXPONENTS
        if config.density.initial_position_distribution == "point"
        else SHORT_TIME_EXPONENTS
    )
    figure = Figure(figsize=(7.1, 3.95), dpi=300)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        2,
        5,
        width_ratios=(0.92, 1.0, 1.0, 1.0, 1.0),
        hspace=0.24,
        wspace=0.34,
        left=0.055,
        right=0.958,
        top=0.845,
        bottom=0.125,
    )

    background = figure.add_subplot(grid[:, 0])
    _plot_background(background, series, config, gamma1)

    for row, cases in enumerate((LEVEL_CASES, GRADIENT_CASES)):
        for column, (name, _, label) in enumerate(OBSERVABLES, start=1):
            axis = figure.add_subplot(grid[row, column])
            _plot_family(axis, series, reference, cases, name, gamma1, exponents)
            if row == 0:
                axis.set_title(label, fontsize=8.2, color=INK, pad=4)
            else:
                axis.set_xlabel(r"$t\,\mathcal{D}_R^{(0)}$", fontsize=8, color=MUTED)
            if column == len(OBSERVABLES):
                twin = axis.twinx()
                twin.set_yticks(())
                for side in twin.spines.values():
                    side.set_visible(False)
                twin.set_ylabel(
                    FAMILY_LABELS[row],
                    fontsize=8,
                    color=INK,
                    labelpad=8,
                    rotation=270,
                    va="bottom",
                )

    handles = [
        Line2D([], [], color=INK, label=name, **style)
        for name, style in METHOD_STYLE.items()
        if name in series
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.005),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        handlelength=2.6,
        columnspacing=1.8,
        labelcolor=INK,
    )
    output = Path(figure_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    if also_png:
        figure.savefig(output.with_suffix(".png"))
    return output


def _background_spread(data: TransportObservables) -> NDArray[np.float64]:
    """Baseline second moment; Case 0 has no response, so the MSD is exact.

    For the particle path the mean-square displacement is used instead of
    ``<|r|^2> - <|r(0)|^2>``: the two agree identically in the uniform
    background, but the displacement form is free of the cross term between a
    particle's random starting point and its later motion.
    """

    if data.method == "particles" and data.mean_squared_displacement is not None:
        return data.mean_squared_displacement[0]
    return data.spread[0] - data.spread[0, 0]


def _plot_background(axis, series, config: ProjectConfig, gamma1: float) -> None:
    """Uniform-background spread and outward alignment, both normalized to one."""

    saturation = config.kinetics.v0 / gamma1
    diffusive = 4.0 * effective_diffusivity(config)
    for method, style in METHOD_STYLE.items():
        if method not in series:
            continue
        data = series[method]
        sample = _sampled(len(data.times), method)
        scaled = data.times[sample] * gamma1
        with np.errstate(invalid="ignore", divide="ignore"):
            axis.plot(
                scaled,
                (_background_spread(data) / (diffusive * data.times))[sample],
                color=SERIES_COLORS[0],
                **style,
            )
        if data.outward_polarity is not None:
            axis.plot(
                scaled,
                data.outward_polarity[0][sample] / saturation,
                color=SERIES_COLORS[1],
                **style,
            )
    axis.axhline(1.0, color=FAINT, lw=0.7, ls=(0, (1, 2)), zorder=0)
    axis.axvline(1.0, color=GRID, lw=0.8, zorder=0)
    axis.set_ylim(0.0, 1.06)
    _tidy(axis, logy=False)
    axis.set_xlabel(r"$t\,\mathcal{D}_R^{(0)}$", fontsize=8, color=MUTED)
    axis.set_title("uniform background", fontsize=8.2, color=INK, pad=4)
    axis.annotate(
        "common asymptote",
        (0.03, 1.0),
        xycoords=("axes fraction", "data"),
        color=MUTED,
        fontsize=6.8,
        ha="left",
        va="bottom",
    )
    axis.annotate(
        r"$S^{(0)}/4D_{\mathrm{eff}}^{(0)}t$",
        (0.06, 0.60),
        xycoords="axes fraction",
        color=SERIES_COLORS[0],
        fontsize=7.5,
        ha="left",
        va="bottom",
        path_effects=_HALO,
    )
    axis.annotate(
        r"$\Omega^{(0)}\mathcal{D}_R^{(0)}/v^{(0)}$",
        (0.96, 0.30),
        xycoords="axes fraction",
        color=SERIES_COLORS[1],
        fontsize=7.5,
        ha="right",
        va="top",
        path_effects=_HALO,
    )


def _sampled(count: int, method: str) -> NDArray[np.int64]:
    """Markers on a subset of slices; lines everywhere."""

    if method != "particles":
        return np.arange(1, count)
    return np.unique(np.geomspace(1, count - 1, _MARKER_SLICES).astype(int))


def _plot_family(axis, series, reference, cases, name, gamma1, exponents) -> None:
    scaled = reference.times * gamma1
    silent = []
    endpoints: list[tuple[float, str, str]] = []
    for slot, case in enumerate(cases):
        magnitude = _magnitude(reference, name, case)
        if np.max(magnitude) <= 0.0:
            silent.append(case)
            continue
        color = SERIES_COLORS[slot]
        for method, style in METHOD_STYLE.items():
            if method not in series:
                continue
            data = series[method]
            sample = _sampled(len(data.times), method)
            values = _magnitude(data, name, case)
            error = _uncertainty(data, name, case)
            if error is not None:
                # Only show a measurement that stands clear of its own paired
                # sampling error; below that the magnitude is noise, not signal.
                sample = sample[values[sample] >= _VISIBLE_SIGNIFICANCE * error[sample]]
            axis.plot(
                data.times[sample] * gamma1,
                values[sample],
                color=color,
                **style,
                zorder=3,
            )
        endpoints.append((magnitude[-1], f"{case}", color))
    _tidy(axis)
    axis.autoscale_view()
    limits = axis.get_ylim()
    _slope_guides(axis, reference, cases, name, scaled, exponents)
    axis.set_ylim(limits)
    axis.axvline(1.0, color=GRID, lw=0.8, zorder=0)
    _label_endpoints(axis, endpoints, scaled[-1])
    if silent:
        joined = ", ".join(str(case) for case in silent)
        axis.annotate(
            ("Cases " if len(silent) > 1 else "Case ") + joined + ": zero",
            # Bottom right: every response here rises on log-log axes, so
            # that corner is the one place a label cannot collide with data
            # or with a slope guide.
            (0.96, 0.05),
            xycoords="axes fraction",
            color=MUTED,
            fontsize=6.8,
            ha="right",
            va="bottom",
        )
    axis.set_xlim(scaled[1], scaled[-1] * 1.55)


def _label_endpoints(axis, endpoints, x_position: float) -> None:
    """Case numbers at the right end of each curve, nudged apart when they pile up."""

    if not endpoints:
        return
    axis.autoscale_view()
    bottom, top = axis.get_ylim()
    decades = np.log10(top / bottom)
    ordered = sorted(endpoints, key=lambda item: item[0])
    previous = -np.inf
    for value, text, color in ordered:
        level = np.log10(max(value, bottom))
        level = max(level, previous + 0.055 * decades)
        previous = level
        axis.annotate(
            text,
            (x_position, 10.0**level),
            xytext=(3, 0),
            textcoords="offset points",
            color=color,
            fontsize=7.2,
            ha="left",
            va="center",
            path_effects=_HALO,
            annotation_clip=False,
        )


__all__ = ["render_macroscopic_observable_report"]
