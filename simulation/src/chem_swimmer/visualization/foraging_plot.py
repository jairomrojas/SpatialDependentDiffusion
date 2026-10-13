"""What the population gains: uptake, the two clocks, and distinguishability.

Three panels that answer the foraging questions rather than the transport ones:
how much nutrient a strategy collects, how long collecting it takes compared
with exploring the arena, and how many cells an observer needs to tell one
strategy from another.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.ticker import LogLocator

from ..config import ProjectConfig
from ..density import PDF_CASES, resolve_initial_gaussian_sigma
from ..fields import ChemicalField
from ..numerics import inferred_characteristic_length
from ..observables import (
    TransportObservables,
    chemotactic_mobility,
    domain_mixing_time,
    even_mode_amplitude,
    invert_saturations,
    landscape_uptake_time,
    pulse_drift_coefficient,
    uptake_gain,
)

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK = "#0b0b0b"
# Symlog linear threshold of the coefficient panel; points within it overlap.
_LABEL_MERGE = 0.02
MUTED = "#6b6a66"
FAINT = "#a9a8a3"
GRID = "#dedcd8"
_HALO = [withStroke(linewidth=2.0, foreground="white")]

METHOD_STYLE = {
    "particles": {"marker": "o", "ls": "none", "ms": 3.0, "mfc": "none", "mew": 0.9},
    "polar PDE": {"ls": (0, (4.5, 2.0)), "lw": 1.5},
    "prediction": {"ls": "-", "lw": 2.2, "alpha": 0.55},
}
# Cases grouped by the exponent their discrimination inherits, which is twice
# the exponent of their density response.
EVIDENCE_CLASSES = (
    (2, (1, 7), "#2a78d6"),
    (4, (3, 5, 6, 8, 9), "#eb6834"),
    (6, (2, 4), "#1baf7a"),
)


def _tidy(axis, *, logx=True, logy=True) -> None:
    if logx:
        axis.set_xscale("log")
    if logy:
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(numticks=6))
    axis.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    axis.tick_params(labelsize=7, color=FAINT, labelcolor=MUTED, length=2.4, pad=1.5)
    axis.tick_params(which="minor", length=1.2, color=FAINT)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def render_foraging_report(
    config: ProjectConfig,
    field: ChemicalField,
    *,
    times,
    particle_uptake,
    particle_error,
    solver_uptake,
    closed_uptake,
    clocks: TransportObservables,
    evidence: TransportObservables,
    evidence_config: ProjectConfig,
    figure_path: str | Path,
    also_png: bool = True,
) -> Path:
    """Write the three foraging panels."""

    # One manuscript column wide, so the type keeps its native size.
    figure = Figure(figsize=(3.4, 4.6), dpi=300)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        2, 2, wspace=0.50, hspace=0.52, left=0.165, right=0.975, top=0.905, bottom=0.082
    )
    harvest = landscape_uptake_time(config, field.wavenumber)

    panels = {}
    panels["a"] = figure.add_subplot(grid[0, 0])
    _plot_coefficients(panels["a"], config, field, clocks)
    panels["b"] = figure.add_subplot(grid[0, 1])
    _plot_collapse(
        panels["b"],
        config,
        field,
        times,
        particle_uptake,
        particle_error,
        solver_uptake,
        closed_uptake,
        harvest,
    )
    panels["c"] = figure.add_subplot(grid[1, 0])
    _plot_harvest_law(
        panels["c"],
        config,
        field,
        times,
        particle_uptake,
        particle_error,
        solver_uptake,
        closed_uptake,
    )
    panels["d"] = figure.add_subplot(grid[1, 1])
    _plot_evidence(panels["d"], evidence, evidence_config)

    # The caption refers to the panels by letter, so label them.
    for letter, axis in panels.items():
        axis.annotate(
            f"({letter})",
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-26, 13),
            textcoords="offset points",
            fontsize=6.4,
            fontweight="bold",
            color=INK,
            ha="left",
            va="center",
            annotation_clip=False,
        )

    handles = [
        Line2D([], [], color=INK, label=name, **style)
        for name, style in METHOD_STYLE.items()
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.000),
        ncol=3,
        frameon=False,
        fontsize=6.0,
        handlelength=2.4,
        columnspacing=1.5,
        labelcolor=INK,
    )
    output = Path(figure_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    if also_png:
        figure.savefig(output.with_suffix(".png"))
    return output


def _plot_coefficients(axis, config, field, clocks) -> None:
    """One landscape mode returns the mobility and the drift, and they differ.

    The trace of the second moment saturates on ``-4 eps kappa/(D_eff q^2)``
    and its traceless part on ``-2 eps V/(D_eff q^2)``, so inverting the two
    places each rule at a point in the ``(V, kappa)`` plane.  Rules on the
    diagonal are steered; rules off it are being carried by a spatially varying
    diffusivity instead.
    """

    sigma = resolve_initial_gaussian_sigma(
        config.density, inferred_characteristic_length(config, field)
    )
    amplitude = even_mode_amplitude(field, config.initial.position)
    baseline_shape = clocks.shape[0, -1, 0]
    baseline_spread = clocks.spread[0, -1]
    span = 0.0
    label_groups: list[dict] = []
    for case_index, case_number in enumerate(PDF_CASES):
        if case_number == 0:
            continue
        drift = pulse_drift_coefficient(config, case_number)
        mobility = chemotactic_mobility(config, case_number)
        measured = invert_saturations(
            config,
            field.wavenumber,
            amplitude,
            sigma,
            float(clocks.shape[case_index, -1, 0] - baseline_shape),
            float(clocks.spread[case_index, -1] - baseline_spread),
        )
        color = SERIES_COLORS[(case_number - 1) % len(SERIES_COLORS)]
        span = max(span, abs(drift), abs(mobility), abs(measured[0]), abs(measured[1]))
        axis.plot(
            *measured,
            marker="o",
            ms=4.5,
            mfc="none",
            mew=1.1,
            color=color,
            ls="none",
            zorder=4,
        )
        axis.plot(
            drift, mobility, marker="+", ms=7, mew=1.3, color=color, ls="none", zorder=5
        )
        # Degenerate and near-null rules land on top of each other (2 and 4 by
        # construction, and 5 at the origin).  Stacking their labels is
        # unreadable, so points closer than the symlog linear threshold -- i.e.
        # visually coincident -- are collected and labelled once.
        for group in label_groups:
            if abs(group["x"] - drift) <= _LABEL_MERGE and (
                abs(group["y"] - mobility) <= _LABEL_MERGE
            ):
                group["cases"].append(case_number)
                break
        else:
            label_groups.append(
                {"x": drift, "y": mobility, "cases": [case_number], "color": color}
            )
    for group in label_groups:
        axis.annotate(
            ", ".join(str(number) for number in group["cases"]),
            (group["x"], group["y"]),
            xytext=(5, 3),
            textcoords="offset points",
            color=group["color"],
            fontsize=5.8,
            ha="left",
            va="bottom",
            path_effects=_HALO,
            annotation_clip=False,
        )
    limit = 1.25 * span
    axis.plot(
        [-limit, limit], [-limit, limit], color=FAINT, lw=0.8, ls=(0, (1, 2)), zorder=1
    )
    axis.axhline(0.0, color=GRID, lw=0.6, zorder=1)
    axis.axvline(0.0, color=GRID, lw=0.6, zorder=1)
    axis.annotate(
        r"$\kappa=\mathcal{V}$",
        (0.96, 0.90),
        xycoords="axes fraction",
        color=MUTED,
        fontsize=5.6,
        ha="right",
        va="top",
        path_effects=_HALO,
    )
    _tidy(axis, logx=False, logy=False)
    axis.set_xscale("symlog", linthresh=0.02)
    axis.set_yscale("symlog", linthresh=0.02)
    axis.set_xlim(-limit, limit)
    # Headroom so the topmost case label clears the panel title.
    axis.set_ylim(-limit, 2.6 * limit)
    # A full symlog decade ladder does not fit one column; keep three per side.
    ticks = [-1.0, -1.0e-1, -1.0e-2, 0.0, 1.0e-2, 1.0e-1, 1.0]
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xlabel(
        r"drift $\mathcal{V}$, from the pulse shape", fontsize=6.4, color=MUTED
    )
    axis.set_ylabel(
        r"mobility $\kappa$, from the spread", fontsize=6.4, color=INK, labelpad=2
    )
    axis.set_title("one mode, both coefficients", fontsize=6.6, color=INK, pad=4)


def _plot_collapse(
    axis, config, field, times, particle, particle_error, solver, closed, harvest
):
    """Every responding rule, divided by its own predicted gain, on one curve."""

    scaled = times / harvest
    silent, drawn = [], 0
    for case_index, case_number in enumerate(PDF_CASES):
        if case_number == 0:
            continue
        gain = uptake_gain(config, field, case_number)
        if abs(gain) < 1.0e-12:
            silent.append(case_number)
            continue
        color = SERIES_COLORS[drawn % len(SERIES_COLORS)]
        drawn += 1
        series = {
            "particles": (particle[case_index] - particle[0]) / gain,
            "polar PDE": (solver[case_index] - solver[0]) / gain,
            "prediction": (closed[case_index] - closed[0]) / gain,
        }
        error = particle_error[case_index] / abs(gain)
        for name, values in series.items():
            style = METHOD_STYLE[name]
            keep = slice(1, None)
            if name == "particles":
                sample = np.unique(np.geomspace(1, len(times) - 1, 11).astype(int))
                sample = sample[np.abs(values[sample]) >= 2.0 * error[sample]]
                keep = sample
            axis.plot(scaled[keep], values[keep], color=color, zorder=3, **style)
        axis.annotate(
            f"{case_number}",
            (scaled[-1], series["prediction"][-1]),
            xytext=(3, 0),
            textcoords="offset points",
            color=color,
            fontsize=5.8,
            ha="left",
            va="center",
            path_effects=_HALO,
            annotation_clip=False,
        )
    reference = np.linspace(0.0, scaled[-1], 200)
    axis.plot(
        reference,
        1.0 - np.exp(-reference),
        color=FAINT,
        lw=0.8,
        ls=(0, (1, 2)),
        zorder=1,
    )
    axis.annotate(
        r"$1-e^{-t/\tau_\ell}$",
        (0.30, 0.72),
        xycoords="axes fraction",
        color=MUTED,
        fontsize=5.6,
        ha="left",
        va="top",
        path_effects=_HALO,
    )
    _tidy(axis, logx=False, logy=False)
    axis.set_xlim(0.0, scaled[-1] * 1.12)
    axis.set_ylim(-0.08, 1.22)
    axis.set_xlabel(r"$t/\tau_\ell$", fontsize=6.4, color=MUTED)
    axis.set_ylabel("uptake gain / predicted gain", fontsize=6.4, color=INK, labelpad=3)
    axis.set_title(
        "every rule earns $\\kappa\\varepsilon^2/2D_{\\mathrm{eff}}^{(0)}$",
        fontsize=6.6,
        color=INK,
        pad=4,
    )
    if silent:
        axis.annotate(
            "Cases " + ", ".join(str(c) for c in silent) + r": $\kappa=0$",
            (0.04, 0.05),
            xycoords="axes fraction",
            color=MUTED,
            fontsize=5.4,
            ha="left",
            va="bottom",
        )


def _plot_harvest_law(
    axis, config, field, times, particle, particle_error, solver, closed
) -> None:
    """The harvest law itself: what a rule earns is its mobility, and nothing else.

    Equation (harvest law) says the collected nutrient is
    ``kappa <|grad eta|^2>``, so plotting each rule's measured gain against its
    prescribed ``kappa`` should put all nine on one line through the origin.
    The interesting departures are the rules with ``kappa = 0``: Cases 2 and 4
    sit at the origin, while Case 5 earns a gain the mobility cannot account
    for, because its contribution enters one order further out in the gradient
    expansion.
    """

    mobilities, measured, errors, colors, labels = [], [], [], [], []
    for case_index, case_number in enumerate(PDF_CASES):
        if case_number == 0:
            continue
        mobilities.append(chemotactic_mobility(config, case_number))
        measured.append(float(particle[case_index, -1] - particle[0, -1]))
        errors.append(float(particle_error[case_index, -1]))
        colors.append(SERIES_COLORS[(case_number - 1) % len(SERIES_COLORS)])
        labels.append(case_number)

    mobilities = np.asarray(mobilities)
    measured = np.asarray(measured)
    errors = np.asarray(errors)

    # The predicted line, kappa * epsilon^2 / 2 D_eff, evaluated once.
    reference = next(
        (index for index, number in enumerate(labels) if abs(mobilities[index]) > 0.0),
        0,
    )
    slope = uptake_gain(config, field, labels[reference]) / mobilities[reference]
    span = np.linspace(mobilities.min() * 1.25, mobilities.max() * 1.25, 400)
    axis.plot(span, slope * span, color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)

    for index, number in enumerate(labels):
        axis.errorbar(
            mobilities[index],
            measured[index],
            yerr=errors[index],
            marker="o",
            ms=3.4,
            mfc="none",
            mew=1.0,
            color=colors[index],
            ls="none",
            elinewidth=0.7,
            capsize=1.2,
            zorder=3,
        )
        axis.annotate(
            f"{number}",
            (mobilities[index], measured[index]),
            xytext=(3.5, 2.5),
            textcoords="offset points",
            color=colors[index],
            fontsize=5.6,
            ha="left",
            va="bottom",
            path_effects=_HALO,
            annotation_clip=False,
        )
    axis.axhline(0.0, color=GRID, lw=0.5, zorder=1)
    axis.axvline(0.0, color=GRID, lw=0.5, zorder=1)
    _tidy(axis, logx=False, logy=False)
    # Case 7 outweighs the rest by two decades, so a linear pair of axes
    # crushes the other eight onto the origin; symlog spreads them while
    # still showing the exact zeros of Cases 2 and 4.
    gain_scale = float(np.max(np.abs(measured))) * 1.0e-3
    axis.set_xscale("symlog", linthresh=0.02)
    axis.set_yscale("symlog", linthresh=gain_scale)
    axis.set_xticks([-1.0e-1, 0.0, 1.0e-1, 1.0])
    axis.set_xlabel(r"mobility $\kappa$", fontsize=6.4, color=MUTED)
    axis.set_ylabel("uptake gain", fontsize=6.4, color=INK, labelpad=1)
    axis.tick_params(axis="y", labelsize=5.0)
    axis.set_title(r"the harvest is $\kappa$", fontsize=6.6, color=INK, pad=4)
    axis.annotate(
        r"slope $\varepsilon^{2}/2D_{\rm eff}^{(0)}$",
        (0.03, 0.97),
        xycoords="axes fraction",
        color=MUTED,
        fontsize=5.4,
        ha="left",
        va="top",
        path_effects=_HALO,
    )


def _plot_clocks(axis, config, clocks, harvest, *, early_times, early_uptake):
    """The uptake finishes on the landscape clock, the spread on the domain one.

    The uptake is drawn from the short run, whose slices resolve the landscape
    time; the spread needs the long one, which reaches the mixing time.  Both
    are the same solver on the same configuration.
    """

    mixing = domain_mixing_time(config)
    reference = 7
    gain = early_uptake[reference] - early_uptake[0]
    plateau = clocks.uptake[reference, -1] - clocks.uptake[0, -1]
    spread = clocks.spread[0] - clocks.spread[0, 0]
    axis.plot(
        early_times[1:] / harvest,
        gain[1:] / plateau,
        color=SERIES_COLORS[0],
        lw=1.8,
        zorder=3,
    )
    axis.plot(
        clocks.times[1:] / harvest,
        spread[1:] / spread[-1],
        color=SERIES_COLORS[1],
        lw=1.8,
        zorder=3,
    )
    for position, label in ((1.0, r"$\tau_\ell$"), (mixing / harvest, r"$\tau_R$")):
        axis.axvline(position, color=FAINT, lw=0.8, ls=(0, (1, 2)), zorder=1)
        axis.annotate(
            label,
            (position, 1.16),
            color=MUTED,
            fontsize=6.0,
            ha="center",
            va="bottom",
            path_effects=_HALO,
        )
    _tidy(axis, logy=False)
    axis.set_ylim(0.0, 1.14)
    axis.set_xlabel(r"$t/\tau_\ell$", fontsize=6.4, color=MUTED)
    axis.set_ylabel("fraction of final value", fontsize=6.4, color=INK, labelpad=3)
    axis.set_title(
        f"two clocks, {mixing / harvest:.0f} apart", fontsize=6.6, color=INK, pad=4
    )
    axis.annotate(
        "uptake",
        (0.06, 0.92),
        xycoords="axes fraction",
        color=SERIES_COLORS[0],
        fontsize=6.0,
        ha="left",
        va="top",
        path_effects=_HALO,
    )
    axis.annotate(
        "spread",
        (0.70, 0.40),
        xycoords="axes fraction",
        color=SERIES_COLORS[1],
        fontsize=6.0,
        ha="left",
        va="top",
        path_effects=_HALO,
    )


def _plot_evidence(axis, evidence, config):
    """Divergence from the null forager, grouped by the exponent it inherits."""

    times = evidence.times[1:]
    gamma = config.kinetics.D_R0 + config.kinetics.lambda0 * 0.70
    scaled = times * gamma
    for exponent, cases, color in EVIDENCE_CLASSES:
        for case_number in cases:
            index = evidence.case_numbers.index(case_number)
            values = evidence.discrimination[index][1:]
            axis.plot(
                scaled,
                np.maximum(values, 1e-30),
                color=color,
                lw=1.4,
                zorder=3,
                alpha=0.9,
            )
        anchor = len(scaled) // 3
        index = evidence.case_numbers.index(cases[0])
        level = evidence.discrimination[index][1:][anchor]
        guide = 3.0 * level * (scaled / scaled[anchor]) ** exponent
        window = slice(anchor // 3, int(0.75 * len(scaled)))
        axis.plot(
            scaled[window], guide[window], color=FAINT, lw=0.7, ls=(0, (1, 2)), zorder=1
        )
        axis.annotate(
            rf"$t^{exponent}$: " + ", ".join(str(c) for c in cases),
            (scaled[window][0], guide[window][0]),
            xytext=(-1, 3),
            textcoords="offset points",
            color=color,
            fontsize=5.4,
            ha="left",
            va="bottom",
            path_effects=_HALO,
        )
    _tidy(axis)
    axis.set_xlim(scaled[0], scaled[-1] * 1.25)
    axis.set_ylim(1e-17, 3e-4)
    axis.set_xlabel(r"$t\,\mathcal{D}_R^{(0)}$", fontsize=6.4, color=MUTED)
    axis.set_ylabel(
        r"divergence $\mathcal{K}$ per cell", fontsize=6.4, color=INK, labelpad=3
    )
    axis.set_title(
        r"$M\gtrsim1/\mathcal{K}$ cells to detect", fontsize=6.6, color=INK, pad=4
    )


__all__ = ["render_foraging_report"]
