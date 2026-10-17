"""What the population knows: navigation, degeneracy breaking, and information.

Three panels for the finite-lag observables.  The first shows that every rule's
projected velocity is its own pulse drift times one of three universal lag
shapes.  The second shows that resolving the heading returns two independent
combinations of the steering and turning sensitivities where the population
fields return only one.  The third shows that navigation and information are
different observables: the two rules that never navigate carry the most
information about the state, and the rules that navigate best carry almost
none.
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
from ..information import (
    NavigationObservables,
    _gamma_rates,
    _response_terms,
    lag_functions,
    pulse_drift,
)

INK = "#0b0b0b"
MUTED = "#6b6a66"
FAINT = "#a9a8a3"
GRID = "#dedcd8"
_HALO = [withStroke(linewidth=2.0, foreground="white")]

METHOD_STYLE = {
    "particles": {"marker": "o", "ls": "none", "ms": 3.0, "mfc": "none", "mew": 0.9},
    "kinetic": {"ls": (0, (4.5, 2.0)), "lw": 1.5},
    "prediction": {"ls": "-", "lw": 2.2, "alpha": 0.55},
}

#: Rules grouped by the lag shape their drift follows, from Eq. (lag-hierarchy).
LAG_CLASSES = (
    ("instant", (7,), "#2a78d6", r"$\Phi_7=1$"),
    ("one relaxation", (3, 6, 8, 9), "#eb6834", r"$\Phi=1-W$"),
    ("two relaxations", (2, 4), "#1baf7a", r"$\Psi$"),
)
SILENT_CASES = (1, 5)
#: The three rules the population fields merge into one mobility.
TURNING_CASES = ((9, "#2a78d6"), (6, "#eb6834"), (8, "#1baf7a"))
INFORMATION_CASES = (
    (1, "#2a78d6"),
    (5, "#eb6834"),
    (7, "#1baf7a"),
    (3, "#eda100"),
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


def _sample(count: int, points: int = 12) -> np.ndarray:
    return np.unique(np.geomspace(1, count - 1, points).astype(int))


def render_information_report(
    config: ProjectConfig,
    gradient_magnitude: float,
    *,
    particles: NavigationObservables,
    kinetic: NavigationObservables,
    prediction: NavigationObservables,
    figure_path: str | Path,
    also_png: bool = True,
) -> Path:
    """Write the three finite-lag panels."""

    # One manuscript column wide: the three panels stack instead of sitting
    # side by side, so the type stays at its native size rather than being
    # scaled down by \includegraphics.
    figure = Figure(figsize=(3.4, 3.9), dpi=300)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        2,
        2,
        hspace=0.62,
        wspace=0.52,
        left=0.135,
        right=0.975,
        top=0.905,
        bottom=0.095,
    )
    paths = {"particles": particles, "kinetic": kinetic, "prediction": prediction}

    letters = ["a", "b", "c", "d"]
    axes = [
        figure.add_subplot(grid[row, column]) for row in range(2) for column in range(2)
    ]
    _plot_collapse(axes[0], config, gradient_magnitude, paths)
    _plot_degeneracy(axes[1], config, gradient_magnitude, paths)
    _plot_level_harmonic(axes[2], config, paths)
    _plot_information(axes[3], config, paths)

    # The caption refers to the panels by letter, so label them.
    for letter, axis in zip(letters, axes):
        axis.annotate(
            f"({letter})",
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-30, 12),
            textcoords="offset points",
            fontsize=7.0,
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
        bbox_to_anchor=(0.5, 1.004),
        ncol=3,
        frameon=False,
        fontsize=6.4,
        handlelength=1.8,
        columnspacing=0.9,
        labelcolor=INK,
    )
    output = Path(figure_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    if also_png:
        figure.savefig(output.with_suffix(".png"))
    return output


def _plot_collapse(axis, config, gradient_magnitude, paths) -> None:
    """Each rule's projected velocity, divided by its own pulse drift."""

    gamma1 = _gamma_rates(config)[0]
    lags = paths["prediction"].lags
    scaled = gamma1 * lags

    for _, cases, color, _ in LAG_CLASSES:
        for case_number in cases:
            drift = pulse_drift(config, case_number) * gradient_magnitude
            for name, result in paths.items():
                values = result.response("projected_velocity")[case_number] / drift
                style = METHOD_STYLE[name]
                keep = slice(1, None)
                if name == "particles":
                    error = result.projected_velocity_error[case_number] / abs(drift)
                    sample = _sample(len(lags))
                    keep = sample[np.abs(values[sample]) >= 2.0 * error[sample]]
                axis.plot(scaled[keep], values[keep], color=color, zorder=3, **style)

    reference = np.geomspace(scaled[1], scaled[-1], 200)
    shapes = lag_functions(*_gamma_rates(config), reference / gamma1)
    for (_, _, color, label), curve, height in zip(
        LAG_CLASSES,
        (np.ones_like(reference), shapes.Phi, shapes.Psi),
        (1.06, 0.86, 0.24),
        strict=True,
    ):
        axis.plot(reference, curve, color=color, lw=0.7, ls=(0, (1, 2)), zorder=1)
        axis.annotate(
            label,
            (scaled[-1], height),
            xytext=(-2, 0),
            textcoords="offset points",
            color=color,
            fontsize=5.9,
            ha="right",
            va="bottom",
            path_effects=_HALO,
        )
    _tidy(axis, logy=False)
    axis.set_ylim(-0.06, 1.22)
    axis.set_xlabel(r"lag $\mathcal{D}_R^{(0)}\tau$", fontsize=6.6, color=MUTED)
    axis.set_ylabel(
        r"$V_\parallel\,/\,\mathcal{V}_j\langle g\rangle$",
        fontsize=6.6,
        color=INK,
        labelpad=2,
    )
    axis.set_title("three lag shapes", fontsize=6.8, color=INK, pad=4)
    axis.annotate(
        "Cases "
        + ", ".join(str(c) for c in SILENT_CASES)
        + r": $\mathcal{V}=0$ at every lag",
        (0.045, 0.20),
        xycoords="axes fraction",
        color=MUTED,
        fontsize=6.5,
        ha="left",
        va="center",
        path_effects=_HALO,
    )


def _recovered_pair(config, gradient_magnitude, result, case_number):
    """Return ``(chi - D_R^zeta, chi + D_R^zeta)`` read off the two harmonics."""

    gamma1, gamma2, gamma3 = _gamma_rates(config)
    shapes = lag_functions(gamma1, gamma2, gamma3, result.lags)
    v0 = config.kinetics.v0
    g = gradient_magnitude
    difference = (
        2.0
        * gamma1
        * result.response("projected_velocity")[case_number]
        / (v0 * g * np.where(shapes.Phi > 0.0, shapes.Phi, np.nan))
    )
    total = (
        -2.0
        * result.response("alignment_harmonic")[case_number]
        / (v0 * g * np.where(shapes.Theta > 0.0, shapes.Theta, np.nan))
    )
    return difference, total


def _plot_degeneracy(axis, config, gradient_magnitude, paths) -> None:
    """The two harmonics return two combinations where the fields return one."""

    lags = paths["prediction"].lags
    window = slice(len(lags) // 3, None)
    span = 0.0
    for case_number, color in TURNING_CASES:
        terms = _response_terms(config, case_number)
        truth = (
            terms["chi"] - terms["gamma_zeta"],
            terms["chi"] + terms["gamma_zeta"],
        )
        span = max(span, abs(truth[0]), abs(truth[1]))
        axis.plot(
            *truth,
            marker="+",
            ms=8,
            mew=1.4,
            color=color,
            ls="none",
            zorder=5,
        )
        axis.annotate(
            f"case {case_number}",
            truth,
            xytext=(6, -2),
            textcoords="offset points",
            color=color,
            fontsize=5.9,
            ha="left",
            va="top",
            path_effects=_HALO,
        )
        for name, result in paths.items():
            difference, total = _recovered_pair(
                config, gradient_magnitude, result, case_number
            )
            style = dict(METHOD_STYLE[name])
            if name == "particles":
                keep = _sample(len(lags), 9)
                keep = keep[keep >= len(lags) // 3]
                axis.plot(difference[keep], total[keep], color=color, zorder=3, **style)
            else:
                axis.plot(
                    difference[window], total[window], color=color, zorder=3, **style
                )
    limit = 1.35 * span
    line = np.array([-limit, limit])
    for slope, label, position in (
        (1.0, r"pure steering", 0.72),
        (-1.0, r"pure turning", 0.28),
    ):
        axis.plot(line, slope * line, color=FAINT, lw=0.7, ls=(0, (1, 2)), zorder=1)
        axis.annotate(
            label,
            (0.97, position),
            xycoords="axes fraction",
            color=MUTED,
            fontsize=6.8,
            ha="right",
            va="center",
            path_effects=_HALO,
        )
    axis.axvspan(-limit, limit, ymin=0.47, ymax=0.53, color="#f0efec", zorder=0)
    _tidy(axis, logx=False, logy=False)
    axis.set_xlim(-limit, 1.35 * limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel(
        r"drift harmonic $\to\ \chi-\mathcal{D}_R^{(\zeta_\parallel)}$",
        fontsize=6.6,
        color=MUTED,
    )
    axis.set_ylabel(
        r"$\cos 2\alpha$ harmonic $\to\ \chi+\mathcal{D}_R^{(\zeta_\parallel)}$",
        fontsize=6.6,
        color=INK,
        labelpad=2,
    )
    axis.set_title("steering vs. turning", fontsize=6.8, color=INK, pad=4)


#: The level-sensing rules the population fields also merge: a level speed
#: response and a level turning response reach the drift the same way.
LEVEL_CASES = ((3, "#eb6834"), (2, "#1baf7a"), (4, "#8f6bd1"))


def _plot_level_harmonic(axis, config, paths) -> None:
    """The second degeneracy the lag breaks: level speed against level turning.

    The drift carries Cases 2, 3 and 4 through one combination, but the
    ``delta_eta cos(alpha)`` harmonic separates them by lag shape and by sign.
    A level speed response returns the bare sensitivity as the lag closes,
    ``C_3 -> v^(eta)``, whereas a level turning response has to build the
    polarity it modulates and so vanishes with the lag.
    """

    gamma1 = _gamma_rates(config)[0]
    lags = paths["prediction"].lags
    scaled = gamma1 * lags

    for case_number, color in LEVEL_CASES:
        for name, result in paths.items():
            values = result.response("level_harmonic")[case_number]
            style = METHOD_STYLE[name]
            keep = slice(1, None)
            if name == "particles":
                error = result.level_harmonic_error
                keep = _sample(len(lags))
                if error is not None:
                    err = error[case_number]
                    keep = keep[np.abs(values[keep]) >= 2.0 * err[keep]]
            axis.plot(scaled[keep], values[keep], color=color, zorder=3, **style)
        finite = np.isfinite(
            paths["prediction"].response("level_harmonic")[case_number]
        )
        if finite.any():
            last = np.flatnonzero(finite)[-1]
            axis.annotate(
                f"{case_number}",
                (
                    scaled[last],
                    paths["prediction"].response("level_harmonic")[case_number][last],
                ),
                xytext=(3, 0),
                textcoords="offset points",
                color=color,
                fontsize=6.4,
                ha="left",
                va="center",
                path_effects=_HALO,
                annotation_clip=False,
            )

    axis.axhline(0.0, color=FAINT, lw=0.6, zorder=1)
    _tidy(axis, logy=False)
    axis.set_xlabel(r"lag $\mathcal{D}_R^{(0)}\tau$", fontsize=6.6, color=MUTED)
    axis.set_ylabel(
        r"$\delta\eta\cos\alpha$ harmonic", fontsize=6.6, color=INK, labelpad=2
    )
    axis.set_title("level speed vs. turning", fontsize=6.8, color=INK, pad=4)
    axis.annotate(
        r"$\to v^{(\eta)}$ as $\tau\to0$",
        (0.04, 0.94),
        xycoords="axes fraction",
        color=MUTED,
        fontsize=6.0,
        ha="left",
        va="top",
        path_effects=_HALO,
    )


def _plot_information(axis, config, paths) -> None:
    """Information carried by the realized displacement, against the baseline."""

    gamma1 = _gamma_rates(config)[0]
    lags = paths["prediction"].lags
    scaled = gamma1 * lags
    for case_number, color in INFORMATION_CASES:
        for name, result in paths.items():
            information = result.response("information")[case_number]
            values = np.abs(information * result.lags)
            finite = np.isfinite(values) & (values > 0.0)
            style = dict(METHOD_STYLE[name])
            axis.plot(scaled[finite], values[finite], color=color, zorder=3, **style)
        reference = paths["prediction"].response("information")[case_number]
        last = np.isfinite(reference).nonzero()[0][-1]
        axis.annotate(
            f"{case_number}",
            (scaled[last], abs(reference[last] * lags[last])),
            xytext=(3, 0),
            textcoords="offset points",
            color=color,
            fontsize=5.9,
            ha="left",
            va="center",
            path_effects=_HALO,
            annotation_clip=False,
        )
    baseline = paths["prediction"].information[0] * lags
    finite = np.isfinite(baseline) & (baseline > 0.0)
    axis.plot(
        scaled[finite], baseline[finite], color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1
    )
    axis.annotate(
        r"baseline $I_0$",
        (scaled[finite][-1], baseline[finite][-1]),
        xytext=(-3, 4),
        textcoords="offset points",
        color=MUTED,
        fontsize=5.7,
        ha="right",
        va="bottom",
        path_effects=_HALO,
    )
    _tidy(axis)
    axis.set_xlabel(r"lag $\mathcal{D}_R^{(0)}\tau$", fontsize=6.6, color=MUTED)
    axis.set_ylabel(
        r"$\Delta I(S_t;Y^\tau)$  (nats)", fontsize=6.6, color=INK, labelpad=2
    )
    axis.set_title("what the motion reveals", fontsize=6.8, color=INK, pad=4)
    axis.annotate(
        "Cases 2, 4, 6, 8, 9\nbelow the floor",
        (0.04, 0.06),
        xycoords="axes fraction",
        color=MUTED,
        fontsize=6.8,
        ha="left",
        va="bottom",
    )
