"""The nutrient landscape, and why this particular mode was chosen.

Every closed form in the manuscript's observable and foraging sections is a
function of one wavenumber, so the configuration uses one cosine mode.  Which
mode is not free: three dimensionless numbers pull in different directions and
the panels here show where they cross.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from scipy import special

from ..config import ProjectConfig
from ..density import resolve_initial_gaussian_sigma
from ..fields import ChemicalField, create_field
from ..numerics import inferred_characteristic_length
from ..observables import effective_diffusivity, polarity_relaxation_rate
from ..turning import create_turning_kernel

INK = "#0b0b0b"
MUTED = "#6b6a66"
FAINT = "#a9a8a3"
GRID = "#dedcd8"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
_HALO = [withStroke(linewidth=2.0, foreground="white")]


def _disk_mesh(points: int, radius: float):
    span = np.linspace(-radius, radius, points)
    x_grid, y_grid = np.meshgrid(span, span, indexing="xy")
    return x_grid, y_grid, x_grid**2 + y_grid**2 <= radius**2


def _second_harmonic_rate(config: ProjectConfig) -> float:
    kernel = create_turning_kernel(config.turning)
    return 4.0 * config.kinetics.D_R0 + config.kinetics.lambda0 * (
        1.0 - kernel.cosine_moment(2)
    )


def _tidy(axis) -> None:
    axis.tick_params(labelsize=7, color=FAINT, labelcolor=MUTED, length=2.4, pad=1.5)
    axis.tick_params(which="minor", length=1.2, color=FAINT)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def _plot_map(axis, config: ProjectConfig, field: ChemicalField, sigma: float) -> None:
    """The landscape over the arena, with the release and the crest lines."""

    radius = config.domain.radius
    x_grid, y_grid, mask = _disk_mesh(401, radius)
    values = np.asarray(field.value(np.stack((x_grid, y_grid), axis=-1)), float)
    image = axis.imshow(
        np.where(mask, values, np.nan),
        origin="lower",
        extent=(-radius, radius, -radius, radius),
        cmap="BrBG",
        vmin=values[mask].min(),
        vmax=values[mask].max(),
        interpolation="bilinear",
        rasterized=True,
    )
    wavevector = np.asarray(field.wavevector, dtype=float)
    wavenumber = float(np.hypot(*wavevector))
    unit = wavevector / wavenumber
    normal = np.array([-unit[1], unit[0]])
    line = np.linspace(-2.0 * radius, 2.0 * radius, 2)
    # The same crest/trough marks the response panels carry, clipped to the
    # arena so they never trail off into the margin.
    clip = Circle((0.0, 0.0), radius, transform=axis.transData)
    for index in range(-4, 5):
        offset = (index * np.pi - field.phase) / wavenumber
        points = offset * unit + line[:, None] * normal[None, :]
        # A pale underlay: the colormap runs from dark brown to dark teal, so a
        # single ink colour would vanish at both ends of the scale.
        (halo,) = axis.plot(
            points[:, 0], points[:, 1], color="white", lw=1.9, alpha=0.55, zorder=3
        )
        halo.set_clip_path(clip)
        (drawn,) = axis.plot(
            points[:, 0],
            points[:, 1],
            color="0.10",
            lw=0.8,
            ls="-" if index % 2 == 0 else (0, (2.4, 2.0)),
            alpha=0.9,
            zorder=3.1,
        )
        drawn.set_clip_path(clip)
    angle = np.linspace(0, 2 * np.pi, 361)
    axis.plot(
        radius * np.cos(angle), radius * np.sin(angle), color=INK, lw=0.9, zorder=4
    )
    axis.plot(
        2 * sigma * np.cos(angle),
        2 * sigma * np.sin(angle),
        color="white",
        lw=1.6,
        zorder=5,
    )
    axis.plot(
        2 * sigma * np.cos(angle),
        2 * sigma * np.sin(angle),
        color=ORANGE,
        lw=1.0,
        zorder=6,
    )
    axis.annotate(
        r"release $2\sigma_0$",
        (0.0, 2 * sigma),
        xytext=(4, 3),
        textcoords="offset points",
        color=ORANGE,
        fontsize=6.2,
        ha="left",
        va="bottom",
        path_effects=_HALO,
        zorder=7,
    )
    tail = -0.30 * radius * unit - 0.62 * radius * normal
    axis.annotate(
        "",
        xy=tail + 0.42 * radius * unit,
        xytext=tail,
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.0, "mutation_scale": 8},
        zorder=7,
    )
    axis.text(
        *(tail + 0.21 * radius * unit - 0.16 * radius * normal),
        r"$\hat q$",
        color=INK,
        fontsize=7.5,
        ha="center",
        va="bottom",
        path_effects=_HALO,
        zorder=7,
    )
    axis.set(
        xlim=(-1.04 * radius, 1.04 * radius),
        ylim=(-1.04 * radius, 1.04 * radius),
        aspect="equal",
    )
    axis.set_xticks(())
    axis.set_yticks(())
    for spine in axis.spines.values():
        spine.set_visible(False)
    bar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    bar.ax.set_title(r"$\eta$", fontsize=8, color=INK, pad=3)
    bar.ax.tick_params(labelsize=6.5, length=2.0, color=FAINT, labelcolor=MUTED)
    bar.outline.set_linewidth(0.5)


def _plot_cut(axis, config: ProjectConfig, field: ChemicalField, sigma: float) -> None:
    """A cut along q: what the release actually straddles."""

    wavenumber = float(np.hypot(*field.wavevector))
    reach = config.domain.radius
    distance = np.linspace(-reach, reach, 801)
    modulation = field.amplitude * np.cos(wavenumber * distance + field.phase)
    even = 0.5 * (modulation + modulation[::-1])
    odd = 0.5 * (modulation - modulation[::-1])
    release = np.exp(-0.5 * (distance / sigma) ** 2)

    axis.fill_between(
        distance,
        0.0,
        field.amplitude * release,
        color=ORANGE,
        alpha=0.16,
        lw=0,
        zorder=1,
    )
    axis.plot(distance, modulation, color=INK, lw=1.6, zorder=4)
    axis.plot(distance, even, color=BLUE, lw=1.0, ls=(0, (4.0, 2.0)), zorder=3)
    axis.plot(distance, odd, color=GREEN, lw=1.0, ls=(0, (1.4, 1.6)), zorder=3)
    axis.axhline(0.0, color=GRID, lw=0.8, zorder=0)
    axis.axvline(0.0, color=GRID, lw=0.8, zorder=0)

    _tidy(axis)
    axis.set_xlim(-reach, 1.22 * reach)
    axis.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    axis.set_xlabel(r"distance along $\hat q$", fontsize=8, color=MUTED, labelpad=1)
    axis.set_ylabel(r"$\delta\eta$", fontsize=8, color=INK, labelpad=1)
    # Named where each curve ends, so no legend sits on top of the data.
    for values, text, color in (
        (modulation, r"$\delta\eta$", INK),
        (even, "even", BLUE),
        (odd, "odd", GREEN),
    ):
        axis.annotate(
            text,
            (reach, values[-1]),
            xytext=(3, 0),
            textcoords="offset points",
            color=color,
            fontsize=7,
            ha="left",
            va="center",
        )
    axis.annotate(
        rf"$q\sigma_0={wavenumber * sigma:.2f}$",
        (0.97, 0.05),
        xycoords="axes fraction",
        color=MUTED,
        fontsize=7,
        ha="right",
        va="bottom",
        path_effects=_HALO,
    )


def _plot_design(axis, config: ProjectConfig, sigma: float, chosen: float) -> None:
    """Why this mode: a discrete set of candidates, and three sliding costs.

    Only the zeros of ``J_1`` leave the disk average of the modulation exactly
    zero, so the admissible modes are a ladder, not a continuum.  Along that
    ladder three quantities move in opposite directions, and the second rung is
    the only one where all three are comfortable.
    """

    radius = config.domain.radius
    scaled = np.linspace(3.0, 20.0, 800)
    wavenumber = scaled / radius
    window = np.exp(-0.5 * (wavenumber * sigma) ** 2)
    closure = (config.kinetics.v0 * wavenumber / _second_harmonic_rate(config)) ** 2
    lifetime = 1.0 / (
        effective_diffusivity(config) * wavenumber**2 * config.simulation.duration
    )

    for zero in special.jn_zeros(1, 5):
        if zero > scaled[-1]:
            continue
        axis.axvline(zero, color=FAINT, lw=0.7, ls=(0, (1.0, 2.0)), zorder=1)
    axis.axvline(chosen, color=INK, lw=1.0, zorder=2)

    axis.semilogy(
        scaled,
        window,
        color=BLUE,
        lw=1.6,
        zorder=4,
        label=r"release window  $e^{-q^2\sigma_0^2/2}$",
    )
    axis.semilogy(
        scaled,
        lifetime,
        color=ORANGE,
        lw=1.6,
        ls=(0, (4.0, 2.0)),
        zorder=4,
        label=r"mode life / run  $\tau_q/t_{\rm f}$",
    )
    axis.semilogy(
        scaled,
        closure,
        color=GREEN,
        lw=1.6,
        ls=(0, (1.4, 1.6)),
        zorder=4,
        label=r"closure error  $(v^{(0)}q/\mathcal{D}_{R,2})^2$",
    )

    _tidy(axis)
    axis.set_xlim(scaled[0], scaled[-1])
    # Headroom below the lowest curve, so the legend never sits on the data.
    axis.set_ylim(6e-5, 40.0)
    axis.set_xlabel(
        r"$qR$   (zeros of $J_1$ marked)", fontsize=8, color=MUTED, labelpad=1
    )
    axis.legend(
        loc="lower left",
        fontsize=6.2,
        frameon=False,
        labelcolor="linecolor",
        handlelength=2.2,
        borderpad=0.1,
        labelspacing=0.3,
    )
    axis.annotate(
        rf"$qR={chosen:.2f}$",
        (chosen, 30.0),
        xytext=(3, 0),
        textcoords="offset points",
        color=INK,
        fontsize=7,
        ha="left",
        va="top",
        path_effects=_HALO,
    )


def _panel_letter(axis, letter: str, *, inside: bool = False) -> None:
    axis.annotate(
        f"({letter})",
        (0.01, 1.0) if not inside else (0.03, 0.97),
        xycoords="axes fraction",
        fontsize=9,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top" if inside else "bottom",
        path_effects=_HALO if inside else None,
    )


def render_landscape_report(
    config: ProjectConfig,
    figure_path: str | Path,
    field: ChemicalField | None = None,
    *,
    also_png: bool = True,
) -> Path:
    """Write the nutrient-landscape figure: what it is, and why this mode."""

    field = create_field(config.field) if field is None else field
    if not hasattr(field, "wavevector"):
        raise ValueError("the landscape figure is written for a single cosine mode")
    sigma = resolve_initial_gaussian_sigma(
        config.density, inferred_characteristic_length(config, field)
    )
    chosen = float(np.hypot(*field.wavevector)) * config.domain.radius

    figure = Figure(figsize=(7.1, 2.35), dpi=300)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(0.86, 1.0, 1.0),
        left=0.008,
        right=0.978,
        top=0.855,
        bottom=0.185,
        wspace=0.42,
    )
    map_axis = figure.add_subplot(grid[0, 0])
    cut_axis = figure.add_subplot(grid[0, 1])
    design_axis = figure.add_subplot(grid[0, 2])

    _plot_map(map_axis, config, field, sigma)
    _plot_cut(cut_axis, config, field, sigma)
    _plot_design(design_axis, config, sigma, chosen)
    _panel_letter(map_axis, "a", inside=True)
    _panel_letter(cut_axis, "b")
    _panel_letter(design_axis, "c")

    output = Path(figure_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    if also_png:
        figure.savefig(output.with_suffix(".png"))
    return output


def landscape_summary(
    config: ProjectConfig, field: ChemicalField | None = None
) -> dict[str, float]:
    """The dimensionless numbers quoted beside the figure."""

    field = create_field(config.field) if field is None else field
    sigma = resolve_initial_gaussian_sigma(
        config.density, inferred_characteristic_length(config, field)
    )
    wavenumber = float(np.hypot(*field.wavevector))
    scaled = wavenumber * config.domain.radius
    diffusivity = effective_diffusivity(config)
    return {
        "qR": scaled,
        "wavelength_over_radius": 2.0 * np.pi / scaled,
        "amplitude": float(field.amplitude),
        "phase": float(field.phase),
        "disk_average_residue": float(abs(2.0 * special.j1(scaled) / scaled)),
        "release_window": float(np.exp(-0.5 * (wavenumber * sigma) ** 2)),
        "q_sigma_0": float(wavenumber * sigma),
        "closure_error": float(
            (config.kinetics.v0 * wavenumber / _second_harmonic_rate(config)) ** 2
        ),
        "mode_time": float(1.0 / (diffusivity * wavenumber**2)),
        # First Neumann mode of the disk: the no-flux eigenvalue is a zero of
        # J_1', not of J_1.
        "domain_time": float(
            config.domain.radius**2 / (diffusivity * special.jnp_zeros(1, 1)[0] ** 2)
        ),
        "polarity_rate": float(polarity_relaxation_rate(config)),
    }


__all__ = ["landscape_summary", "render_landscape_report"]
