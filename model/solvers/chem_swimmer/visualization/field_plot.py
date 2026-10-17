"""Static plot of the dimensionless prescribed chemical field."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from ..config import ProjectConfig
from ..fields import ChemicalField


def render_chemical_field(
    field: ChemicalField,
    config: ProjectConfig,
    output_path: str | Path,
) -> Path:
    """Render ``eta(x,y)`` throughout the dimensionless unit disk."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    configured_grid = config.video.field_grid
    base_grid = 301 if isinstance(configured_grid, str) else int(configured_grid)
    grid_size = max(301, base_grid)
    coordinates = np.linspace(-1.0, 1.0, grid_size)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    points = np.stack((x_grid, y_grid), axis=-1)
    values = np.asarray(field.value(points), dtype=float)
    if values.shape != x_grid.shape:
        raise ValueError("the chemical field returned an incompatible plot shape")
    disk_mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    if not np.all(np.isfinite(values[disk_mask])):
        raise ValueError("the chemical field must be finite throughout the disk")
    masked_values = np.ma.masked_where(~disk_mask, values)

    figure = Figure(figsize=(8.0, 7.2), dpi=150, constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    image = axis.imshow(
        masked_values,
        origin="lower",
        extent=(-1.0, 1.0, -1.0, 1.0),
        interpolation="bilinear",
        cmap="viridis",
        aspect="equal",
    )
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            1.0,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
    )
    axis.set(
        xlabel="x",
        ylabel="y",
        title=r"Dimensionless nutrient field $\eta(x,y)$",
        xlim=(-1.03, 1.03),
        ylim=(-1.03, 1.03),
    )
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.92, pad=0.04)
    colorbar.set_label(r"$\eta$")
    figure.savefig(
        output,
        format="png",
        dpi=150,
        facecolor="white",
    )
    figure.clear()
    return output
