"""Shared plotting helpers."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from numpy.typing import NDArray

from ..fields import ChemicalField


def field_image(
    field: ChemicalField, radius: float, resolution: int
) -> tuple[NDArray[np.float64], tuple[float, float, float, float]]:
    coordinates = np.linspace(-radius, radius, resolution)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    points = np.stack((x_grid, y_grid), axis=-1)
    values = np.asarray(field.value(points), dtype=float)
    values[x_grid * x_grid + y_grid * y_grid > radius**2] = np.nan
    return values, (-1.0, 1.0, -1.0, 1.0)


def prepare_axis(
    ax: Axes,
    field_values: NDArray[np.float64],
    extent: tuple[float, float, float, float],
    radius: float,
    *,
    title: str = "",
    colorbar: bool = False,
) -> None:
    finite = field_values[np.isfinite(field_values)]
    norm = Normalize(float(finite.min()), float(finite.max()))
    image = ax.imshow(
        field_values,
        origin="lower",
        extent=extent,
        cmap="viridis",
        norm=norm,
        interpolation="bilinear",
        alpha=0.82,
        zorder=0,
    )
    ax.add_patch(
        Circle((0.0, 0.0), radius, fill=False, color="white", lw=1.8, zorder=4)
    )
    ax.set_xlim(-1.04 * radius, 1.04 * radius)
    ax.set_ylim(-1.04 * radius, 1.04 * radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if colorbar:
        ax.figure.colorbar(
            image, ax=ax, fraction=0.046, pad=0.04, label=r"$\eta(\mathbf r)$"
        )


def video_frame_indices(
    times: NDArray[np.float64], fps: int, playback_speed: float = 1.0
) -> NDArray[np.int64]:
    video_seconds = min(10.0, 10.0 / playback_speed)
    frame_count = max(2, round(video_seconds * fps))
    frame_times = np.linspace(times[0], times[-1], frame_count)
    indices = np.searchsorted(times, frame_times, side="left")
    return np.clip(indices, 0, len(times) - 1)


def trail_start_index(
    times: NDArray[np.float64], current_index: int, trail_time: float
) -> int:
    return int(np.searchsorted(times, times[current_index] - trail_time, side="left"))
