"""Interactive real-time playback."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..angles import heading_vector
from ..config import ProjectConfig
from ..fields import ChemicalField
from ..simulator import Trajectory
from .common import (
    field_image,
    prepare_axis,
    trail_start_index,
    video_frame_indices,
)


def play_live(
    trajectory: Trajectory,
    field: ChemicalField,
    config: ProjectConfig,
    playback_speed: float = 1.0,
) -> None:
    """Play a trajectory at physical real time in an interactive window."""

    values, extent = field_image(field, config.domain.radius, config.video.field_grid)
    figure, ax = plt.subplots(figsize=(8.2, 7.2))
    prepare_axis(
        ax,
        values,
        extent,
        1.0,
        title=f"Case {trajectory.case_number}: {trajectory.case_title}",
        colorbar=True,
    )

    (trail,) = ax.plot([], [], color="#ffdd57", lw=2.0, alpha=0.92, zorder=2)
    (particle,) = ax.plot([], [], "o", ms=8, color="#ff496c", mec="white", zorder=5)
    direction = ax.quiver(
        [0.0],
        [0.0],
        [1.0],
        [0.0],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.008,
        zorder=6,
    )
    tumble_ring = ax.scatter(
        [], [], s=220, facecolors="none", edgecolors="#ff3344", lw=2.2, zorder=7
    )
    status = ax.text(
        0.02,
        0.02,
        "",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        color="white",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 5},
        zorder=8,
    )

    frame_indices = video_frame_indices(
        trajectory.times, config.video.fps, playback_speed
    )
    arrow_length = 0.08

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        start = trail_start_index(trajectory.times, index, config.video.trail_time)
        xy = trajectory.positions[index]
        direction_xy = arrow_length * heading_vector(trajectory.headings[index])

        trail.set_data(
            trajectory.positions[start : index + 1, 0],
            trajectory.positions[start : index + 1, 1],
        )
        particle.set_data([xy[0]], [xy[1]])
        direction.set_offsets(xy[None, :])
        direction.set_UVC([direction_xy[0]], [direction_xy[1]])
        if trajectory.tumble_counts[index] > 0:
            tumble_ring.set_offsets(xy[None, :])
        else:
            tumble_ring.set_offsets(np.empty((0, 2)))

        status.set_text(
            f"t = {trajectory.times[index]:9.3g}\n"
            f"c = {trajectory.concentration[index]:7.3f}   "
            f"S = {trajectory.alignment[index]:+7.3f}\n"
            f"v = {trajectory.speed[index]:7.3f}   "
            f"D_T = {trajectory.translational_diffusivity[index]:7.4f}\n"
            f"D_R = {trajectory.rotational_diffusivity[index]:7.3f}   "
            f"lambda = {trajectory.tumble_rate[index]:7.3f}"
        )
        return trail, particle, direction, tumble_ring, status

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / config.video.fps,
        blit=False,
        repeat=True,
    )
    figure._chem_swimmer_animation = animation
    plt.tight_layout()
    plt.show()
