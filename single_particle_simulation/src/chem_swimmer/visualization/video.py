"""MP4 rendering for one case or all eight canonical cases."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from ..angles import heading_vector
from ..cases import get_case
from ..config import ProjectConfig
from ..fields import ChemicalField
from ..progress import progress_bar
from ..simulator import Trajectory
from ..statistics import MSDResult, free_space_baseline_msd
from .common import (
    field_image,
    prepare_axis,
    trail_start_index,
    video_frame_indices,
)
from .encoding import choose_codec

COMPARISON_TITLES = {
    1: "Concentration-dependent diffusion",
    2: "Concentration-dependent rotation",
    3: "Orthokinesis",
    4: "Concentration-dependent tumbling",
    5: "Alignment-gated translation noise",
    6: "Alignment-gated rotation noise",
    7: "Alignment-gated propulsion",
    8: "Local tumble surrogate",
}

CASE_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#6A3D9A",
)


def _writer(config: ProjectConfig) -> FFMpegWriter:
    codec = choose_codec(config.video.codec)
    return FFMpegWriter(
        fps=config.video.fps,
        codec=codec,
        bitrate=config.video.bitrate_kbps,
        extra_args=[
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
        metadata={"title": "Chemically responsive active swimmer"},
    )


def _encode_frames(
    figure,
    update,
    frame_count: int,
    config: ProjectConfig,
    output: Path,
) -> None:
    """Render frames directly so terminal progress includes encoding ETA."""

    writer = _writer(config)
    frames = progress_bar(
        range(frame_count),
        total=frame_count,
        description="Rendering MP4",
        unit="frame",
        enabled=config.performance.progress,
    )
    with writer.saving(figure, output, dpi=config.video.dpi):
        for frame_number in frames:
            update(frame_number)
            writer.grab_frame()


def render_single_video(
    trajectory: Trajectory,
    field: ChemicalField,
    config: ProjectConfig,
    output_path: str | Path,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    values, extent = field_image(field, config.domain.radius, config.video.field_grid)
    figure, ax = plt.subplots(figsize=(8.0, 7.0))
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
    event_text = ax.text(
        0.98,
        0.03,
        "",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#ff3344",
        fontsize=11,
        weight="bold",
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
    frame_indices = video_frame_indices(trajectory.times, config.video.fps)

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        start = trail_start_index(trajectory.times, index, config.video.trail_time)
        xy = trajectory.positions[index]
        vector = 0.08 * heading_vector(trajectory.headings[index])

        trail.set_data(
            trajectory.positions[start : index + 1, 0],
            trajectory.positions[start : index + 1, 1],
        )
        particle.set_data([xy[0]], [xy[1]])
        direction.set_offsets(xy[None, :])
        direction.set_UVC([vector[0]], [vector[1]])

        events = []
        if trajectory.tumble_counts[index]:
            events.append("TUMBLE")
        if trajectory.wall_hit_counts[index]:
            events.append("WALL")
        event_text.set_text("  ".join(events))
        status.set_text(
            f"t = {trajectory.times[index]:9.3g}\n"
            f"c = {trajectory.concentration[index]:7.3f}   "
            f"S = {trajectory.alignment[index]:+7.3f}\n"
            f"v = {trajectory.speed[index]:7.3f}   "
            f"D_T = {trajectory.translational_diffusivity[index]:7.4f}\n"
            f"D_R = {trajectory.rotational_diffusivity[index]:7.3f}   "
            f"lambda = {trajectory.tumble_rate[index]:7.3f}"
        )
        return trail, particle, direction, event_text, status

    figure.tight_layout()
    _encode_frames(figure, update, len(frame_indices), config, output)
    plt.close(figure)
    return output


def render_comparison_video(
    trajectories: list[Trajectory],
    field: ChemicalField,
    config: ProjectConfig,
    output_path: str | Path,
) -> Path:
    if [trajectory.case_number for trajectory in trajectories] != list(range(1, 9)):
        raise ValueError("comparison video expects trajectories for cases 1 through 8")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    values, extent = field_image(field, config.domain.radius, config.video.field_grid)
    figure, axes = plt.subplots(2, 4, figsize=(14.0, 7.2), constrained_layout=True)
    lines = []
    particles = []
    directions = []

    for axis, trajectory in zip(axes.flat, trajectories, strict=True):
        case = get_case(trajectory.case_number)
        prepare_axis(
            axis,
            values,
            extent,
            1.0,
            title=f"{case.number}. {case.math_label}\n{COMPARISON_TITLES[case.number]}",
        )
        axis.title.set_fontsize(9.5)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("")
        axis.set_ylabel("")
        (line,) = axis.plot([], [], color="#ffdd57", lw=1.5, zorder=2)
        (particle,) = axis.plot(
            [], [], "o", ms=5.5, color="#ff496c", mec="white", zorder=5
        )
        direction = axis.quiver(
            [0.0],
            [0.0],
            [1.0],
            [0.0],
            color="white",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.011,
            zorder=6,
        )
        lines.append(line)
        particles.append(particle)
        directions.append(direction)

    clock = figure.suptitle("", fontsize=13, weight="bold")
    frame_indices = video_frame_indices(trajectories[0].times, config.video.fps)

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        for trajectory, line, particle, direction in zip(
            trajectories, lines, particles, directions, strict=True
        ):
            start = trail_start_index(trajectory.times, index, config.video.trail_time)
            xy = trajectory.positions[index]
            vector = 0.08 * heading_vector(trajectory.headings[index])
            line.set_data(
                trajectory.positions[start : index + 1, 0],
                trajectory.positions[start : index + 1, 1],
            )
            particle.set_data([xy[0]], [xy[1]])
            direction.set_offsets(xy[None, :])
            direction.set_UVC([vector[0]], [vector[1]])
        clock.set_text(
            f"Eight canonical mechanisms — t = {trajectories[0].times[index]:.4g}"
        )
        return [*lines, *particles, *directions, clock]

    _encode_frames(figure, update, len(frame_indices), config, output)
    plt.close(figure)
    return output


def render_overlay_msd_video(
    trajectories: list[Trajectory],
    msd: MSDResult,
    field: ChemicalField,
    config: ProjectConfig,
    output_path: str | Path,
) -> Path:
    """Overlay all eight paths in one disk and animate paired ensemble MSD."""

    expected_cases = list(range(1, 9))
    if [trajectory.case_number for trajectory in trajectories] != expected_cases:
        raise ValueError("overlay video expects trajectories for cases 1 through 8")
    if list(msd.case_numbers) != expected_cases:
        raise ValueError("MSD result must contain cases 1 through 8")
    if not np.array_equal(trajectories[0].times, msd.times):
        raise ValueError("trajectory and MSD time grids must match")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    values, extent = field_image(field, config.domain.radius, config.video.field_grid)

    figure, (disk_axis, msd_axis) = plt.subplots(
        1,
        2,
        # With the default 110 dpi this gives an H.264-safe even 1496x726.
        figsize=(13.6, 6.6),
        gridspec_kw={"width_ratios": (1.03, 1.0)},
        constrained_layout=True,
    )
    prepare_axis(
        disk_axis,
        values,
        extent,
        1.0,
        title="One realization: common initial state and common noise",
    )

    path_lines = []
    particles = []
    msd_lines = []
    error_bands = []
    for case_number, color in zip(expected_cases, CASE_COLORS, strict=True):
        case = get_case(case_number)
        label = f"{case.number}: {case.math_label}"
        (path_line,) = disk_axis.plot(
            [],
            [],
            color=color,
            lw=1.7,
            alpha=0.88,
            label=label,
            zorder=2,
        )
        (particle,) = disk_axis.plot(
            [],
            [],
            "o",
            ms=6.0,
            color=color,
            mec="white",
            mew=0.8,
            zorder=6,
        )
        (msd_line,) = msd_axis.plot([], [], color=color, lw=1.9, label=label)
        band = msd_axis.fill_between(
            [],
            [],
            [],
            color=color,
            alpha=0.10,
            linewidth=0.0,
        )
        path_lines.append(path_line)
        particles.append(particle)
        msd_lines.append(msd_line)
        error_bands.append(band)

    initial = np.asarray(config.initial.position) / config.domain.radius
    disk_axis.plot(
        [initial[0]],
        [initial[1]],
        marker="*",
        ms=12,
        color="white",
        mec="black",
        mew=0.6,
        zorder=8,
        label="shared start",
    )
    disk_axis.legend(
        loc="lower left",
        ncol=3,
        fontsize=7.5,
        frameon=True,
        framealpha=0.82,
        borderpad=0.5,
    )

    reference = free_space_baseline_msd(config, msd.times)
    reference = np.where(reference <= 1.0, reference, np.nan)
    msd_axis.plot(
        msd.times,
        reference,
        color="0.25",
        lw=1.4,
        ls="--",
        label="free-space baseline",
    )
    msd_axis.set_xlim(0.0, msd.times[-1])
    maximum_msd = float(np.nanmax(msd.mean + msd.standard_error))
    msd_axis.set_ylim(0.0, min(1.03, maximum_msd * 1.18 + 1.0e-9))
    msd_axis.set_xlabel("t")
    msd_axis.set_ylabel("MSD")
    msd_axis.set_title(f"Ensemble mean squared displacement (M = {msd.ensemble_size})")
    msd_axis.grid(alpha=0.25)
    msd_axis.legend(loc="upper left", ncol=3, fontsize=7.5, framealpha=0.85)
    clock = figure.suptitle("", fontsize=13, weight="bold")

    frame_indices = video_frame_indices(trajectories[0].times, config.video.fps)

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        for case_index, (trajectory, path_line, particle, msd_line, band) in enumerate(
            zip(
                trajectories,
                path_lines,
                particles,
                msd_lines,
                error_bands,
                strict=True,
            )
        ):
            xy = trajectory.positions[index]
            path_line.set_data(
                trajectory.positions[: index + 1, 0],
                trajectory.positions[: index + 1, 1],
            )
            particle.set_data([xy[0]], [xy[1]])
            time_values = msd.times[: index + 1]
            mean_values = msd.mean[case_index, : index + 1]
            msd_line.set_data(time_values, mean_values)

            if config.statistics.show_standard_error and len(time_values) > 1:
                error = msd.standard_error[case_index, : index + 1]
                lower = np.maximum(0.0, mean_values - error)
                upper = mean_values + error
                polygon = np.column_stack(
                    (
                        np.concatenate((time_values, time_values[::-1])),
                        np.concatenate((lower, upper[::-1])),
                    )
                )
                band.set_verts([polygon])
            else:
                band.set_verts([])

        clock.set_text(
            f"Canonical mechanisms under paired common noise — "
            f"t = {trajectories[0].times[index]:.4g}"
        )
        return [*path_lines, *particles, *msd_lines, *error_bands, clock]

    _encode_frames(figure, update, len(frame_indices), config, output)
    plt.close(figure)
    return output
