"""Render the two 10-second, 30-fps supplementary videos.

Video 1 animates the Case-1 validation figure, Video 2 the Cases 2--9
appendix sheet.  Both run on the archives written by ``run.py compare``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, TwoSlopeNorm

from chem_swimmer.cases import CASES
from chem_swimmer.config import load_config
from chem_swimmer.density import PositionPDFResult, load_position_pdf
from chem_swimmer.fields import create_field
from chem_swimmer.visualization.comparison_plot import (
    _disk_mesh,
    _display_density,
    _display_polarity,
    _masked,
    _nearest_time_index,
)
from chem_swimmer.visualization.manuscript_plot import (
    _DIFFERENCE_CMAP,
    PAPER_COLUMN_LABELS,
    PAPER_REPORT_TIME,
    _extent,
    _landscape_guides,
    _paper_methods,
    shared_paper_crop,
    shared_paper_limits,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"
VIDEOS = ROOT / "results" / "videos"


def _field():
    """The landscape, so the frames carry the same crest marks as the figures."""

    return create_field(load_config(ROOT / "config" / "default.toml").field)


def _load() -> list[tuple[str, PositionPDFResult]]:
    return _paper_methods(
        {
            "micro": load_position_pdf(DATA / "pdf_cases_00_09.npz"),
            "macro": load_position_pdf(DATA / "pdf_macro_cases_00_09.npz"),
            "analytic": load_position_pdf(DATA / "pdf_analytic_cases_00_09.npz"),
        }
    )


def _frame_values(
    result: PositionPDFResult, case_number: int, time: float
) -> tuple[np.ndarray, np.ndarray]:
    case_index = result.case_numbers.index(case_number)
    time_index = _nearest_time_index(result, time)
    density = _display_density(result, case_index, time_index)
    px, py = _display_polarity(result, case_index, time_index)
    return density, np.hypot(px, py)


def _style_axis(axis, crop: float) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(-crop, crop)
    axis.set_ylim(-crop, crop)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("0.25")
        spine.set_linewidth(0.55)


def render_video_one(methods: list[tuple[str, PositionPDFResult]]) -> Path:
    final_time = min(
        PAPER_REPORT_TIME, *(float(result.times[-1]) for _, result in methods)
    )
    synchronized_cases = tuple(range(1, 10))
    density_limit, polarity_limit = shared_paper_limits(
        methods, synchronized_cases, final_time, 99.5
    )
    crop = shared_paper_crop(methods, synchronized_cases, final_time)
    density_norm = TwoSlopeNorm(vmin=-density_limit, vcenter=0, vmax=density_limit)
    polarity_norm = Normalize(0, polarity_limit)
    field = _field()
    figure, axes = plt.subplots(2, 3, figsize=(8.0, 5.6), dpi=120)
    figure.subplots_adjust(
        left=0.07, right=0.93, bottom=0.16, top=0.88, hspace=0.08, wspace=0.07
    )
    images = []
    for column, (name, result) in enumerate(methods):
        density, polarity = _frame_values(result, 1, 0.0)
        _, _, mask = _disk_mesh(result.coordinates)
        images.append(
            axes[0, column].imshow(
                _masked(density, mask),
                origin="lower",
                extent=_extent(result.coordinates),
                cmap=_DIFFERENCE_CMAP,
                norm=density_norm,
                interpolation="bilinear",
            )
        )
        images.append(
            axes[1, column].imshow(
                _masked(polarity, mask),
                origin="lower",
                extent=_extent(result.coordinates),
                cmap="cividis",
                norm=polarity_norm,
                interpolation="bilinear",
            )
        )
        axes[0, column].set_title(
            PAPER_COLUMN_LABELS.get(name, name), fontsize=11, fontweight="bold"
        )
        _style_axis(axes[0, column], crop)
        _style_axis(axes[1, column], crop)
        _landscape_guides(axes[0, column], field, crop)
    figure.text(
        0.02,
        0.66,
        r"Density  $\rho_1-\rho_0$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.28,
        r"Polarity  $|\mathbf{p}_1-\mathbf{p}_0|$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    time_text = figure.suptitle("", fontsize=11)
    density_cax = figure.add_axes((0.12, 0.085, 0.34, 0.018))
    polarity_cax = figure.add_axes((0.57, 0.085, 0.34, 0.018))
    density_bar = figure.colorbar(
        images[0], cax=density_cax, orientation="horizontal", label=r"$\rho_1-\rho_0$"
    )
    density_bar.set_ticks((-density_limit, 0.0, density_limit))
    polarity_bar = figure.colorbar(
        images[1],
        cax=polarity_cax,
        orientation="horizontal",
        label=r"$|\mathbf{p}_1-\mathbf{p}_0|$",
    )
    polarity_bar.set_ticks((0.0, polarity_limit / 2.0, polarity_limit))
    times = np.linspace(0.0, final_time, 300)
    output = VIDEOS / "video-1-method-comparison-30fps.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=30,
        codec="libx264",
        bitrate=6000,
        extra_args=("-pix_fmt", "yuv420p", "-movflags", "+faststart"),
    )
    with writer.saving(figure, output, 120):
        for time in times:
            for column, (_, result) in enumerate(methods):
                density, polarity = _frame_values(result, 1, float(time))
                _, _, mask = _disk_mesh(result.coordinates)
                images[2 * column].set_data(_masked(density, mask))
                images[2 * column + 1].set_data(_masked(polarity, mask))
            time_text.set_text(rf"Case 1  |  $t={time:.5f}$")
            writer.grab_frame()
    plt.close(figure)
    return output


def _save_frames(
    figure, output: Path, update, times: np.ndarray, *, bitrate: int
) -> None:
    writer = FFMpegWriter(
        fps=30,
        codec="libx264",
        bitrate=bitrate,
        extra_args=("-pix_fmt", "yuv420p", "-movflags", "+faststart"),
    )
    with writer.saving(figure, output, figure.dpi):
        for time in times:
            update(float(time))
            writer.grab_frame()


def render_video_two(methods: list[tuple[str, PositionPDFResult]]) -> Path:
    cases = tuple(range(2, 10))
    final_time = min(
        PAPER_REPORT_TIME, *(float(result.times[-1]) for _, result in methods)
    )
    synchronized_cases = tuple(range(1, 10))
    density_limit, polarity_limit = shared_paper_limits(
        methods, synchronized_cases, final_time, 99.5
    )
    crop = shared_paper_crop(methods, synchronized_cases, final_time)
    density_norm = TwoSlopeNorm(vmin=-density_limit, vcenter=0, vmax=density_limit)
    polarity_norm = Normalize(0, polarity_limit)
    field = _field()
    figure, axes = plt.subplots(8, 6, figsize=(8.3, 10.6), dpi=100)
    figure.subplots_adjust(
        left=0.105, right=0.985, bottom=0.115, top=0.89, hspace=0.07, wspace=0.055
    )
    images: list = []
    for row, case_number in enumerate(cases):
        for column, (_, result) in enumerate(methods):
            density, polarity = _frame_values(result, case_number, 0.0)
            _, _, mask = _disk_mesh(result.coordinates)
            images.append(
                axes[row, column].imshow(
                    _masked(density, mask),
                    origin="lower",
                    extent=_extent(result.coordinates),
                    cmap=_DIFFERENCE_CMAP,
                    norm=density_norm,
                    interpolation="bilinear",
                )
            )
            images.append(
                axes[row, 3 + column].imshow(
                    _masked(polarity, mask),
                    origin="lower",
                    extent=_extent(result.coordinates),
                    cmap="cividis",
                    norm=polarity_norm,
                    interpolation="bilinear",
                )
            )
        axes[row, 0].text(
            -0.08,
            0.5,
            f"Case {case_number}  {CASES[case_number].math_label}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7.8,
            fontweight="bold",
        )
        for axis in axes[row]:
            _style_axis(axis, crop)
        for axis in axes[row][:3]:
            _landscape_guides(axis, field, crop)
    headings = [PAPER_COLUMN_LABELS.get(name, name) for name, _ in methods] * 2
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading, fontsize=10.5, fontweight="bold", pad=5)
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
    time_text = figure.text(0.545, 0.992, "", ha="center", va="top", fontsize=10.5)
    density_cax = figure.add_axes((0.135, 0.055, 0.35, 0.014))
    polarity_cax = figure.add_axes((0.595, 0.055, 0.35, 0.014))
    density_bar = figure.colorbar(
        images[0],
        cax=density_cax,
        orientation="horizontal",
        label=r"Shared $\rho_i-\rho_0$",
    )
    density_bar.set_ticks((-density_limit, 0.0, density_limit))
    polarity_bar = figure.colorbar(
        images[1],
        cax=polarity_cax,
        orientation="horizontal",
        label=r"Shared $|\mathbf{p}_i-\mathbf{p}_0|$",
    )
    polarity_bar.set_ticks((0.0, polarity_limit / 2.0, polarity_limit))
    times = np.linspace(0.0, final_time, 300)
    output = VIDEOS / "video-2-cases-2-9-summary-30fps.mp4"

    def update(time: float) -> None:
        image_index = 0
        for case_number in cases:
            for _, result in methods:
                density, polarity = _frame_values(result, case_number, time)
                _, _, mask = _disk_mesh(result.coordinates)
                images[image_index].set_data(_masked(density, mask))
                images[image_index + 1].set_data(_masked(polarity, mask))
                image_index += 2
        time_text.set_text(rf"Common time  $t={time:.5f}$")

    _save_frames(figure, output, update, times, bitrate=8000)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=("1", "2", "both"), default="both")
    args = parser.parse_args()
    methods = _load()
    if args.which in {"1", "both"}:
        print(render_video_one(methods))
    if args.which in {"2", "both"}:
        print(render_video_two(methods))


if __name__ == "__main__":
    main()
