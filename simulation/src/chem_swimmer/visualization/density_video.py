"""Fast direct-raster MP4 rendering for the Cases 0-9 spatial PDFs."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..cases import CASES
from ..config import ProjectConfig
from ..density import PositionPDFResult
from ..progress import progress_bar
from .encoding import choose_codec


def _terse_case_title(number: int) -> str:
    """``"Case N: <quantity> response to <eta|zeta>"``, derived from
    ``cases.CASES`` (the single canonical description of all cases)
    rather than a second, hand-maintained copy of the same text -- keeps
    this compact video-row-label style in sync with ``cases.py`` by
    construction instead of by discipline.
    """

    response_name = CASES[number].response_name
    if not response_name:
        return f"Case {number}: background coefficients only"
    if response_name == "chi_theta":
        return f"Case {number}: continuous angular steering"
    quantity, signal = response_name.rsplit("_", 1)
    return f"Case {number}: {quantity} response to {signal}"


CASE_TITLES = tuple(_terse_case_title(number) for number in sorted(CASES))


def _detail_grid_geometry(
    frame_size: int,
    header: int,
    footer: int,
    margin: int,
    gap: int,
    case_count: int,
) -> tuple[int, int, int, int]:
    """Return ``columns, cell_size, left, top`` for a compact case grid."""

    columns = 3 if case_count <= 9 else 4
    rows = ceil(case_count / columns)
    available_width = frame_size - 2 * margin
    available_height = frame_size - header - footer - 2 * margin
    cell_size = min(
        (available_width - (columns - 1) * gap) // columns,
        (available_height - (rows - 1) * gap) // rows,
    )
    grid_width = columns * cell_size + (columns - 1) * gap
    grid_height = rows * cell_size + (rows - 1) * gap
    grid_left = (frame_size - grid_width) // 2
    grid_top = header + margin + (available_height - grid_height) // 2
    return columns, cell_size, grid_left, grid_top


def _case_specific_linear_limits(
    differences: np.ndarray,
    coordinates: np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed robust display limits and exact maxima for every case."""

    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    disk_mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    absolute = np.abs(differences[:, :, disk_mask])
    if not np.all(np.isfinite(absolute)):
        raise ValueError("probability-density differences must be finite")
    limits = np.percentile(
        absolute,
        percentile,
        axis=(1, 2),
    )
    maxima = np.max(absolute, axis=(1, 2))
    return np.asarray(limits, dtype=float), np.asarray(maxima, dtype=float)


def synchronized_case_linear_limits(
    results: tuple[PositionPDFResult, ...] | list[PositionPDFResult],
    percentile: float,
) -> np.ndarray:
    """Resolve one per-case linear range jointly across several methods."""

    if not results:
        raise ValueError("at least one density result is required")
    case_numbers = results[0].case_numbers
    samples: list[list[np.ndarray]] = [[] for _ in case_numbers]
    for result in results:
        if result.case_numbers != case_numbers:
            raise ValueError("density results use inconsistent case numbers")
        density = (
            result.density
            if result.difference_density is None
            else result.difference_density
        )
        differences = density - density[0:1]
        x_grid, y_grid = np.meshgrid(result.coordinates, result.coordinates)
        disk_mask = x_grid * x_grid + y_grid * y_grid <= 1.0
        for case_index in range(len(case_numbers)):
            values = np.abs(differences[case_index, :, disk_mask]).reshape(-1)
            if not np.all(np.isfinite(values)):
                raise ValueError("probability-density differences must be finite")
            samples[case_index].append(values)
    return np.asarray(
        [
            np.percentile(np.concatenate(case_samples), percentile)
            for case_samples in samples
        ],
        dtype=float,
    )


def synchronized_global_linear_limits(
    results: tuple[PositionPDFResult, ...] | list[PositionPDFResult],
    percentile: float,
) -> np.ndarray:
    """Resolve one linear range shared by every case and method."""

    if not results:
        raise ValueError("at least one density result is required")
    case_numbers = results[0].case_numbers
    samples: list[np.ndarray] = []
    for result in results:
        if result.case_numbers != case_numbers:
            raise ValueError("density results use inconsistent case numbers")
        density = (
            result.density
            if result.difference_density is None
            else result.difference_density
        )
        differences = density - density[0:1]
        x_grid, y_grid = np.meshgrid(result.coordinates, result.coordinates)
        disk_mask = x_grid * x_grid + y_grid * y_grid <= 1.0
        values = np.abs(differences[1:, :, disk_mask]).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("probability-density differences must be finite")
        samples.append(values)
    shared_limit = float(np.percentile(np.concatenate(samples), percentile))
    return np.full(len(case_numbers), shared_limit, dtype=float)


def _normalize_linear_difference(
    values: np.ndarray,
    limit: float,
) -> np.ndarray:
    """Linearly map one case's signed physical differences to ``[-1, 1]``."""

    if limit <= 0.0:
        return np.zeros_like(values)
    return np.clip(values / limit, -1.0, 1.0)


def _method_label(result: PositionPDFResult, *, difference: bool = False) -> str:
    if result.method in {"moment", "macro", "analytic"}:
        bandwidth = (
            result.bandwidth_fraction
            if not difference or result.difference_bandwidth_fraction is None
            else result.difference_bandwidth_fraction
        )
        symbol = "b_delta" if difference else "b"
        closure = {
            "moment": "polar (Q=0)",
            "macro": "macroscopic polar (Q=0)",
            "analytic": "polar Fourier/Green (Q=0)",
        }[result.method]
        return (
            f"{closure}-closed PDE, N = {len(result.coordinates)}, "
            f"{symbol} = {bandwidth:.3g}"
        )
    bandwidth = (
        result.bandwidth_fraction
        if not difference or result.difference_bandwidth_fraction is None
        else result.difference_bandwidth_fraction
    )
    symbol = "b_delta" if difference else "b"
    return f"M = {result.ensemble_size}, {symbol} = {bandwidth:.3g}"


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def _color_lut() -> np.ndarray:
    anchor_positions = np.linspace(0.0, 1.0, 6)
    anchors = np.asarray(
        (
            (0, 0, 4),
            (59, 15, 112),
            (140, 41, 129),
            (222, 73, 104),
            (254, 159, 109),
            (252, 253, 191),
        ),
        dtype=float,
    )
    positions = np.linspace(0.0, 1.0, 256)
    return np.asarray(
        np.round(
            np.column_stack(
                [
                    np.interp(positions, anchor_positions, anchors[:, channel])
                    for channel in range(3)
                ]
            )
        ),
        dtype=np.uint8,
    )


def _difference_color_lut() -> np.ndarray:
    anchor_positions = np.linspace(0.0, 1.0, 7)
    anchors = np.asarray(
        (
            (49, 54, 149),
            (69, 117, 180),
            (210, 226, 240),
            (248, 248, 248),
            (253, 219, 199),
            (214, 96, 77),
            (103, 0, 31),
        ),
        dtype=float,
    )
    positions = np.linspace(0.0, 1.0, 256)
    return np.asarray(
        np.round(
            np.column_stack(
                [
                    np.interp(positions, anchor_positions, anchors[:, channel])
                    for channel in range(3)
                ]
            )
        ),
        dtype=np.uint8,
    )


def _ffmpeg_command(
    config: ProjectConfig,
    output: Path,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> list[str]:
    codec = choose_codec(config.video.codec)
    width = config.density.frame_size if frame_width is None else frame_width
    height = config.density.frame_size if frame_height is None else frame_height
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(config.density.video_fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        codec,
        "-b:v",
        f"{config.video.bitrate_kbps}k",
    ]
    if codec == "h264_videotoolbox":
        command.extend(("-realtime", "1", "-prio_speed", "1"))
    elif codec == "libx264":
        command.extend(("-preset", "fast"))
    command.extend(
        (
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        )
    )
    return command


def _density_frame_indices(
    result: PositionPDFResult,
    config: ProjectConfig,
    *,
    frame_times: np.ndarray | None = None,
) -> np.ndarray:
    """Indices into ``result.times`` nearest an evenly spaced display schedule.

    ``frame_times`` lets several :class:`PositionPDFResult` objects (e.g.
    micro/macro/analytic, which are not guaranteed to share the exact same
    saved time array) be driven from one shared schedule so a comparison
    video advances all of its panels in lockstep; when omitted, the schedule
    is generated from ``result`` itself, exactly as before.
    """

    if frame_times is None:
        maximum_frames = max(
            2,
            round(config.density.max_video_seconds * config.density.video_fps),
        )
        frame_count = min(maximum_frames, len(result.times))
        frame_times = np.linspace(
            result.times[0],
            result.times[-1],
            frame_count,
        )
    right = np.searchsorted(result.times, frame_times, side="left")
    right = np.clip(right, 0, len(result.times) - 1)
    left = np.maximum(0, right - 1)
    choose_left = np.abs(frame_times - result.times[left]) <= np.abs(
        result.times[right] - frame_times
    )
    return np.where(choose_left, left, right).astype(np.int64)


def shared_display_times(
    results: tuple[PositionPDFResult, ...] | list[PositionPDFResult],
    config: ProjectConfig,
) -> np.ndarray:
    """One evenly spaced time schedule spanning every result's saved range.

    Used so a multi-method comparison video (micro/macro/analytic side by
    side) advances every column through matching physical times, even though
    each method's own saved-time array is stored and thinned independently.
    """

    if not results:
        raise ValueError("at least one density result is required")
    start = min(float(result.times[0]) for result in results)
    end = max(float(result.times[-1]) for result in results)
    maximum_frames = max(
        2,
        round(config.density.max_video_seconds * config.density.video_fps),
    )
    frame_count = min(maximum_frames, max(len(result.times) for result in results))
    return np.linspace(start, end, frame_count)


_GRID_HEADER = 74
_GRID_FOOTER = 58
_GRID_MARGIN = 16
_GRID_GAP = 8


_GRID_ROW_LABEL = 22


def _comparison_grid_geometry(
    frame_size_hint: int,
    n_rows: int,
    n_columns: int,
) -> tuple[int, int, int]:
    """``(cell_size, frame_width, frame_height)`` for an ``n_rows`` x
    ``n_columns`` grid of square plot panels (one labeled text row above
    each panel row for the case title), sized off the same
    ``density.frame_size`` config knob used by the single-method grids."""

    cell_size = max(140, frame_size_hint // 5)
    row_block = _GRID_ROW_LABEL + cell_size
    width = 2 * _GRID_MARGIN + n_columns * cell_size + (n_columns - 1) * _GRID_GAP
    height = (
        _GRID_HEADER
        + _GRID_FOOTER
        + 2 * _GRID_MARGIN
        + n_rows * row_block
        + (n_rows - 1) * _GRID_GAP
    )
    return cell_size, width + width % 2, height + height % 2


def _method_display_name(method: str) -> str:
    return {
        "micro": "microscopic simulation",
        "moment": "polar closure Q=0 (finite volume)",
        "macro": "macroscopic dynamic p, Q=0 (finite volume)",
        "analytic": "dynamic p Fourier/Green solution (infinite plane)",
    }.get(method, method)


def _pipe_frames_to_ffmpeg(
    config: ProjectConfig,
    output: Path,
    density_indices: np.ndarray,
    description: str,
    frame_builder: Callable[[int], bytes],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> None:
    process = subprocess.Popen(
        _ffmpeg_command(
            config, output, frame_width=frame_width, frame_height=frame_height
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        process.kill()
        process.wait()
        raise RuntimeError("FFmpeg did not provide a video input pipe")

    previous_index = -1
    frame_bytes = b""
    frames = progress_bar(
        density_indices,
        total=len(density_indices),
        description=description,
        unit="frame",
        enabled=config.performance.progress,
    )
    try:
        for density_index_value in frames:
            density_index = int(density_index_value)
            if density_index != previous_index:
                frame_bytes = frame_builder(density_index)
                previous_index = density_index
            process.stdin.write(frame_bytes)
        process.stdin.close()
    except BrokenPipeError as exc:
        error = process.stderr.read().decode("utf-8", errors="replace")
        process.wait()
        output.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg encoding failed: {error}") from exc
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise

    error = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg encoding failed: {error}")


def _render_density_frame(
    result: PositionPDFResult,
    density_index: int,
    frame_size: int,
    lut: np.ndarray,
    title_font: ImageFont.FreeTypeFont,
    case_font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont,
) -> bytes:
    background = (18, 20, 25)
    frame = Image.new("RGB", (frame_size, frame_size), background)
    draw = ImageDraw.Draw(frame)
    header = 64
    footer = 34
    margin = 18
    gap = 10
    columns, cell_size, grid_left, grid_top = _detail_grid_geometry(
        frame_size, header, footer, margin, gap, len(CASE_TITLES)
    )
    plot_size = cell_size - 30

    values = result.density[:, density_index]
    disk_values = values[:, values.shape[-1] // 2, values.shape[-1] // 2]
    del disk_values
    x = result.coordinates
    x_grid, y_grid = np.meshgrid(x, x)
    mask = x_grid * x_grid + y_grid * y_grid <= 1.0

    time_text = (
        f"Spatial probability density — t = "
        f"{result.times[density_index]:.4g}, {_method_label(result)}"
    )
    draw.text(
        (frame_size // 2, 18),
        time_text,
        font=title_font,
        fill=(245, 245, 245),
        anchor="ma",
    )

    for case_index, title in enumerate(CASE_TITLES):
        row, column = divmod(case_index, columns)
        left = grid_left + column * (cell_size + gap)
        top = grid_top + row * (cell_size + gap)
        draw.text(
            (left + cell_size // 2, top),
            title,
            font=case_font,
            fill=(235, 235, 238),
            anchor="ma",
        )

        # Plot the dimensionless density itself.  The display range is fixed
        # at [0, 1] for every case and time; values above one saturate in the
        # image but remain unchanged in the saved PDF data.
        normalized = np.clip(values[case_index], 0.0, 1.0)
        color_indices = np.asarray(
            np.round(255.0 * normalized),
            dtype=np.uint8,
        )
        rgb = lut[color_indices]
        alpha = np.where(mask, 255, 0).astype(np.uint8)
        rgba = np.dstack((rgb, alpha))
        density_image = Image.fromarray(
            np.flipud(rgba),
            mode="RGBA",
        ).resize(
            (plot_size, plot_size),
            resample=Image.Resampling.BILINEAR,
        )
        plot_left = left + (cell_size - plot_size) // 2
        plot_top = top + 24
        frame.paste(
            density_image,
            (plot_left, plot_top),
            density_image,
        )
        draw.ellipse(
            (
                plot_left,
                plot_top,
                plot_left + plot_size - 1,
                plot_top + plot_size - 1,
            ),
            outline=(245, 245, 245),
            width=2,
        )
    bar_width = 220
    bar_height = 13
    bar_left = frame_size - margin - bar_width - 55
    bar_top = frame_size - 25
    bar_values = np.linspace(0.0, 1.0, 256)
    bar_indices = np.asarray(np.round(255.0 * bar_values), dtype=np.uint8)
    gradient = np.tile(lut[bar_indices][None, :, :], (bar_height, 1, 1))
    gradient = Image.fromarray(gradient, mode="RGB").resize(
        (bar_width, bar_height),
    )
    frame.paste(gradient, (bar_left, bar_top))
    draw.text(
        (margin, frame_size - 27),
        "p(x,y,t)   (dimensionless; fixed 0–1 display range; p > 1 saturates)",
        font=small_font,
        fill=(205, 207, 214),
    )
    draw.text(
        (bar_left - 6, bar_top + bar_height // 2),
        "0",
        font=small_font,
        fill=(210, 210, 215),
        anchor="rm",
    )
    draw.text(
        (bar_left + bar_width + 5, bar_top + bar_height // 2),
        "1",
        font=small_font,
        fill=(210, 210, 215),
        anchor="lm",
    )
    return np.asarray(frame, dtype=np.uint8).tobytes()


def render_position_pdf_video(
    result: PositionPDFResult,
    config: ProjectConfig,
    output_path: str | Path,
) -> Path:
    """Pipe rasterized density frames directly to FFmpeg without Matplotlib draws."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    density_indices = _density_frame_indices(result, config)

    lut = _color_lut()
    title_font = _font(max(18, config.density.frame_size // 48))
    case_font = _font(max(13, config.density.frame_size // 72))
    small_font = _font(max(12, config.density.frame_size // 86))
    density_max = float(np.max(result.density))
    if not np.isfinite(density_max) or density_max <= 0:
        raise ValueError("probability density must have a positive finite maximum")
    _pipe_frames_to_ffmpeg(
        config,
        output,
        density_indices,
        "Rendering density MP4",
        lambda density_index: _render_density_frame(
            result,
            density_index,
            config.density.frame_size,
            lut,
            title_font,
            case_font,
            small_font,
        ),
    )
    return output


def _render_difference_frame(
    result: PositionPDFResult,
    density_index: int,
    frame_size: int,
    lut: np.ndarray,
    title_font: ImageFont.FreeTypeFont,
    case_font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont,
    case_limits: np.ndarray,
) -> bytes:
    background = (18, 20, 25)
    frame = Image.new("RGB", (frame_size, frame_size), background)
    draw = ImageDraw.Draw(frame)
    header = 64
    footer = 50
    margin = 18
    gap = 10
    columns, cell_size, grid_left, grid_top = _detail_grid_geometry(
        frame_size, header, footer, margin, gap, len(CASE_TITLES)
    )
    plot_size = cell_size - 30

    difference_density = (
        result.density
        if result.difference_density is None
        else result.difference_density
    )
    values = difference_density[:, density_index]
    x = result.coordinates
    x_grid, y_grid = np.meshgrid(x, x)
    mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    globally_shared = bool(
        len(case_limits) < 2
        or np.allclose(
            case_limits[1:],
            case_limits[1],
            rtol=1.0e-12,
            atol=1.0e-15,
        )
    )

    draw.text(
        (frame_size // 2, 18),
        (
            "Signed PDF difference from Case 0 — "
            f"t = {result.times[density_index]:.4g}, "
            f"{_method_label(result, difference=True)}"
        ),
        font=title_font,
        fill=(245, 245, 245),
        anchor="ma",
    )

    for case_index, title in enumerate(CASE_TITLES):
        row, column = divmod(case_index, columns)
        left = grid_left + column * (cell_size + gap)
        top = grid_top + row * (cell_size + gap)
        range_text = (
            ""
            if case_index == 0 or globally_shared
            else f"  [±{case_limits[case_index]:.2g}]"
        )
        draw.text(
            (left + cell_size // 2, top),
            f"{title}{range_text}",
            font=case_font,
            fill=(235, 235, 238),
            anchor="ma",
        )

        difference = values[case_index] - values[0]
        normalized = 0.5 + 0.5 * _normalize_linear_difference(
            difference,
            float(case_limits[case_index]),
        )
        color_indices = np.asarray(
            np.round(255.0 * normalized),
            dtype=np.uint8,
        )
        rgb = lut[color_indices]
        alpha = np.where(mask, 255, 0).astype(np.uint8)
        rgba = np.dstack((rgb, alpha))
        difference_image = Image.fromarray(
            np.flipud(rgba),
            mode="RGBA",
        ).resize(
            (plot_size, plot_size),
            resample=Image.Resampling.BILINEAR,
        )
        plot_left = left + (cell_size - plot_size) // 2
        plot_top = top + 24
        frame.paste(
            difference_image,
            (plot_left, plot_top),
            difference_image,
        )
        draw.ellipse(
            (
                plot_left,
                plot_top,
                plot_left + plot_size - 1,
                plot_top + plot_size - 1,
            ),
            outline=(245, 245, 245),
            width=2,
        )

    bar_width = 220
    bar_height = 13
    bar_left = frame_size - margin - bar_width - 55
    bar_top = frame_size - 30
    gradient = np.tile(lut[None, :, :], (bar_height, 1, 1))
    gradient_image = Image.fromarray(gradient, mode="RGB").resize(
        (bar_width, bar_height),
    )
    frame.paste(gradient_image, (bar_left, bar_top))
    draw.text(
        (margin, frame_size - 43),
        (
            "delta p = p_case - p_Case0   |   "
            + (
                f"one shared linear range L = {case_limits[0]:.3g}"
                if globally_shared
                else "detail view: separate linear range for each case"
            )
        ),
        font=small_font,
        fill=(205, 207, 214),
    )
    draw.text(
        (margin, frame_size - 24),
        (
            "Monte Carlo texture is not significance; rank weak cases with "
            "the signed-observable confidence intervals"
            if result.method == "micro"
            else "range is fixed over time; rare values beyond it saturate only in the image"
        ),
        font=small_font,
        fill=(205, 207, 214),
    )
    draw.text(
        (bar_left - 6, bar_top + bar_height // 2),
        "-L" if globally_shared else "-L_case",
        font=small_font,
        fill=(210, 210, 215),
        anchor="rm",
    )
    draw.text(
        (bar_left + bar_width // 2, bar_top - 2),
        "0",
        font=small_font,
        fill=(210, 210, 215),
        anchor="ms",
    )
    draw.text(
        (bar_left + bar_width + 5, bar_top + bar_height // 2),
        "+L" if globally_shared else "+L_case",
        font=small_font,
        fill=(210, 210, 215),
        anchor="lm",
    )
    return np.asarray(frame, dtype=np.uint8).tobytes()


def render_position_pdf_difference_video(
    result: PositionPDFResult,
    config: ProjectConfig,
    output_path: str | Path,
    *,
    case_limits: np.ndarray | None = None,
) -> Path:
    """Render signed differences with one fixed linear range per case."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    density_indices = _density_frame_indices(result, config)

    difference_density = (
        result.density
        if result.difference_density is None
        else result.difference_density
    )
    differences = difference_density - difference_density[0:1]
    automatic_limits, case_maxima = _case_specific_linear_limits(
        differences,
        result.coordinates,
        config.density.difference_case_percentile,
    )
    synchronized = case_limits is not None
    if case_limits is None:
        case_limits = automatic_limits
    else:
        case_limits = np.asarray(case_limits, dtype=float)
        if case_limits.shape != (len(result.case_numbers),):
            raise ValueError("case_limits must contain one value per density case")
        if not np.all(np.isfinite(case_limits)) or np.any(case_limits < 0.0):
            raise ValueError("case_limits must be finite and non-negative")
    globally_shared = bool(
        len(case_limits) < 2
        or np.allclose(
            case_limits[1:],
            case_limits[1],
            rtol=1.0e-12,
            atol=1.0e-15,
        )
    )
    limit_text = ", ".join(
        f"C{case_number}=+/-{limit:.4g}"
        for case_number, limit in zip(
            result.case_numbers,
            case_limits,
            strict=True,
        )
    )
    maximum_text = ", ".join(
        f"C{case_number}={maximum:.4g}"
        for case_number, maximum in zip(
            result.case_numbers,
            case_maxima,
            strict=True,
        )
    )
    print(
        "\nDifference-density display\n"
        "  mapping: conventional linear, fixed over time, "
        f"{'shared across cases' if globally_shared else 'separate by case'}\n"
        f"  limits ({'synchronized micro+macro, ' if synchronized else ''}"
        f"q{config.density.difference_case_percentile:g} of "
        f"|delta p|): {limit_text}\n"
        f"  exact max_(x,t)|delta p|: {maximum_text}",
        flush=True,
    )

    lut = _difference_color_lut()
    title_font = _font(max(18, config.density.frame_size // 48))
    case_font = _font(max(12, config.density.frame_size // 86))
    small_font = _font(max(12, config.density.frame_size // 86))
    _pipe_frames_to_ffmpeg(
        config,
        output,
        density_indices,
        "Rendering PDF-difference MP4",
        lambda density_index: _render_difference_frame(
            result,
            density_index,
            config.density.frame_size,
            lut,
            title_font,
            case_font,
            small_font,
            case_limits,
        ),
    )
    return output


def _render_density_comparison_frame(
    results: dict[str, PositionPDFResult],
    display_indices: dict[str, int],
    display_time: float,
    frame_width: int,
    frame_height: int,
    cell_size: int,
    lut: np.ndarray,
    title_font: ImageFont.FreeTypeFont,
    method_font: ImageFont.FreeTypeFont,
    case_font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont,
) -> bytes:
    """One frame of the multi-method absolute-density comparison grid: one
    row per case, one column per method, sharing the fixed ``[0, 1]``
    display range every single-method density video already uses."""

    background = (18, 20, 25)
    frame = Image.new("RGB", (frame_width, frame_height), background)
    draw = ImageDraw.Draw(frame)
    methods = list(results)
    plot_size = cell_size - 14
    row_block = _GRID_ROW_LABEL + cell_size

    draw.text(
        (frame_width // 2, 16),
        f"rho(x,y,t) — t = {display_time:.4g}",
        font=title_font,
        fill=(245, 245, 245),
        anchor="ma",
    )
    for column_index, method in enumerate(methods):
        left = _GRID_MARGIN + column_index * (cell_size + _GRID_GAP)
        draw.text(
            (left + cell_size // 2, 46),
            _method_display_name(method),
            font=method_font,
            fill=(200, 213, 235),
            anchor="ma",
        )

    for case_index, case_title in enumerate(CASE_TITLES):
        row_top = _GRID_HEADER + _GRID_MARGIN + case_index * (row_block + _GRID_GAP)
        draw.text(
            (_GRID_MARGIN, row_top),
            case_title,
            font=case_font,
            fill=(235, 235, 238),
            anchor="la",
        )
        plot_top = row_top + _GRID_ROW_LABEL
        for column_index, method in enumerate(methods):
            result = results[method]
            values = result.density[case_index, display_indices[method]]
            coordinates = result.coordinates
            x_grid, y_grid = np.meshgrid(coordinates, coordinates)
            mask = x_grid * x_grid + y_grid * y_grid <= 1.0
            normalized = np.clip(values, 0.0, 1.0)
            color_indices = np.asarray(np.round(255.0 * normalized), dtype=np.uint8)
            rgb = lut[color_indices]
            alpha = np.where(mask, 255, 0).astype(np.uint8)
            rgba = np.dstack((rgb, alpha))
            panel = Image.fromarray(np.flipud(rgba), mode="RGBA").resize(
                (plot_size, plot_size), resample=Image.Resampling.BILINEAR
            )
            left = _GRID_MARGIN + column_index * (cell_size + _GRID_GAP)
            plot_left = left + (cell_size - plot_size) // 2
            frame.paste(panel, (plot_left, plot_top), panel)
            draw.ellipse(
                (
                    plot_left,
                    plot_top,
                    plot_left + plot_size - 1,
                    plot_top + plot_size - 1,
                ),
                outline=(245, 245, 245),
                width=2,
            )

    bar_width = 220
    bar_height = 12
    bar_left = frame_width - _GRID_MARGIN - bar_width - 50
    bar_top = frame_height - 28
    bar_values = np.linspace(0.0, 1.0, 256)
    bar_indices = np.asarray(np.round(255.0 * bar_values), dtype=np.uint8)
    gradient = Image.fromarray(
        np.tile(lut[bar_indices][None, :, :], (bar_height, 1, 1)), mode="RGB"
    ).resize((bar_width, bar_height))
    frame.paste(gradient, (bar_left, bar_top))
    draw.text(
        (_GRID_MARGIN, frame_height - 24),
        r"dimensionless; fixed 0-1 range, values above 1 saturate",
        font=small_font,
        fill=(205, 207, 214),
    )
    draw.text(
        (bar_left - 6, bar_top + bar_height // 2),
        "0",
        font=small_font,
        fill=(210, 210, 215),
        anchor="rm",
    )
    draw.text(
        (bar_left + bar_width + 5, bar_top + bar_height // 2),
        "1",
        font=small_font,
        fill=(210, 210, 215),
        anchor="lm",
    )
    return np.asarray(frame, dtype=np.uint8).tobytes()


def render_density_comparison_video(
    results: dict[str, PositionPDFResult],
    config: ProjectConfig,
    output_path: str | Path,
) -> Path:
    """Absolute density, one video: 10 rows (Cases 0-9) x one column per
    method present in ``results`` (typically micro/moment/macro/analytic), so the
    same case's panels sit directly next to each other for comparison.
    Falls back to being a 10x1 column when only one method is given.
    """

    if not results:
        raise ValueError("at least one density result is required")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = list(results)
    shared_times = shared_display_times(list(results.values()), config)
    display_indices = {
        method: _density_frame_indices(result, config, frame_times=shared_times)
        for method, result in results.items()
    }

    cell_size, frame_width, frame_height = _comparison_grid_geometry(
        config.density.frame_size, n_rows=len(CASE_TITLES), n_columns=len(methods)
    )
    lut = _color_lut()
    title_font = _font(max(18, frame_width // 40))
    method_font = _font(max(14, frame_width // 60))
    case_font = _font(max(12, cell_size // 16))
    small_font = _font(max(11, frame_width // 90))

    def build_frame(frame_position: int) -> bytes:
        indices = {
            method: int(display_indices[method][frame_position]) for method in methods
        }
        return _render_density_comparison_frame(
            results,
            indices,
            float(shared_times[frame_position]),
            frame_width,
            frame_height,
            cell_size,
            lut,
            title_font,
            method_font,
            case_font,
            small_font,
        )

    _pipe_frames_to_ffmpeg(
        config,
        output,
        np.arange(len(shared_times)),
        "Rendering density comparison MP4",
        build_frame,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return output


def _render_density_difference_comparison_frame(
    results: dict[str, PositionPDFResult],
    display_indices: dict[str, int],
    display_time: float,
    frame_width: int,
    frame_height: int,
    cell_size: int,
    lut: np.ndarray,
    title_font: ImageFont.FreeTypeFont,
    method_font: ImageFont.FreeTypeFont,
    case_font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont,
    case_limits: np.ndarray,
) -> bytes:
    """One frame of the multi-method signed-difference comparison grid, one
    row per case, one column per method, sharing one linear range per case
    (``case_limits``) across every method's column."""

    background = (18, 20, 25)
    frame = Image.new("RGB", (frame_width, frame_height), background)
    draw = ImageDraw.Draw(frame)
    methods = list(results)
    plot_size = cell_size - 14
    row_block = _GRID_ROW_LABEL + cell_size

    draw.text(
        (frame_width // 2, 16),
        f"delta rho = rho_case - rho_Case0 — t = {display_time:.4g}",
        font=title_font,
        fill=(245, 245, 245),
        anchor="ma",
    )
    for column_index, method in enumerate(methods):
        left = _GRID_MARGIN + column_index * (cell_size + _GRID_GAP)
        draw.text(
            (left + cell_size // 2, 46),
            _method_display_name(method),
            font=method_font,
            fill=(200, 213, 235),
            anchor="ma",
        )

    for case_index, case_title in enumerate(CASE_TITLES):
        row_top = _GRID_HEADER + _GRID_MARGIN + case_index * (row_block + _GRID_GAP)
        limit = float(case_limits[case_index])
        row_label = (
            case_title if case_index == 0 else f"{case_title}  [range +/-{limit:.2g}]"
        )
        draw.text(
            (_GRID_MARGIN, row_top),
            row_label,
            font=case_font,
            fill=(235, 235, 238),
            anchor="la",
        )
        plot_top = row_top + _GRID_ROW_LABEL
        for column_index, method in enumerate(methods):
            result = results[method]
            difference_density = (
                result.density
                if result.difference_density is None
                else result.difference_density
            )
            time_index = display_indices[method]
            case_value = difference_density[case_index, time_index]
            baseline_value = difference_density[0, time_index]
            difference = case_value - baseline_value
            coordinates = result.coordinates
            x_grid, y_grid = np.meshgrid(coordinates, coordinates)
            mask = x_grid * x_grid + y_grid * y_grid <= 1.0
            normalized = 0.5 + 0.5 * _normalize_linear_difference(difference, limit)
            color_indices = np.asarray(np.round(255.0 * normalized), dtype=np.uint8)
            rgb = lut[color_indices]
            alpha = np.where(mask, 255, 0).astype(np.uint8)
            rgba = np.dstack((rgb, alpha))
            panel = Image.fromarray(np.flipud(rgba), mode="RGBA").resize(
                (plot_size, plot_size), resample=Image.Resampling.BILINEAR
            )
            left = _GRID_MARGIN + column_index * (cell_size + _GRID_GAP)
            plot_left = left + (cell_size - plot_size) // 2
            frame.paste(panel, (plot_left, plot_top), panel)
            draw.ellipse(
                (
                    plot_left,
                    plot_top,
                    plot_left + plot_size - 1,
                    plot_top + plot_size - 1,
                ),
                outline=(245, 245, 245),
                width=2,
            )

    bar_width = 220
    bar_height = 12
    bar_left = frame_width - _GRID_MARGIN - bar_width - 50
    bar_top = frame_height - 28
    gradient = Image.fromarray(
        np.tile(lut[None, :, :], (bar_height, 1, 1)), mode="RGB"
    ).resize((bar_width, bar_height))
    frame.paste(gradient, (bar_left, bar_top))
    draw.text(
        (_GRID_MARGIN, frame_height - 24),
        "each case has its own linear range, shared across every method's column",
        font=small_font,
        fill=(205, 207, 214),
    )
    draw.text(
        (bar_left - 6, bar_top + bar_height // 2),
        "-",
        font=small_font,
        fill=(210, 210, 215),
        anchor="rm",
    )
    draw.text(
        (bar_left + bar_width // 2, bar_top - 2),
        "0",
        font=small_font,
        fill=(210, 210, 215),
        anchor="ms",
    )
    draw.text(
        (bar_left + bar_width + 5, bar_top + bar_height // 2),
        "+",
        font=small_font,
        fill=(210, 210, 215),
        anchor="lm",
    )
    return np.asarray(frame, dtype=np.uint8).tobytes()


def render_density_difference_comparison_video(
    results: dict[str, PositionPDFResult],
    config: ProjectConfig,
    output_path: str | Path,
    *,
    case_limits: np.ndarray | None = None,
) -> Path:
    """Signed Case-minus-Case-0 density, one video: 10 rows (cases) x one
    column per method, with one shared linear range per case across every
    method's column so the same case is directly comparable at a glance.
    """

    if not results:
        raise ValueError("at least one density result is required")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = list(results)
    shared_times = shared_display_times(list(results.values()), config)
    display_indices = {
        method: _density_frame_indices(result, config, frame_times=shared_times)
        for method, result in results.items()
    }

    if case_limits is None:
        case_limits = synchronized_case_linear_limits(
            list(results.values()), config.density.difference_case_percentile
        )
    else:
        case_limits = np.asarray(case_limits, dtype=float)
        if case_limits.shape != (len(CASE_TITLES),):
            raise ValueError("case_limits must contain one value per density case")

    cell_size, frame_width, frame_height = _comparison_grid_geometry(
        config.density.frame_size, n_rows=len(CASE_TITLES), n_columns=len(methods)
    )
    lut = _difference_color_lut()
    title_font = _font(max(18, frame_width // 40))
    method_font = _font(max(14, frame_width // 60))
    case_font = _font(max(12, cell_size // 16))
    small_font = _font(max(11, frame_width // 90))

    def build_frame(frame_position: int) -> bytes:
        indices = {
            method: int(display_indices[method][frame_position]) for method in methods
        }
        return _render_density_difference_comparison_frame(
            results,
            indices,
            float(shared_times[frame_position]),
            frame_width,
            frame_height,
            cell_size,
            lut,
            title_font,
            method_font,
            case_font,
            small_font,
            case_limits,
        )

    _pipe_frames_to_ffmpeg(
        config,
        output,
        np.arange(len(shared_times)),
        "Rendering density-difference comparison MP4",
        build_frame,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return output
