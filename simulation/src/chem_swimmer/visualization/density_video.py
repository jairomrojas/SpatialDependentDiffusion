"""Fast direct-raster MP4 rendering for the Cases 0-8 spatial PDFs."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import ProjectConfig
from ..density import PositionPDFResult
from ..progress import progress_bar
from .encoding import choose_codec

CASE_TITLES = (
    "Case 0: background coefficients only",
    "Case 1: D_T response to c",
    "Case 2: D_R response to c",
    "Case 3: v response to c",
    "Case 4: lambda response to c",
    "Case 5: D_T response to S",
    "Case 6: D_R response to S",
    "Case 7: v response to S",
    "Case 8: lambda response to S",
)


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
    if result.method == "macro":
        bandwidth = (
            result.bandwidth_fraction
            if not difference or result.difference_bandwidth_fraction is None
            else result.difference_bandwidth_fraction
        )
        symbol = "b_delta" if difference else "b"
        return f"closed PDE, N = {len(result.coordinates)}, {symbol} = {bandwidth:.3g}"
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
) -> list[str]:
    codec = choose_codec(config.video.codec)
    size = config.density.frame_size
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
        f"{size}x{size}",
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
) -> np.ndarray:
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


def _pipe_frames_to_ffmpeg(
    config: ProjectConfig,
    output: Path,
    density_indices: np.ndarray,
    description: str,
    frame_builder: Callable[[int], bytes],
) -> None:
    process = subprocess.Popen(
        _ffmpeg_command(config, output),
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
    available = frame_size - header - footer - 2 * margin
    cell_size = (available - 2 * gap) // 3
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
        row, column = divmod(case_index, 3)
        left = margin + column * (cell_size + gap)
        top = header + margin + row * (cell_size + gap)
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
    available = frame_size - header - footer - 2 * margin
    cell_size = (available - 2 * gap) // 3
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
        row, column = divmod(case_index, 3)
        left = margin + column * (cell_size + gap)
        top = header + margin + row * (cell_size + gap)
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
