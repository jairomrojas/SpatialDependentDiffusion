"""Fast direct-raster MP4 rendering for the Cases 0-9 polarity-difference field.

Mirrors ``density_video.py``'s difference-detail movie, but the polarity
difference ``delta p = p_case - p_Case0`` is a 2-vector, not a scalar, so it
is drawn as a quiver plot -- one arrow per subsampled grid point, pointing
along ``delta p`` -- with each arrow colored (and its length scaled) by
``|delta p|`` relative to a fixed per-case linear range, exactly analogous to
the case-specific range in the density detail movie.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import ProjectConfig
from ..density import PositionPDFResult
from .density_video import (
    _GRID_GAP,
    _GRID_HEADER,
    _GRID_MARGIN,
    _GRID_ROW_LABEL,
    CASE_TITLES,
    _color_lut,
    _comparison_grid_geometry,
    _density_frame_indices,
    _detail_grid_geometry,
    _font,
    _method_display_name,
    _method_label,
    _pipe_frames_to_ffmpeg,
    shared_display_times,
)

_ARROWHEAD_ANGLE = 0.5
_MIN_NORMALIZED_TO_DRAW = 0.03


def _display_polarity_fields(
    result: PositionPDFResult,
) -> tuple[np.ndarray, np.ndarray]:
    """The polarity field to render: the smoother difference-bandwidth KDE when
    available (as for ``difference_density``), else the sharp field as-is (e.g.
    the noise-free theoretical macro field, which needs no extra smoothing).
    """

    if (
        result.difference_polarity_x is not None
        and result.difference_polarity_y is not None
    ):
        return result.difference_polarity_x, result.difference_polarity_y
    if result.polarity_x is not None and result.polarity_y is not None:
        return result.polarity_x, result.polarity_y
    raise ValueError(f"{result.method} result has no polarity field to render")


def _polarity_case_limits(
    polarity_x: np.ndarray,
    polarity_y: np.ndarray,
    coordinates: np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed robust display limits and exact maxima of ``|delta p|`` per case."""

    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    disk_mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    difference_x = polarity_x - polarity_x[0:1]
    difference_y = polarity_y - polarity_y[0:1]
    magnitude = np.hypot(difference_x, difference_y)[:, :, disk_mask]
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("polarity differences must be finite")
    limits = np.percentile(magnitude, percentile, axis=(1, 2))
    maxima = np.max(magnitude, axis=(1, 2))
    return np.asarray(limits, dtype=float), np.asarray(maxima, dtype=float)


def synchronized_polarity_case_limits(
    results: tuple[PositionPDFResult, ...] | list[PositionPDFResult],
    percentile: float,
) -> np.ndarray:
    """Resolve one per-case ``|delta p|`` range jointly across several methods."""

    if not results:
        raise ValueError("at least one polarity result is required")
    case_numbers = results[0].case_numbers
    samples: list[list[np.ndarray]] = [[] for _ in case_numbers]
    for result in results:
        if result.case_numbers != case_numbers:
            raise ValueError("polarity results use inconsistent case numbers")
        display_x, display_y = _display_polarity_fields(result)
        x_grid, y_grid = np.meshgrid(result.coordinates, result.coordinates)
        disk_mask = x_grid * x_grid + y_grid * y_grid <= 1.0
        difference_x = display_x - display_x[0:1]
        difference_y = display_y - display_y[0:1]
        magnitude = np.hypot(difference_x, difference_y)
        for case_index in range(len(case_numbers)):
            values = magnitude[case_index, :, disk_mask].reshape(-1)
            if not np.all(np.isfinite(values)):
                raise ValueError("polarity differences must be finite")
            samples[case_index].append(values)
    return np.asarray(
        [
            np.percentile(np.concatenate(case_samples), percentile)
            for case_samples in samples
        ],
        dtype=float,
    )


def _quiver_sample_points(coordinates: np.ndarray, target_count: int) -> np.ndarray:
    """Indices into ``coordinates`` for an evenly spaced quiver subsample."""

    grid_size = len(coordinates)
    stride = max(1, grid_size // target_count)
    return np.arange(grid_size // 2 % stride, grid_size, stride)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    origin: tuple[float, float],
    delta: tuple[float, float],
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Draw one arrow shaft plus a small open arrowhead, in pixel coordinates."""

    x0, y0 = origin
    dx, dy = delta
    x1, y1 = x0 + dx, y0 + dy
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    length = float(np.hypot(dx, dy))
    if length < 1.0e-9:
        return
    direction = np.arctan2(dy, dx)
    head_length = min(0.4 * length, 7.0)
    for sign in (-1.0, 1.0):
        wing_angle = direction + np.pi - sign * _ARROWHEAD_ANGLE
        wing_x = x1 + head_length * np.cos(wing_angle)
        wing_y = y1 + head_length * np.sin(wing_angle)
        draw.line((x1, y1, wing_x, wing_y), fill=color, width=width)


def _render_polarity_frame(
    result: PositionPDFResult,
    time_index: int,
    frame_size: int,
    lut: np.ndarray,
    sample_indices: np.ndarray,
    title_font: ImageFont.FreeTypeFont,
    case_font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont,
    case_limits: np.ndarray,
) -> bytes:
    background = (18, 20, 25)
    disk_fill = (32, 34, 40)
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

    display_x, display_y = _display_polarity_fields(result)
    polarity_x = display_x[:, time_index].astype(float)
    polarity_y = display_y[:, time_index].astype(float)
    difference_x = polarity_x - polarity_x[0]
    difference_y = polarity_y - polarity_y[0]
    coordinates = result.coordinates
    x_at_sample = coordinates[sample_indices]
    inside = x_at_sample[None, :] ** 2 + x_at_sample[:, None] ** 2 <= 1.0

    draw.text(
        (frame_size // 2, 18),
        (
            "Polarity difference from Case 0 — "
            f"t = {result.times[time_index]:.4g}, "
            f"{_method_label(result, difference=True)}"
        ),
        font=title_font,
        fill=(245, 245, 245),
        anchor="ma",
    )

    sample_spacing = plot_size / max(1, len(sample_indices))
    max_arrow_length = 0.85 * sample_spacing
    min_arrow_length = 0.22 * sample_spacing

    for case_index, title in enumerate(CASE_TITLES):
        row, column = divmod(case_index, columns)
        left = grid_left + column * (cell_size + gap)
        top = grid_top + row * (cell_size + gap)
        range_text = f"  [|delta p| <= {case_limits[case_index]:.2g}]"
        draw.text(
            (left + cell_size // 2, top),
            f"{title}{range_text}",
            font=case_font,
            fill=(235, 235, 238),
            anchor="ma",
        )

        plot_left = left + (cell_size - plot_size) // 2
        plot_top = top + 24
        draw.ellipse(
            (
                plot_left,
                plot_top,
                plot_left + plot_size - 1,
                plot_top + plot_size - 1,
            ),
            fill=disk_fill,
            outline=(245, 245, 245),
            width=2,
        )

        limit = float(case_limits[case_index])
        dx_grid = difference_x[case_index][np.ix_(sample_indices, sample_indices)]
        dy_grid = difference_y[case_index][np.ix_(sample_indices, sample_indices)]
        magnitude = np.hypot(dx_grid, dy_grid)
        normalized = (
            np.clip(magnitude / limit, 0.0, 1.0)
            if limit > 0.0
            else np.zeros_like(magnitude)
        )
        color_indices = np.asarray(np.round(255.0 * normalized), dtype=np.uint8)

        for row_index in range(len(sample_indices)):
            for column_index in range(len(sample_indices)):
                if not inside[row_index, column_index]:
                    continue
                value = float(normalized[row_index, column_index])
                if value < _MIN_NORMALIZED_TO_DRAW:
                    continue
                x_value = float(coordinates[sample_indices[column_index]])
                y_value = float(coordinates[sample_indices[row_index]])
                origin_x = plot_left + plot_size * (x_value + 1.0) / 2.0
                origin_y = plot_top + plot_size * (1.0 - y_value) / 2.0
                arrow_length = (
                    min_arrow_length + (max_arrow_length - min_arrow_length) * value
                )
                angle = float(
                    np.arctan2(
                        -dy_grid[row_index, column_index],
                        dx_grid[row_index, column_index],
                    )
                )
                delta = (arrow_length * np.cos(angle), arrow_length * np.sin(angle))
                color = tuple(
                    int(channel)
                    for channel in lut[color_indices[row_index, column_index]]
                )
                _draw_arrow(draw, (origin_x, origin_y), delta, color, width=2)

    bar_width = 220
    bar_height = 13
    bar_left = frame_size - margin - bar_width - 55
    bar_top = frame_size - 30
    gradient = np.tile(lut[None, :, :], (bar_height, 1, 1))
    gradient_image = Image.fromarray(gradient, mode="RGB").resize(
        (bar_width, bar_height)
    )
    frame.paste(gradient_image, (bar_left, bar_top))
    draw.text(
        (bar_left - 6, bar_top + bar_height // 2),
        "0",
        font=small_font,
        fill=(210, 210, 215),
        anchor="rm",
    )
    draw.text(
        (bar_left + bar_width + 5, bar_top + bar_height // 2),
        "L_case",
        font=small_font,
        fill=(210, 210, 215),
        anchor="lm",
    )
    draw.text(
        (margin, frame_size - 43),
        "delta p = p_case - p_Case0   |   arrow direction and color = delta p; length and color both scale with |delta p| / L_case",
        font=small_font,
        fill=(205, 207, 214),
    )
    draw.text(
        (margin, frame_size - 24),
        (
            "Monte Carlo texture is not significance; see polarity_boundary_layer_summary.csv"
            if result.method == "micro"
            else "computed from equation (15), the adiabatic slaving relation, not measured"
        ),
        font=small_font,
        fill=(205, 207, 214),
    )
    return np.asarray(frame, dtype=np.uint8).tobytes()


def render_polarity_difference_video(
    result: PositionPDFResult,
    config: ProjectConfig,
    output_path: str | Path,
    *,
    case_limits: np.ndarray | None = None,
) -> Path:
    """Render the polarity-difference field as a quiver plot colored by magnitude.

    Prefers ``result.difference_polarity_x``/``difference_polarity_y`` (the
    smoother difference-bandwidth KDE, mirroring ``difference_density``) when
    present, falling back to ``polarity_x``/``polarity_y``. One of the two
    pairs must be populated: directly measured for a microscopic result, or
    attached via :func:`compute_theoretical_polarity_field` for the
    closed-equation (macro) result, which has no polarity field of its own.
    """

    display_x, display_y = _display_polarity_fields(result)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = _density_frame_indices(result, config)

    automatic_limits, case_maxima = _polarity_case_limits(
        display_x,
        display_y,
        result.coordinates,
        config.density.difference_case_percentile,
    )
    synchronized = case_limits is not None
    if case_limits is None:
        case_limits = automatic_limits
    else:
        case_limits = np.asarray(case_limits, dtype=float)
        if case_limits.shape != (len(result.case_numbers),):
            raise ValueError("case_limits must contain one value per polarity case")
        if not np.all(np.isfinite(case_limits)) or np.any(case_limits < 0.0):
            raise ValueError("case_limits must be finite and non-negative")

    limit_text = ", ".join(
        f"C{case_number}<={limit:.4g}"
        for case_number, limit in zip(result.case_numbers, case_limits, strict=True)
    )
    maximum_text = ", ".join(
        f"C{case_number}={maximum:.4g}"
        for case_number, maximum in zip(result.case_numbers, case_maxima, strict=True)
    )
    print(
        "\nPolarity-difference display\n"
        "  mapping: quiver, arrow direction = delta p, "
        "color and length = |delta p| / L_case, "
        f"{'synchronized micro+macro, ' if synchronized else ''}separate range by case\n"
        f"  limits (q{config.density.difference_case_percentile:g} of |delta p|): {limit_text}\n"
        f"  exact max_(x,t)|delta p|: {maximum_text}",
        flush=True,
    )

    lut = _color_lut()
    sample_indices = _quiver_sample_points(result.coordinates, target_count=18)
    title_font = _font(max(18, config.density.frame_size // 48))
    case_font = _font(max(12, config.density.frame_size // 86))
    small_font = _font(max(11, config.density.frame_size // 100))
    _pipe_frames_to_ffmpeg(
        config,
        output,
        frame_indices,
        "Rendering polarity-difference MP4",
        lambda time_index: _render_polarity_frame(
            result,
            time_index,
            config.density.frame_size,
            lut,
            sample_indices,
            title_font,
            case_font,
            small_font,
            case_limits,
        ),
    )
    return output


def _render_polarity_comparison_frame(
    results: dict[str, PositionPDFResult],
    display_indices: dict[str, int],
    display_time: float,
    frame_width: int,
    frame_height: int,
    cell_size: int,
    lut: np.ndarray,
    sample_indices: np.ndarray,
    title_font: ImageFont.FreeTypeFont,
    method_font: ImageFont.FreeTypeFont,
    case_font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont,
    case_limits: np.ndarray,
) -> bytes:
    """One frame of the multi-method polarity-difference comparison grid:
    one row per case, one column per method, sharing one ``|delta p|``
    range per case (``case_limits``) across every method's column."""

    background = (18, 20, 25)
    disk_fill = (32, 34, 40)
    frame = Image.new("RGB", (frame_width, frame_height), background)
    draw = ImageDraw.Draw(frame)
    methods = list(results)
    plot_size = cell_size - 14
    row_block = _GRID_ROW_LABEL + cell_size
    sample_spacing = plot_size / max(1, len(sample_indices))
    max_arrow_length = 0.85 * sample_spacing
    min_arrow_length = 0.22 * sample_spacing

    draw.text(
        (frame_width // 2, 16),
        f"delta p = p_case - p_Case0 — t = {display_time:.4g}",
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
            case_title if case_index == 0 else f"{case_title}  [range <={limit:.2g}]"
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
            display_x, display_y = _display_polarity_fields(result)
            time_index = display_indices[method]
            polarity_x = display_x[:, time_index].astype(float)
            polarity_y = display_y[:, time_index].astype(float)
            difference_x = polarity_x[case_index] - polarity_x[0]
            difference_y = polarity_y[case_index] - polarity_y[0]
            coordinates = result.coordinates
            x_at_sample = coordinates[sample_indices]
            inside = x_at_sample[None, :] ** 2 + x_at_sample[:, None] ** 2 <= 1.0

            left = _GRID_MARGIN + column_index * (cell_size + _GRID_GAP)
            plot_left = left + (cell_size - plot_size) // 2
            draw.ellipse(
                (
                    plot_left,
                    plot_top,
                    plot_left + plot_size - 1,
                    plot_top + plot_size - 1,
                ),
                fill=disk_fill,
                outline=(245, 245, 245),
                width=2,
            )

            dx_grid = difference_x[np.ix_(sample_indices, sample_indices)]
            dy_grid = difference_y[np.ix_(sample_indices, sample_indices)]
            magnitude = np.hypot(dx_grid, dy_grid)
            normalized = (
                np.clip(magnitude / limit, 0.0, 1.0)
                if limit > 0.0
                else np.zeros_like(magnitude)
            )
            color_indices = np.asarray(np.round(255.0 * normalized), dtype=np.uint8)

            for row_index in range(len(sample_indices)):
                for sample_column in range(len(sample_indices)):
                    if not inside[row_index, sample_column]:
                        continue
                    value = float(normalized[row_index, sample_column])
                    if value < _MIN_NORMALIZED_TO_DRAW:
                        continue
                    x_value = float(coordinates[sample_indices[sample_column]])
                    y_value = float(coordinates[sample_indices[row_index]])
                    origin_x = plot_left + plot_size * (x_value + 1.0) / 2.0
                    origin_y = plot_top + plot_size * (1.0 - y_value) / 2.0
                    arrow_length = (
                        min_arrow_length + (max_arrow_length - min_arrow_length) * value
                    )
                    angle = float(
                        np.arctan2(
                            -dy_grid[row_index, sample_column],
                            dx_grid[row_index, sample_column],
                        )
                    )
                    delta = (arrow_length * np.cos(angle), arrow_length * np.sin(angle))
                    color = tuple(
                        int(channel)
                        for channel in lut[color_indices[row_index, sample_column]]
                    )
                    _draw_arrow(draw, (origin_x, origin_y), delta, color, width=2)

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
        "arrow direction and color = delta p; length and color scale with |delta p| / range, per case, shared across methods",
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
        "range",
        font=small_font,
        fill=(210, 210, 215),
        anchor="lm",
    )
    return np.asarray(frame, dtype=np.uint8).tobytes()


def render_polarity_comparison_video(
    results: dict[str, PositionPDFResult],
    config: ProjectConfig,
    output_path: str | Path,
    *,
    case_limits: np.ndarray | None = None,
) -> Path:
    """Polarity difference from Case 0, one video: 10 rows (cases) x one
    column per method present in ``results``, with one shared ``|delta p|``
    range per case across every method's column so the same case is
    directly comparable at a glance.
    """

    if not results:
        raise ValueError("at least one polarity result is required")
    for name, result in results.items():
        if result.polarity_x is None:
            raise ValueError(f"{name} result has no polarity field to render")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    methods = list(results)
    shared_times = shared_display_times(list(results.values()), config)
    display_indices = {
        method: _density_frame_indices(result, config, frame_times=shared_times)
        for method, result in results.items()
    }

    if case_limits is None:
        case_limits = synchronized_polarity_case_limits(
            list(results.values()), config.density.difference_case_percentile
        )
    else:
        case_limits = np.asarray(case_limits, dtype=float)
        if case_limits.shape != (len(CASE_TITLES),):
            raise ValueError("case_limits must contain one value per polarity case")

    cell_size, frame_width, frame_height = _comparison_grid_geometry(
        config.density.frame_size, n_rows=len(CASE_TITLES), n_columns=len(methods)
    )
    lut = _color_lut()
    sample_indices = _quiver_sample_points(
        next(iter(results.values())).coordinates, target_count=14
    )
    title_font = _font(max(18, frame_width // 40))
    method_font = _font(max(14, frame_width // 60))
    case_font = _font(max(12, cell_size // 16))
    small_font = _font(max(11, frame_width // 90))

    def build_frame(frame_position: int) -> bytes:
        indices = {
            method: int(display_indices[method][frame_position]) for method in methods
        }
        return _render_polarity_comparison_frame(
            results,
            indices,
            float(shared_times[frame_position]),
            frame_width,
            frame_height,
            cell_size,
            lut,
            sample_indices,
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
        "Rendering polarity comparison MP4",
        build_frame,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return output
