"""Lazy visualization exports so density rendering does not import Matplotlib."""

from __future__ import annotations

from typing import Any

__all__ = [
    "play_live",
    "render_chemical_field",
    "render_comparison_video",
    "render_density_inference_report",
    "render_microscopic_mechanism_report",
    "render_overlay_msd_video",
    "render_position_pdf_difference_video",
    "render_position_pdf_video",
    "render_single_video",
    "synchronized_case_linear_limits",
    "synchronized_global_linear_limits",
]


def __getattr__(name: str) -> Any:
    if name == "play_live":
        from .live import play_live

        return play_live
    if name == "render_position_pdf_video":
        from .density_video import render_position_pdf_video

        return render_position_pdf_video
    if name == "render_position_pdf_difference_video":
        from .density_video import render_position_pdf_difference_video

        return render_position_pdf_difference_video
    if name == "synchronized_case_linear_limits":
        from .density_video import synchronized_case_linear_limits

        return synchronized_case_linear_limits
    if name == "synchronized_global_linear_limits":
        from .density_video import synchronized_global_linear_limits

        return synchronized_global_linear_limits
    if name == "render_chemical_field":
        from .field_plot import render_chemical_field

        return render_chemical_field
    if name == "render_density_inference_report":
        from .inference_plot import render_density_inference_report

        return render_density_inference_report
    if name == "render_microscopic_mechanism_report":
        from .inference_plot import render_microscopic_mechanism_report

        return render_microscopic_mechanism_report
    if name in {
        "render_comparison_video",
        "render_overlay_msd_video",
        "render_single_video",
    }:
        from . import video

        return getattr(video, name)
    raise AttributeError(name)
