"""Lazy visualization exports so density rendering does not import Matplotlib."""

from __future__ import annotations

from typing import Any

__all__ = [
    "play_live",
    "render_chemical_field",
    "render_comparison_video",
    "render_density_comparison_video",
    "render_density_difference_comparison_video",
    "render_density_frames_vs_time",
    "render_density_inference_report",
    "render_initial_condition_contours",
    "render_manuscript_strategy_figures",
    "render_microscopic_mechanism_report",
    "render_nematic_closure_report",
    "render_overlay_msd_video",
    "render_polarity_agreement_report",
    "render_polarity_comparison_video",
    "render_polarity_difference_video",
    "render_polarity_frames_vs_time",
    "render_polarity_report",
    "render_position_pdf_difference_video",
    "render_position_pdf_video",
    "render_single_video",
    "synchronized_case_linear_limits",
    "synchronized_global_linear_limits",
    "synchronized_polarity_case_limits",
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
    if name == "render_density_comparison_video":
        from .density_video import render_density_comparison_video

        return render_density_comparison_video
    if name == "render_density_difference_comparison_video":
        from .density_video import render_density_difference_comparison_video

        return render_density_difference_comparison_video
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
    if name == "render_initial_condition_contours":
        from .comparison_plot import render_initial_condition_contours

        return render_initial_condition_contours
    if name == "render_density_frames_vs_time":
        from .comparison_plot import render_density_frames_vs_time

        return render_density_frames_vs_time
    if name == "render_manuscript_strategy_figures":
        from .manuscript_plot import render_manuscript_strategy_figures

        return render_manuscript_strategy_figures
    if name == "render_microscopic_mechanism_report":
        from .inference_plot import render_microscopic_mechanism_report

        return render_microscopic_mechanism_report
    if name == "render_nematic_closure_report":
        from .nematic_plot import render_nematic_closure_report

        return render_nematic_closure_report
    if name == "render_polarity_report":
        from .polarity_plot import render_polarity_report

        return render_polarity_report
    if name == "render_polarity_agreement_report":
        from .polarity_plot import render_polarity_agreement_report

        return render_polarity_agreement_report
    if name == "render_polarity_difference_video":
        from .polarity_video import render_polarity_difference_video

        return render_polarity_difference_video
    if name == "render_polarity_comparison_video":
        from .polarity_video import render_polarity_comparison_video

        return render_polarity_comparison_video
    if name == "render_polarity_frames_vs_time":
        from .comparison_plot import render_polarity_frames_vs_time

        return render_polarity_frames_vs_time
    if name == "synchronized_polarity_case_limits":
        from .polarity_video import synchronized_polarity_case_limits

        return synchronized_polarity_case_limits
    if name in {
        "render_comparison_video",
        "render_overlay_msd_video",
        "render_single_video",
    }:
        from . import trajectory_video

        return getattr(trajectory_video, name)
    raise AttributeError(name)
