"""Command-line interface kept intentionally small and discoverable."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .diagnostics import (
    diagnostic_report,
    theory_regime_report,
    timestep_report,
)
from .fields import create_field
from .numerics import resolve_numerics
from .simulator import simulate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float, help="override simulation duration")
    parser.add_argument("--dt", type=float, help="override integration time step")
    parser.add_argument("--seed", type=int, help="override random seed")
    parser.add_argument(
        "--fps", type=int, help="override display/video frames per second"
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "numba", "python"),
        help="override the configured simulation engine",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Numba native threads; 0 means all available cores",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable terminal progress bars",
    )


def _load(args: argparse.Namespace):
    config = load_config(args.config)
    return config.with_overrides(
        duration=args.duration,
        dt=args.dt,
        seed=args.seed,
        fps=args.fps,
        engine=args.engine,
        threads=args.threads,
        progress=False if args.no_progress else None,
    )


def _density_strategy_arguments(parser: argparse.ArgumentParser) -> None:
    strategy = parser.add_mutually_exclusive_group()
    strategy.add_argument(
        "--micro",
        dest="strategy",
        action="store_const",
        const="micro",
        help="run only the microscopic particle/KDE strategy",
    )
    strategy.add_argument(
        "--macro",
        dest="strategy",
        action="store_const",
        const="macro",
        help="run only the closed density equation from Theory.pdf",
    )
    parser.set_defaults(strategy="both")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chem-swimmer",
        description="Simulate equation (6) for one active swimmer in a disk.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser("live", help="play one case in a real-time window")
    _common_arguments(live)
    live.add_argument("--case", type=int, choices=range(9), default=8)
    live.add_argument(
        "--speed", type=float, default=1.0, help="playback speed multiplier"
    )

    video = subparsers.add_parser("video", help="render one case to MP4")
    _common_arguments(video)
    video.add_argument("--case", type=int, choices=range(9), default=8)
    video.add_argument("--output", type=Path)

    compare = subparsers.add_parser(
        "compare",
        help="render the evolving spatial PDF for Cases 0-8",
    )
    _common_arguments(compare)
    _density_strategy_arguments(compare)
    compare.add_argument("--output", type=Path)
    compare.add_argument(
        "--ensemble",
        type=int,
        help="override density.ensemble_size (must meet the occupancy floor)",
    )
    compare.add_argument("--data-output", type=Path, help="density NPZ output path")
    compare.add_argument(
        "--difference-output",
        type=Path,
        help="signed Case-minus-Case-0 PDF MP4 output path",
    )
    compare.add_argument("--field-output", type=Path, help="chemical-field PNG path")

    grid = subparsers.add_parser(
        "grid",
        help="alias for the Cases 0-8 evolving spatial-PDF grid",
    )
    _common_arguments(grid)
    _density_strategy_arguments(grid)
    grid.add_argument("--output", type=Path)
    grid.add_argument(
        "--ensemble",
        type=int,
        help="override density.ensemble_size (must meet the occupancy floor)",
    )
    grid.add_argument("--data-output", type=Path, help="density NPZ output path")
    grid.add_argument(
        "--difference-output",
        type=Path,
        help="signed Case-minus-Case-0 PDF MP4 output path",
    )
    grid.add_argument("--field-output", type=Path, help="chemical-field PNG path")

    check = subparsers.add_parser(
        "check", help="scan parameters and print theory coefficients"
    )
    _common_arguments(check)
    check.add_argument(
        "--case",
        default="all",
        help="0-8 or 'all' (default: all canonical cases)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load(args)
    field = create_field(config.field)

    if args.command == "check":
        cases = list(range(9)) if args.case == "all" else [int(args.case)]
        config, plan = resolve_numerics(config, field, cases)
        print(diagnostic_report(config, field, cases))
        return

    cases = (args.case,) if args.command in {"live", "video"} else tuple(range(9))
    strategy = getattr(args, "strategy", "micro")
    input_config = config
    resolution_input = config
    if strategy == "macro":
        resolution_input = replace(
            config,
            simulation=replace(config.simulation, dt="auto"),
        )
    resolved_config, plan = resolve_numerics(resolution_input, field, cases)
    if plan is not None:
        if strategy == "macro":
            resolution_text = (
                f"Theory-validity scan: {plan.scan_grid}x{plan.scan_grid}, "
                f"narrowest field length={plan.characteristic_length_star:.4g}\n"
                f"\n{theory_regime_report(resolved_config, plan, list(cases))}"
            )
        else:
            resolution_text = (
                f"Resolution: dimensionless dt={plan.dt:.4g} "
                f"(limited by {plan.limiting_constraint}), "
                f"disk scan={plan.scan_grid}x{plan.scan_grid}, "
                f"field grid={plan.field_grid}x{plan.field_grid}, "
                f"Delta(x)={plan.grid_spacing_star:.4g}\n"
                f"\n{theory_regime_report(resolved_config, plan, list(cases))}"
            )
            resolution_text += f"\n\n{timestep_report(resolved_config, plan)}"
        print(resolution_text, flush=True)

    if args.command == "live":
        from .visualization import play_live

        trajectory = simulate(
            resolved_config,
            args.case,
            field=field,
            show_progress=resolved_config.performance.progress,
        )
        play_live(trajectory, field, resolved_config, playback_speed=args.speed)
        return

    output_directory = PROJECT_ROOT / "output" / "videos"
    if args.command == "video":
        from .visualization import render_single_video

        trajectory = simulate(
            resolved_config,
            args.case,
            field=field,
            show_progress=resolved_config.performance.progress,
        )
        output = args.output or output_directory / f"case_{args.case:02d}.mp4"
        result = render_single_video(trajectory, field, resolved_config, output)
        size_mb = os.path.getsize(result) / (1024.0 * 1024.0)
        print(f"Wrote {result} ({size_mb:.2f} MiB)")
        return

    from .density import compute_position_pdf, save_position_pdf
    from .macroscopic import (
        build_macroscopic_plan,
        compute_macroscopic_pdf,
        macroscopic_timestep_report,
    )
    from .visualization import (
        render_chemical_field,
        render_density_inference_report,
        render_microscopic_mechanism_report,
        render_position_pdf_difference_video,
        render_position_pdf_video,
        synchronized_case_linear_limits,
        synchronized_global_linear_limits,
    )

    if strategy == "both" and any(
        path is not None
        for path in (args.output, args.data_output, args.difference_output)
    ):
        raise ValueError(
            "custom density output paths are ambiguous when both methods run; "
            "add --micro or --macro, or omit the custom paths"
        )
    if strategy == "macro" and args.ensemble is not None:
        raise ValueError(
            "--ensemble applies only to --micro; the closed --macro "
            "equation has no particle sample count M"
        )

    field_output = (
        args.field_output or PROJECT_ROOT / "output" / "figures" / "chemical_field.png"
    )
    field_result = render_chemical_field(
        field,
        resolved_config,
        field_output,
    )
    print(f"Wrote {field_result}")

    results: dict[str, tuple[object, object]] = {}
    if strategy in {"micro", "both"}:
        micro_pdf = compute_position_pdf(
            resolved_config,
            ensemble_size=args.ensemble,
            show_progress=resolved_config.performance.progress,
        )
        results["micro"] = (micro_pdf, resolved_config)

    if strategy in {"macro", "both"}:
        macro_plan, *_ = build_macroscopic_plan(input_config, field)
        print(macroscopic_timestep_report(macro_plan, input_config), flush=True)
        macro_pdf = compute_macroscopic_pdf(
            input_config,
            field=field,
            show_progress=input_config.performance.progress,
        )
        results["macro"] = (macro_pdf, input_config)

    density_results = [item[0] for item in results.values()]
    detail_limits = synchronized_case_linear_limits(
        density_results,
        input_config.density.difference_case_percentile,
    )
    magnitude_limits = synchronized_global_linear_limits(
        density_results,
        input_config.density.difference_case_percentile,
    )
    if strategy == "both":
        inference_outputs = render_density_inference_report(
            results["micro"][0],
            results["macro"][0],
            field,
            input_config,
            PROJECT_ROOT / "output" / "figures" / "density_inference_observables.png",
            PROJECT_ROOT / "output" / "data" / "density_observables_summary.csv",
            PROJECT_ROOT / "output" / "data" / "density_observables_timeseries.csv",
        )
        for inference_output in inference_outputs:
            print(f"Wrote {inference_output}")
    if "micro" in results:
        mechanism_outputs = render_microscopic_mechanism_report(
            results["micro"][0],
            input_config,
            PROJECT_ROOT
            / "output"
            / "figures"
            / "microscopic_mechanism_observables.png",
            PROJECT_ROOT
            / "output"
            / "data"
            / "microscopic_mechanism_summary.csv",
            PROJECT_ROOT
            / "output"
            / "data"
            / "microscopic_mechanism_observables.csv",
        )
        for mechanism_output in mechanism_outputs:
            print(f"Wrote {mechanism_output}")

    names = {
        "micro": (
            "pdf_cases_00_08.npz",
            "probability_density_cases_00_08.mp4",
            "probability_density_difference_detail_cases_00_08.mp4",
        ),
        "macro": (
            "pdf_macro_cases_00_08.npz",
            "probability_density_macro_cases_00_08.mp4",
            "probability_density_difference_detail_macro_cases_00_08.mp4",
        ),
    }
    for method, (pdf, method_config) in results.items():
        data_name, video_name, detail_name = names[method]
        data_output = (
            args.data_output
            if strategy != "both" and args.data_output is not None
            else PROJECT_ROOT / "output" / "data" / data_name
        )
        video_output = (
            args.output
            if strategy != "both" and args.output is not None
            else output_directory / video_name
        )
        detail_output = (
            args.difference_output
            if strategy != "both" and args.difference_output is not None
            else output_directory / detail_name
        )

        data_result = save_position_pdf(pdf, data_output)
        video_result = render_position_pdf_video(
            pdf,
            method_config,
            video_output,
        )
        detail_result = render_position_pdf_difference_video(
            pdf,
            method_config,
            detail_output,
            case_limits=detail_limits,
        )
        print(f"Wrote {detail_result}")
        if method == "macro":
            print(
                "Density used the closed PDE "
                f"on a {len(pdf.coordinates)}x{len(pdf.coordinates)} grid, "
                f"absolute KDE={pdf.bandwidth_fraction:.4g}, "
                f"difference KDE={pdf.difference_bandwidth_fraction:.4g}"
            )
        else:
            print(
                f"Density used M={pdf.ensemble_size}, "
                f"grid={len(pdf.coordinates)}x{len(pdf.coordinates)}, "
                f"{method_config.density.initial_position_distribution} initial "
                "positions, "
                f"{method_config.density.initial_heading_distribution} initial "
                "headings, "
                f"absolute bandwidth={pdf.bandwidth_fraction:.4g}, "
                f"difference bandwidth={pdf.difference_bandwidth_fraction:.4g}"
            )
        print(f"Configuration: {pdf.config_source}")
        print(f"Wrote {data_result}")
        video_size_mb = os.path.getsize(video_result) / (1024.0 * 1024.0)
        detail_size_mb = os.path.getsize(detail_result) / (1024.0 * 1024.0)
        print(f"Wrote {video_result} ({video_size_mb:.2f} MiB)")
        print(f"Wrote {detail_result} ({detail_size_mb:.2f} MiB)")
