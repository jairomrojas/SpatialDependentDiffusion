"""Command-line interface kept intentionally small and discoverable."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import replace
from pathlib import Path

from .cases import STRATEGY_FIGURE_STEMS
from .config import load_config
from .diagnostics import (
    diagnostic_report,
    theory_regime_report,
    timestep_report,
)
from .fields import create_field
from .numerics import resolve_numerics
from .simulator import simulate

# This file sits in model/solvers/chem_swimmer/; the repository root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "model" / "parameters" / "default.toml"
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "simulations" / "population_release"
FIGURE_ROOT = RESULTS_ROOT / "figures"
SUPPLEMENTARY_FIGURES = FIGURE_ROOT / "supplementary"
METRICS_ROOT = RESULTS_ROOT / "metrics"
RAW_DATA_ROOT = RESULTS_ROOT / "data"
VIDEO_ROOT = RESULTS_ROOT / "videos"
DATA_NAMES = {
    "micro": "pdf_cases_00_09.npz",
    "moment": "pdf_moment_cases_00_09.npz",
    "macro": "pdf_macro_cases_00_09.npz",
    "analytic": "pdf_analytic_cases_00_09.npz",
}


def _remove_if_exists(path: Path) -> None:
    """Delete a stale output file before writing its replacement, so a
    re-run never leaves an old file's content next to (or instead of) the
    new one under the same name.
    """

    path.unlink(missing_ok=True)


def _compare_output_paths() -> list[Path]:
    """Every file the ``compare``/``grid`` pipeline can ever produce, under
    every naming scheme (single-method fallback names and the combined
    multi-method comparison names) -- cleared unconditionally before each
    run so switching between e.g. ``--micro`` and the default all-methods
    strategy never leaves a stale file from the other naming scheme behind.
    """

    figures = FIGURE_ROOT
    data = RAW_DATA_ROOT
    videos = VIDEO_ROOT
    per_method_video_stems = {
        "micro": "cases_00_09",
        "moment": "moment_cases_00_09",
        "macro": "macro_cases_00_09",
        "analytic": "analytic_cases_00_09",
    }
    paths = [
        figures / "chemical_field.png",
        figures / "density_inference_observables.png",
        figures / "microscopic_mechanism_observables.png",
        figures / "nematic_closure.png",
        figures / "nematic_closure.pdf",
        figures / "polarity_boundary_layer.png",
        figures / "polarity_agreement.png",
        figures / "initial_conditions.png",
        data / "density_observables_summary.csv",
        data / "density_observables_timeseries.csv",
        data / "microscopic_mechanism_summary.csv",
        data / "microscopic_mechanism_observables.csv",
        data / "nematic_closure_summary.csv",
        data / "nematic_closure_timeseries.csv",
        data / "polarity_boundary_layer_summary.csv",
        data / "polarity_agreement_summary.csv",
        data / "frame_comparison_metrics.csv",
        data / "analytic_validity.txt",
        videos / "probability_density_comparison.mp4",
        videos / "probability_density_difference_comparison.mp4",
        videos / "polarity_difference_comparison.mp4",
    ]
    # Remove the pre-Case-9 names as well so a new run cannot leave archives
    # or single-method movies that silently contain only Cases 0-8.
    paths.extend(())
    for stem in per_method_video_stems.values():
        paths.append(videos / f"probability_density_{stem}.mp4")
        paths.append(videos / f"probability_density_difference_detail_{stem}.mp4")
        paths.append(videos / f"polarity_difference_detail_{stem}.mp4")
    for stem in (
        "cases_00_08",
        "moment_cases_00_08",
        "macro_cases_00_08",
        "analytic_cases_00_08",
    ):
        paths.append(videos / f"probability_density_{stem}.mp4")
        paths.append(videos / f"probability_density_difference_detail_{stem}.mp4")
        paths.append(videos / f"polarity_difference_detail_{stem}.mp4")
    for case_number in range(10):
        paths.append(
            figures / "frames_vs_time" / f"density_frames_case_{case_number:02d}.png"
        )
        paths.append(
            figures / "frames_vs_time" / f"polarity_frames_case_{case_number:02d}.png"
        )
    for case_number in range(1, 10):
        paths.append(
            SUPPLEMENTARY_FIGURES / f"{STRATEGY_FIGURE_STEMS[case_number]}.pdf"
        )
        paths.append(
            SUPPLEMENTARY_FIGURES / f"{STRATEGY_FIGURE_STEMS[case_number]}.png"
        )
    paths.append(SUPPLEMENTARY_FIGURES / "figure-s01-cases-2-9-summary.pdf")
    paths.append(SUPPLEMENTARY_FIGURES / "figure-s01-cases-2-9-summary.png")
    return paths


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
        help="Numba native threads; 0 selects an affinity/hybrid-aware default",
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


_STRATEGIES = ("micro", "moment", "macro", "analytic")


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
        "--moment",
        dest="strategy",
        action="store_const",
        const="moment",
        help="run only the density--polarity PDE with the polar closure Q=0",
    )
    strategy.add_argument(
        "--macro",
        dest="strategy",
        action="store_const",
        const="macro",
        help="run the manuscript macroscopic density--polarity PDE (Q=0)",
    )
    strategy.add_argument(
        "--analytic",
        dest="strategy",
        action="store_const",
        const="analytic",
        help="run the manuscript short-time dynamic-p Fourier/Green solution",
    )
    parser.set_defaults(strategy="all")


def _density_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Custom output paths, only meaningful with a single-method strategy
    (see the "ambiguous when multiple methods run" check in ``main()``).
    """

    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ensemble",
        type=int,
        help="override density.ensemble_size (must meet the occupancy floor)",
    )
    parser.add_argument("--data-output", type=Path, help="density NPZ output path")
    parser.add_argument(
        "--difference-output",
        type=Path,
        help="signed Case-minus-Case-0 PDF MP4 output path",
    )
    parser.add_argument("--field-output", type=Path, help="chemical-field PNG path")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="ignore a compatible numerical cache and recompute every method",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chem-swimmer",
        description="Simulate equation (6) for one active swimmer in a disk.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser("live", help="play one case in a real-time window")
    _common_arguments(live)
    live.add_argument("--case", type=int, choices=range(10), default=9)
    live.add_argument(
        "--speed", type=float, default=1.0, help="playback speed multiplier"
    )

    video = subparsers.add_parser("video", help="render one case to MP4")
    _common_arguments(video)
    video.add_argument("--case", type=int, choices=range(10), default=9)
    video.add_argument("--output", type=Path)

    compare = subparsers.add_parser(
        "compare",
        help="render the evolving spatial PDF for Cases 0-9",
    )
    _common_arguments(compare)
    _density_strategy_arguments(compare)
    _density_output_arguments(compare)

    # "grid" is a plain alias for "compare": same arguments, same behavior
    # (main() branches on args.command only to reuse the "which output
    # directory" check, never on strategy/rendering logic), kept as a
    # separate subcommand only because both names are already in use by
    # collaborators' scripts.
    grid = subparsers.add_parser(
        "grid",
        help="alias for the Cases 0-9 evolving spatial-PDF grid",
    )
    _common_arguments(grid)
    _density_strategy_arguments(grid)
    _density_output_arguments(grid)

    check = subparsers.add_parser(
        "check", help="scan parameters and print theory coefficients"
    )
    _common_arguments(check)
    check.add_argument(
        "--case",
        default="all",
        help="0-9 or 'all' (default: all canonical cases)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load(args)
    field = create_field(config.field)

    if args.command == "check":
        cases = list(range(10)) if args.case == "all" else [int(args.case)]
        config, plan = resolve_numerics(config, field, cases)
        print(diagnostic_report(config, field, cases))
        return

    cases = (args.case,) if args.command in {"live", "video"} else tuple(range(10))
    strategy = getattr(args, "strategy", "micro")
    input_config = config
    resolution_input = config
    if strategy in {"moment", "macro", "analytic"}:
        resolution_input = replace(
            config,
            simulation=replace(config.simulation, dt="auto"),
        )
    resolved_config, plan = resolve_numerics(resolution_input, field, cases)
    if plan is not None:
        if strategy in {"moment", "macro", "analytic"}:
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

    output_directory = RESULTS_ROOT / "videos"
    if args.command == "video":
        from .visualization import render_single_video

        trajectory = simulate(
            resolved_config,
            args.case,
            field=field,
            show_progress=resolved_config.performance.progress,
        )
        output = args.output or output_directory / f"case_{args.case:02d}.mp4"
        _remove_if_exists(output)
        result = render_single_video(trajectory, field, resolved_config, output)
        size_mb = os.path.getsize(result) / (1024.0 * 1024.0)
        print(f"Wrote {result} ({size_mb:.2f} MiB)")
        return

    from .analytic import compute_analytic_pdf, format_analytic_diagnostics
    from .density import compute_position_pdf, save_position_pdf
    from .macroscopic import (
        build_macroscopic_plan,
        compute_macroscopic_pdf,
        macroscopic_timestep_report,
    )
    from .moment import build_moment_plan, compute_moment_pdf, moment_timestep_report
    from .result_cache import load_cached_results, write_cache_manifest
    from .visualization import (
        render_chemical_field,
        render_density_comparison_video,
        render_density_difference_comparison_video,
        render_density_frames_vs_time,
        render_density_inference_report,
        render_initial_condition_contours,
        render_manuscript_strategy_figures,
        render_microscopic_mechanism_report,
        render_nematic_closure_report,
        render_polarity_agreement_report,
        render_polarity_comparison_video,
        render_polarity_difference_video,
        render_polarity_frames_vs_time,
        render_polarity_report,
        render_position_pdf_difference_video,
        render_position_pdf_video,
        synchronized_case_linear_limits,
        synchronized_polarity_case_limits,
    )

    if strategy == "all" and any(
        path is not None
        for path in (args.output, args.data_output, args.difference_output)
    ):
        raise ValueError(
            "custom density output paths are ambiguous when multiple methods run; "
            "add --micro, --moment, --macro, or --analytic, or omit the custom paths"
        )
    if strategy in {"moment", "macro", "analytic"} and args.ensemble is not None:
        raise ValueError(
            "--ensemble applies only to --micro; PDE methods have no particle "
            "sample count M"
        )

    if args.command in {"compare", "grid"}:
        for stale_path in _compare_output_paths():
            _remove_if_exists(stale_path)

    field_output = (
        args.field_output or SUPPLEMENTARY_FIGURES / "figure-s19-chemical-field.png"
    )
    field_result = render_chemical_field(
        field,
        resolved_config,
        field_output,
    )
    print(f"Wrote {field_result}")

    requested_methods = (
        ("micro", "macro", "analytic") if strategy == "all" else (strategy,)
    )
    cached = None
    if not args.force_recompute and args.data_output is None:
        cached = load_cached_results(
            RAW_DATA_ROOT,
            input_config,
            requested_methods,
            ensemble=args.ensemble,
        )
    results: dict[str, tuple[object, object]] = {}
    if cached is not None:
        results = {
            method: (
                cached[method],
                resolved_config if method == "micro" else input_config,
            )
            for method in requested_methods
        }
        print(
            "Using validated numerical cache; simulation inputs are unchanged. "
            "Only figures, metrics, and videos will be regenerated.",
            flush=True,
        )

    if cached is None and strategy in {"micro", "all"}:
        micro_pdf = compute_position_pdf(
            resolved_config,
            ensemble_size=args.ensemble,
            show_progress=resolved_config.performance.progress,
        )
        results["micro"] = (micro_pdf, resolved_config)

    if cached is None and strategy == "moment":
        moment_plan, *_ = build_moment_plan(input_config, field)
        print(moment_timestep_report(moment_plan, input_config), flush=True)
        moment_pdf = compute_moment_pdf(
            input_config,
            field=field,
            show_progress=input_config.performance.progress,
        )
        results["moment"] = (moment_pdf, input_config)

    if cached is None and strategy in {"macro", "all"}:
        macro_plan, *_ = build_macroscopic_plan(input_config, field)
        print(macroscopic_timestep_report(macro_plan, input_config), flush=True)
        macro_pdf = compute_macroscopic_pdf(
            input_config,
            field=field,
            show_progress=input_config.performance.progress,
        )
        results["macro"] = (macro_pdf, input_config)

    if cached is None and strategy in {"analytic", "all"}:
        analytic_pdf, analytic_diagnostics = compute_analytic_pdf(
            input_config,
            field=field,
        )
        analytic_report = format_analytic_diagnostics(analytic_diagnostics)
        print(analytic_report, flush=True)
        analytic_report_path = METRICS_ROOT / "fourier-solution-validity.txt"
        analytic_report_path.parent.mkdir(parents=True, exist_ok=True)
        analytic_report_path.write_text(analytic_report + "\n", encoding="utf-8")
        print(f"Wrote {analytic_report_path}")
        results["analytic"] = (analytic_pdf, input_config)

    # Persist expensive numerical state before any rendering.  A plotting
    # failure must never force the particle/PDE calculations to run again.
    archive_paths: dict[str, Path] = {}
    for method, (pdf, _) in results.items():
        data_output = (
            args.data_output
            if strategy != "all" and args.data_output is not None
            else RAW_DATA_ROOT / DATA_NAMES[method]
        )
        archive_paths[method] = (
            data_output.resolve()
            if cached is not None
            else save_position_pdf(pdf, data_output)
        )
    if strategy == "all":
        cache_manifest = write_cache_manifest(
            RAW_DATA_ROOT,
            input_config,
            archive_paths,
            ensemble=args.ensemble,
        )
        print(f"Wrote validated cache manifest {cache_manifest}")

    density_results = [item[0] for item in results.values()]
    detail_limits = synchronized_case_linear_limits(
        density_results,
        input_config.density.difference_case_percentile,
    )
    if len(results) > 1:
        initial_output = render_initial_condition_contours(
            {name: item[0] for name, item in results.items()},
            input_config,
            SUPPLEMENTARY_FIGURES / "figure-s10-initial-conditions.png",
        )
        print(f"Wrote {initial_output}")
        frame_outputs, frame_metrics = render_density_frames_vs_time(
            {name: item[0] for name, item in results.items()},
            input_config,
            FIGURE_ROOT / "frames_vs_time",
            METRICS_ROOT / "figure-3-method-comparison.csv",
        )
        for frame_output in frame_outputs:
            print(f"Wrote {frame_output}")
        polarity_frame_outputs, frame_metrics = render_polarity_frames_vs_time(
            {name: item[0] for name, item in results.items()},
            input_config,
            FIGURE_ROOT / "frames_vs_time",
            frame_metrics,
        )
        for frame_output in polarity_frame_outputs:
            print(f"Wrote {frame_output}")
        print(f"Wrote {frame_metrics}")
        strategy_outputs = render_manuscript_strategy_figures(
            {name: item[0] for name, item in results.items()},
            input_config,
            SUPPLEMENTARY_FIGURES,
        )
        for strategy_output in strategy_outputs:
            print(f"Wrote {strategy_output}")
        paper_root = PROJECT_ROOT / "manuscript"
        paper_figures = paper_root / "figures"
        if (paper_root / "main.tex").is_file():
            paper_figures.mkdir(parents=True, exist_ok=True)
            case_one_pdf = SUPPLEMENTARY_FIGURES / f"{STRATEGY_FIGURE_STEMS[1]}.pdf"
            if case_one_pdf.is_file():
                paper_figure = paper_figures / "figure-3-case-1-validation.pdf"
                shutil.copyfile(case_one_pdf, paper_figure)
                print(f"Wrote {paper_figure}")
                case_one_pdf.unlink()
                case_one_pdf.with_suffix(".png").unlink(missing_ok=True)
            remaining_pdf = SUPPLEMENTARY_FIGURES / "figure-s01-cases-2-9-summary.pdf"
            if remaining_pdf.is_file():
                paper_figure = paper_figures / "figure-remaining-cases.pdf"
                shutil.copyfile(remaining_pdf, paper_figure)
                print(f"Wrote {paper_figure}")
        inference_outputs = render_density_inference_report(
            {name: item[0] for name, item in results.items()},
            field,
            input_config,
            SUPPLEMENTARY_FIGURES / "figure-s11-density-inference-observables.png",
            METRICS_ROOT / "density-observables-summary.csv",
            RAW_DATA_ROOT / "density-observables-timeseries.csv",
        )
        for inference_output in inference_outputs:
            print(f"Wrote {inference_output}")
        polarity_method_results = {
            name: item[0]
            for name, item in results.items()
            if item[0].polarity_x is not None
        }
        if len(polarity_method_results) > 1:
            agreement_outputs = render_polarity_agreement_report(
                polarity_method_results,
                SUPPLEMENTARY_FIGURES / "figure-s14-polarity-agreement.png",
                METRICS_ROOT / "polarity-agreement-summary.csv",
            )
            for agreement_output in agreement_outputs:
                print(f"Wrote {agreement_output}")
    if "micro" in results:
        mechanism_outputs = render_microscopic_mechanism_report(
            results["micro"][0],
            input_config,
            SUPPLEMENTARY_FIGURES / "figure-s12-microscopic-mechanism-observables.png",
            METRICS_ROOT / "microscopic-mechanism-summary.csv",
            RAW_DATA_ROOT / "microscopic-mechanism-observables.csv",
        )
        for mechanism_output in mechanism_outputs:
            print(f"Wrote {mechanism_output}")
        polarity_outputs = render_polarity_report(
            results["micro"][0],
            input_config,
            field,
            SUPPLEMENTARY_FIGURES / "figure-s15-polarity-boundary-layer.png",
            METRICS_ROOT / "polarity-boundary-layer-summary.csv",
        )
        for polarity_output in polarity_outputs:
            print(f"Wrote {polarity_output}")
        nematic_outputs = render_nematic_closure_report(
            results["micro"][0],
            input_config,
            field,
            SUPPLEMENTARY_FIGURES / "figure-s13-nematic-closure.png",
            METRICS_ROOT / "nematic-closure-summary.csv",
            RAW_DATA_ROOT / "nematic-closure-timeseries.csv",
        )
        for nematic_output in nematic_outputs:
            print(f"Wrote {nematic_output}")

    for method, (pdf, method_config) in results.items():
        data_result = archive_paths[method]
        if method == "moment":
            print(
                "Density and polarity used the polar-closed (Q=0) moment PDE "
                f"on a {len(pdf.coordinates)}x{len(pdf.coordinates)} grid, "
                f"absolute KDE={pdf.bandwidth_fraction:.4g}, "
                f"difference KDE={pdf.difference_bandwidth_fraction:.4g}"
            )
        elif method == "macro":
            print(
                "Density used the auxiliary adiabatic closed PDE "
                f"on a {len(pdf.coordinates)}x{len(pdf.coordinates)} grid, "
                f"absolute KDE={pdf.bandwidth_fraction:.4g}, "
                f"difference KDE={pdf.difference_bandwidth_fraction:.4g}"
            )
        elif method == "analytic":
            if method_config.density.initial_position_distribution == "uniform":
                print(
                    "Density used the auxiliary adiabatic modal solution "
                    f"on a {len(pdf.coordinates)}x{len(pdf.coordinates)} grid "
                    "(see the validity report above); the uniform initial "
                    "condition needs no KDE, the solution is evaluated directly"
                )
            else:
                distribution = method_config.density.initial_position_distribution
                print(
                    "Density used the analytic modal solution "
                    f"on a {len(pdf.coordinates)}x{len(pdf.coordinates)} grid "
                    f"(see the validity report above); the {distribution} initial "
                    "condition is evaluated directly, then smoothed with the "
                    f"same KDE macro/micro use, absolute KDE={pdf.bandwidth_fraction:.4g}, "
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

    density_by_method = {name: item[0] for name, item in results.items()}
    polarity_by_method = {
        name: item[0]
        for name, item in results.items()
        if item[0].polarity_x is not None
    }

    def _report_video(path: Path) -> None:
        size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
        print(f"Wrote {path} ({size_mb:.2f} MiB)")

    if len(results) > 1:
        # Multiple methods: one combined video per family, 10 rows (Cases
        # 0-9) x one column per method, so the same case sits side by side
        # across methods instead of scattered across separate files.
        _report_video(
            render_density_comparison_video(
                density_by_method,
                input_config,
                output_directory / "probability_density_comparison.mp4",
            )
        )
        _report_video(
            render_density_difference_comparison_video(
                density_by_method,
                input_config,
                output_directory / "probability_density_difference_comparison.mp4",
                case_limits=detail_limits,
            )
        )
        if len(polarity_by_method) > 1:
            polarity_limits = synchronized_polarity_case_limits(
                list(polarity_by_method.values()),
                input_config.density.difference_case_percentile,
            )
            _report_video(
                render_polarity_comparison_video(
                    polarity_by_method,
                    input_config,
                    output_directory / "polarity_difference_comparison.mp4",
                    case_limits=polarity_limits,
                )
            )
    else:
        # A single method: nothing to place side by side, so keep the
        # original one-file-per-family, 3x3 case grid.
        (method,) = results
        pdf, method_config = results[method]
        video_stem = {
            "micro": "cases_00_09",
            "moment": "moment_cases_00_09",
            "macro": "macro_cases_00_09",
            "analytic": "analytic_cases_00_09",
        }[method]
        video_output = (
            args.output or output_directory / f"probability_density_{video_stem}.mp4"
        )
        detail_output = (
            args.difference_output
            or output_directory
            / f"probability_density_difference_detail_{video_stem}.mp4"
        )
        _report_video(render_position_pdf_video(pdf, method_config, video_output))
        detail_result = render_position_pdf_difference_video(
            pdf, method_config, detail_output, case_limits=detail_limits
        )
        _report_video(detail_result)
        if pdf.polarity_x is not None:
            _report_video(
                render_polarity_difference_video(
                    pdf,
                    method_config,
                    output_directory / f"polarity_difference_detail_{video_stem}.mp4",
                )
            )
