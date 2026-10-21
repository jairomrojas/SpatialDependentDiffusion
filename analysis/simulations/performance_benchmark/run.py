#!/usr/bin/env python3
"""Reproducible, bounded benchmark for the microscopic PDF pipeline.

The default workload is intentionally much smaller than the manuscript run.
It exercises trajectory generation, histogram/moment reduction, KDE smoothing,
and finalization without writing large results.  Use repeated values for
``--threads`` or ``--batch-size`` to compare settings on a new machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from model.solvers.chem_swimmer.config import load_config
from model.solvers.chem_swimmer.density import compute_position_pdf


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "model" / "parameters" / "default.toml",
    )
    parser.add_argument("--ensemble", type=int, default=4096)
    parser.add_argument("--duration", type=float, default=5.0e-4)
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--time-slices", type=int, default=51)
    parser.add_argument("--threads", type=int, nargs="+", default=[0])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ensemble < 256:
        parser.error("--ensemble must be at least 256")
    if args.grid_size < 21 or args.grid_size % 2 == 0:
        parser.error("--grid-size must be an odd integer of at least 21")
    if args.time_slices < 2 or args.repeats < 1:
        parser.error("--time-slices must be at least 2 and --repeats positive")
    if any(value < 0 for value in args.threads):
        parser.error("--threads values must be non-negative")
    if any(value < 1 for value in args.batch_size):
        parser.error("--batch-size values must be positive")
    return args


def _package_versions() -> dict[str, str]:
    resolved: dict[str, str] = {}
    for package in ("numpy", "scipy", "numba"):
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "unavailable"
    return resolved


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> int:
    args = _arguments()
    base = load_config(args.config)
    if isinstance(base.simulation.dt, str):
        dt = "auto"
    else:
        dt = min(base.simulation.dt, args.duration)
    base = replace(
        base,
        simulation=replace(base.simulation, duration=args.duration, dt=dt),
        density=replace(
            base.density,
            ensemble_size=args.ensemble,
            grid_size=args.grid_size,
            min_samples_per_kernel=5,
            time_slices=args.time_slices,
        ),
        performance=replace(base.performance, progress=False),
    )

    # Compile every Numba kernel before collecting timings.
    warm = replace(
        base,
        density=replace(
            base.density,
            ensemble_size=256,
            batch_size=min(256, args.ensemble),
            grid_size=33,
            time_slices=3,
        ),
    )
    compute_position_pdf(warm, show_progress=False)

    measurements: list[dict[str, object]] = []
    for threads in args.threads:
        for batch_size in args.batch_size:
            run_config = replace(
                base,
                density=replace(base.density, batch_size=batch_size),
                performance=replace(base.performance, threads=threads),
            )
            elapsed = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = compute_position_pdf(run_config, show_progress=False)
                elapsed.append(time.perf_counter() - started)
                del result
            row = {
                "threads": threads,
                "batch_size": batch_size,
                "seconds": elapsed,
                "median_seconds": statistics.median(elapsed),
            }
            measurements.append(row)
            print(
                f"threads={threads or 'auto':>4} batch={batch_size:>5} "
                f"median={row['median_seconds']:.3f}s runs="
                + ", ".join(f"{value:.3f}" for value in elapsed),
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "packages": _package_versions(),
        },
        "workload": {
            "config": str(args.config.resolve()),
            "ensemble": args.ensemble,
            "duration": args.duration,
            "grid_size": args.grid_size,
            "time_slices": args.time_slices,
            "repeats": args.repeats,
        },
        "measurements": measurements,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
