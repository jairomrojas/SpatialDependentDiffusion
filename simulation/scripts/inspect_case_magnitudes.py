"""Quantify whether Case-minus-Case-0 density signals live in the bulk or wall.

Usage:
    python scripts/inspect_case_magnitudes.py result_a.npz [result_b.npz ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def summarize(path: Path, core_radius: float) -> None:
    with np.load(path, allow_pickle=False) as saved:
        density = saved["difference_density"].astype(float)
        difference = density - density[0:1]
        coordinates = saved["coordinates"].astype(float)
        x_grid, y_grid = np.meshgrid(coordinates, coordinates)
        radius = np.hypot(x_grid, y_grid)
        core = radius <= core_radius
        wall = (radius > core_radius) & (radius <= 1.0)
        area = (2.0 / len(coordinates)) ** 2
        observable_names = tuple(
            str(name) for name in saved["microscopic_observable_names"]
        )
        observable_values = saved["microscopic_observable_values"].astype(float)
        observable_errors = saved[
            "microscopic_observable_paired_standard_errors"
        ].astype(float)

    print(f"\n{path}  (core radius {core_radius:g})")
    print("case   max TV   core/TV   wall/TV   final delta<eta> final z")
    eta_index = observable_names.index("mean_eta")
    for case_number in range(1, difference.shape[0]):
        total_variation = (
            0.5 * np.sum(np.abs(difference[case_number]), axis=(-2, -1)) * area
        )
        core_variation = (
            0.5
            * np.sum(
                np.abs(difference[case_number][:, core]),
                axis=-1,
            )
            * area
        )
        wall_variation = (
            0.5
            * np.sum(
                np.abs(difference[case_number][:, wall]),
                axis=-1,
            )
            * area
        )
        peak_index = int(np.argmax(total_variation))
        eta_difference = (
            observable_values[eta_index, case_number] - observable_values[eta_index, 0]
        )
        eta_error = observable_errors[eta_index, case_number]
        final_error = eta_error[-1]
        final_z = eta_difference[-1] / final_error if final_error > 0.0 else 0.0
        peak_tv = total_variation[peak_index]
        print(
            f"C{case_number:<2d}  {peak_tv:8.5f}   "
            f"{core_variation[peak_index] / peak_tv:7.3f}   "
            f"{wall_variation[peak_index] / peak_tv:7.3f}   "
            f"{eta_difference[-1]:14.6g}   "
            f"{final_z:7.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--core-radius", type=float, default=0.8)
    args = parser.parse_args()
    if not 0.0 < args.core_radius < 1.0:
        raise ValueError("--core-radius must lie between zero and one")
    for result in args.results:
        summarize(result.expanduser().resolve(), args.core_radius)


if __name__ == "__main__":
    main()
