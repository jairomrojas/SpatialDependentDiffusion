"""Numerical reflection laws for the disk boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .angles import heading_vector, wrap_angle


@dataclass(frozen=True)
class BoundaryResult:
    position: NDArray[np.float64]
    heading: float
    hits: int


def _exit_fraction(
    position: NDArray[np.float64],
    displacement: NDArray[np.float64],
    radius: float,
) -> float:
    """Fraction of a straight increment at which it exits a circle."""

    a = float(displacement @ displacement)
    if a == 0.0:
        return 0.0
    b = 2.0 * float(position @ displacement)
    c = float(position @ position - radius**2)
    discriminant = max(0.0, b * b - 4.0 * a * c)
    roots = (
        (-b - np.sqrt(discriminant)) / (2.0 * a),
        (-b + np.sqrt(discriminant)) / (2.0 * a),
    )
    admissible = [root for root in roots if -1.0e-12 <= root <= 1.0 + 1.0e-12]
    if not admissible:
        raise RuntimeError("could not locate the disk-boundary intersection")
    return float(np.clip(max(admissible), 0.0, 1.0))


def advance_in_disk(
    position: NDArray[np.float64],
    displacement: NDArray[np.float64],
    heading: float,
    radius: float,
    mode: str,
    wall_uniforms: NDArray[np.float64],
) -> BoundaryResult:
    """Apply a positional increment and the selected wall law.

    The straight increment is reflected at each geometric intersection. This
    is an accurate approximation to the reflected process when ``dt`` is small
    relative to the wall-curvature and persistence time scales.
    """

    current = np.asarray(position, dtype=float).copy()
    remaining = np.asarray(displacement, dtype=float).copy()
    angle = float(heading)
    hits = 0
    tolerance = 2.0e-13 * max(radius, 1.0)

    for reflection_index, wall_uniform in enumerate(wall_uniforms):
        trial = current + remaining
        if float(trial @ trial) <= radius**2 * (1.0 + 1.0e-13):
            norm = float(np.linalg.norm(trial))
            if norm > radius:
                trial *= (radius - tolerance) / norm
            return BoundaryResult(trial, wrap_angle(angle), hits)

        fraction = _exit_fraction(current, remaining, radius)
        hit_point = current + fraction * remaining
        normal = hit_point / np.linalg.norm(hit_point)

        remaining = (1.0 - fraction) * remaining
        remaining -= 2.0 * float(remaining @ normal) * normal
        hits += 1

        if mode == "specular":
            direction = heading_vector(angle)
            direction -= 2.0 * float(direction @ normal) * normal
            angle = float(np.arctan2(direction[1], direction[0]))
        elif mode == "diffuse":
            inward_angle = float(np.arctan2(-normal[1], -normal[0]))
            angle = inward_angle + np.pi * (float(wall_uniform) - 0.5)
        elif mode != "sliding":
            raise ValueError(f"unknown boundary mode {mode!r}")

        current = hit_point - tolerance * normal

    # Accept a step that happens to land inside on the final allowed
    # reflection; otherwise keep the state valid and ask for a smaller dt.
    final = current + remaining
    norm = float(np.linalg.norm(final))
    if norm < radius:
        return BoundaryResult(final, wrap_angle(angle), hits)
    if norm >= radius:
        final *= (radius - tolerance) / norm
    raise RuntimeError(
        f"one integration step required more than {len(wall_uniforms)} wall "
        "reflections; reduce simulation.dt"
    )
