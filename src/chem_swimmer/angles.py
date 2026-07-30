"""Small, well-tested operations on the circle S1."""

from __future__ import annotations

import numpy as np

TAU = 2.0 * np.pi


def wrap_angle(angle: float) -> float:
    """Map an angle to [0, 2*pi)."""
    return float(angle % TAU)


def heading_vector(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)])


def sample_tumble_heading(
    old_heading: float,
    mean_cosine: float,
    normal_innovation: float,
    uniform_innovation: float,
) -> float:
    """Sample an achiral, rotation-invariant tumble with E[cos(turn)] = a1.

    For positive ``a1`` the turn angle is wrapped-normal around zero. For
    negative ``a1`` it is wrapped-normal around pi (reversal dominated). At
    ``a1 = 0`` it is uniform. This realizes the one kernel moment used by the
    manuscript without requiring SciPy or a numerical inverse.
    """

    if not -1.0 <= mean_cosine <= 1.0:
        raise ValueError("mean_cosine must lie in [-1, 1]")
    if mean_cosine == 0.0:
        turn_angle = TAU * uniform_innovation
    elif abs(mean_cosine) == 1.0:
        turn_angle = 0.0 if mean_cosine > 0 else np.pi
    else:
        center = 0.0 if mean_cosine > 0 else np.pi
        standard_deviation = np.sqrt(-2.0 * np.log(abs(mean_cosine)))
        turn_angle = center + standard_deviation * normal_innovation
    return wrap_angle(old_heading + turn_angle)
