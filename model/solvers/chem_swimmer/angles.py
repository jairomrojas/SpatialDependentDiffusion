"""Small, well-tested operations on the circle S1."""

from __future__ import annotations

import numpy as np

TAU = 2.0 * np.pi


def wrap_angle(angle: float) -> float:
    """Map an angle to [0, 2*pi)."""
    return float(angle % TAU)


def heading_vector(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)])
