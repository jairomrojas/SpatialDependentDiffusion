"""One obvious place to define an arbitrary chemical landscape.

Select ``kind = "user"`` in ``config/default.toml``. Edit only the two
functions below. Both functions must accept a NumPy array whose final
dimension is two; this makes the same code work for one particle and for the
background image in the video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import ChemicalField, positions_array


@dataclass(frozen=True)
class UserField(ChemicalField):
    baseline: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.baseline):
            raise ValueError("user-field baseline must be finite")
        object.__setattr__(self, "reference_value", float(self.baseline))

    def value(self, position: ArrayLike) -> NDArray[np.float64] | float:
        points = positions_array(position)
        x, y = points[..., 0], points[..., 1]

        # Example: a genuinely smooth source with a weak four-fold deformation.
        # The Cartesian polynomial below equals r^4 cos(4 theta). Including
        # r^4 is essential: ``cos(4 theta)`` by itself has no unique limit at
        # the origin and would make the advertised smooth field singular there.
        radius_squared = x * x + y * y
        fourfold = x**4 - 6.0 * x * x * y * y + y**4
        result = self.baseline + np.exp(-radius_squared / 5.0) * (1.0 + 0.15 * fourfold)
        return float(result) if np.ndim(result) == 0 else result

    def gradient(self, position: ArrayLike) -> NDArray[np.float64]:
        # A central finite difference is intentionally provided here so a user
        # only has to edit value(). For maximum speed, replace this method with
        # the analytic gradient of the chosen field.
        points = positions_array(position)
        h = 1.0e-5
        shift_x = np.array([h, 0.0])
        shift_y = np.array([0.0, h])
        dc_dx = (
            np.asarray(self.value(points + shift_x))
            - np.asarray(self.value(points - shift_x))
        ) / (2.0 * h)
        dc_dy = (
            np.asarray(self.value(points + shift_y))
            - np.asarray(self.value(points - shift_y))
        ) / (2.0 * h)
        return np.stack((dc_dx, dc_dy), axis=-1)
