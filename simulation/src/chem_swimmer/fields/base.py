"""Interface shared by every prescribed chemical field."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

# One canonical quadrature defines the manuscript reference level eta_0 for
# every microscopic and macroscopic path.  Spectral-assembly resolution is a
# separate numerical choice and must not silently change the response law.
ETA0_RADIAL_POINTS = 96
ETA0_ANGULAR_POINTS = 256


class ChemicalField(ABC):
    """The manuscript's static scalar nutrient field ``eta(r)``."""

    reference_value: float

    @abstractmethod
    def value(self, position: ArrayLike) -> NDArray[np.float64] | float:
        """Return ``eta(r)``; ``position`` may have shape ``(2,)`` or ``(..., 2)``."""

    @abstractmethod
    def gradient(self, position: ArrayLike) -> NDArray[np.float64]:
        """Return ``grad(eta)`` with the same leading shape plus ``(2,)``."""

    def value_and_gradient(
        self, position: ArrayLike
    ) -> tuple[NDArray[np.float64] | float, NDArray[np.float64]]:
        return self.value(position), self.gradient(position)

    def laplacian(self, position: ArrayLike) -> NDArray[np.float64] | float:
        """Return ``lap(eta)``, the curvature that drives passive uptake decay.

        Every built-in field overrides this with its closed form.  The default
        is a symmetric second difference, accurate enough for a diagnostic but
        not for a solver.
        """

        points = positions_array(position)
        step = 1.0e-4
        result = np.zeros(points.shape[:-1], dtype=float)
        for axis in range(2):
            shift = np.zeros(2)
            shift[axis] = step
            result += (
                np.asarray(self.value(points + shift), dtype=float)
                - 2.0 * np.asarray(self.value(points), dtype=float)
                + np.asarray(self.value(points - shift), dtype=float)
            ) / step**2
        return result


def positions_array(position: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(position, dtype=float)
    if array.ndim == 0 or array.shape[-1] != 2:
        raise ValueError("positions must have final dimension 2")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must contain only finite numbers")
    return array


def _disk_area_average_uncached(
    field: ChemicalField,
    radius: float,
    radial_points: int,
    angular_points: int,
) -> float:
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(radial_points)
    radial_nodes = 0.5 * radius * (radial_nodes + 1.0)
    radial_weights = 0.5 * radius * radial_weights
    angles = np.linspace(0.0, 2.0 * np.pi, angular_points, endpoint=False)
    radial_grid, angle_grid = np.meshgrid(radial_nodes, angles, indexing="ij")
    points = np.stack(
        (radial_grid * np.cos(angle_grid), radial_grid * np.sin(angle_grid)),
        axis=-1,
    )
    eta = np.asarray(field.value(points), dtype=float)
    quadrature = (radial_weights * radial_nodes)[:, None] * (
        2.0 * np.pi / angular_points
    )
    return float(np.sum(eta * quadrature) / (np.pi * radius**2))


@lru_cache(maxsize=128)
def _disk_area_average_cached(
    field: ChemicalField,
    radius: float,
    radial_points: int,
    angular_points: int,
) -> float:
    return _disk_area_average_uncached(field, radius, radial_points, angular_points)


def disk_area_average(
    field: ChemicalField,
    radius: float,
    *,
    radial_points: int = ETA0_RADIAL_POINTS,
    angular_points: int = ETA0_ANGULAR_POINTS,
) -> float:
    """Compute ``eta_0 = |Omega|^-1 integral_Omega eta`` on the disk.

    Smooth configured fields are cached, so paired simulations share exactly
    the same reference value without repeating quadrature. Unhashable custom
    field objects use the same deterministic calculation without caching.
    """

    if radius <= 0.0 or radial_points < 2 or angular_points < 4:
        raise ValueError("disk-average quadrature settings must be positive")
    try:
        return _disk_area_average_cached(field, radius, radial_points, angular_points)
    except TypeError:
        return _disk_area_average_uncached(field, radius, radial_points, angular_points)
