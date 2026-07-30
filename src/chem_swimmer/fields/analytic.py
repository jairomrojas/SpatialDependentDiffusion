"""Fast vectorized fields that can be selected in the TOML configuration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import ChemicalField, positions_array


@dataclass(frozen=True)
class GaussianField(ChemicalField):
    """A rotated elliptical Gaussian source on a uniform background."""

    baseline: float
    amplitude: float
    center: tuple[float, float]
    sigma: tuple[float, float]
    angle: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.baseline,
            self.amplitude,
            *self.center,
            *self.sigma,
            self.angle,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Gaussian field parameters must be finite")
        if min(self.sigma) <= 0:
            raise ValueError("Gaussian widths must be positive")
        object.__setattr__(self, "reference_value", float(self.baseline))

    @property
    def rotation(self) -> NDArray[np.float64]:
        cosine, sine = np.cos(self.angle), np.sin(self.angle)
        return np.array([[cosine, -sine], [sine, cosine]])

    def _local_coordinates(self, position: ArrayLike) -> NDArray[np.float64]:
        points = positions_array(position)
        return (points - np.asarray(self.center)) @ self.rotation

    def value(self, position: ArrayLike) -> NDArray[np.float64] | float:
        local = self._local_coordinates(position)
        exponent = -0.5 * (
            (local[..., 0] / self.sigma[0]) ** 2 + (local[..., 1] / self.sigma[1]) ** 2
        )
        result = self.baseline + self.amplitude * np.exp(exponent)
        return float(result) if np.ndim(result) == 0 else result

    def gradient(self, position: ArrayLike) -> NDArray[np.float64]:
        _, gradient = self.value_and_gradient(position)
        return gradient

    def value_and_gradient(
        self, position: ArrayLike
    ) -> tuple[NDArray[np.float64] | float, NDArray[np.float64]]:
        local = self._local_coordinates(position)
        exponent = -0.5 * (
            (local[..., 0] / self.sigma[0]) ** 2 + (local[..., 1] / self.sigma[1]) ** 2
        )
        gaussian_part = self.amplitude * np.exp(exponent)
        concentration = self.baseline + gaussian_part
        local_gradient = np.stack(
            (
                -gaussian_part * local[..., 0] / self.sigma[0] ** 2,
                -gaussian_part * local[..., 1] / self.sigma[1] ** 2,
            ),
            axis=-1,
        )
        gradient = local_gradient @ self.rotation.T
        value = float(concentration) if np.ndim(concentration) == 0 else concentration
        return value, gradient


@dataclass(frozen=True)
class LinearField(ChemicalField):
    """A constant gradient, useful for clean mechanism tests."""

    baseline: float
    gradient_vector: tuple[float, float]
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        values = (self.baseline, *self.gradient_vector, *self.center)
        if not np.all(np.isfinite(values)):
            raise ValueError("linear field parameters must be finite")
        object.__setattr__(self, "reference_value", float(self.baseline))

    def value(self, position: ArrayLike) -> NDArray[np.float64] | float:
        points = positions_array(position)
        result = self.baseline + np.sum(
            (points - np.asarray(self.center)) * np.asarray(self.gradient_vector),
            axis=-1,
        )
        return float(result) if np.ndim(result) == 0 else result

    def gradient(self, position: ArrayLike) -> NDArray[np.float64]:
        points = positions_array(position)
        return np.broadcast_to(np.asarray(self.gradient_vector), points.shape).copy()


@dataclass(frozen=True)
class MultiGaussianField(ChemicalField):
    """A sum of Gaussian sources sharing one background concentration."""

    baseline: float
    sources: tuple[GaussianField, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(self.baseline):
            raise ValueError("multi-Gaussian baseline must be finite")
        if not self.sources:
            raise ValueError("multi_gaussian requires at least one source")
        object.__setattr__(self, "reference_value", float(self.baseline))

    def value(self, position: ArrayLike) -> NDArray[np.float64] | float:
        points = positions_array(position)
        result = np.full(points.shape[:-1], self.baseline, dtype=float)
        for source in self.sources:
            result += np.asarray(source.value(points)) - source.baseline
        return float(result) if np.ndim(result) == 0 else result

    def gradient(self, position: ArrayLike) -> NDArray[np.float64]:
        points = positions_array(position)
        result = np.zeros_like(points, dtype=float)
        for source in self.sources:
            result += source.gradient(points)
        return result
