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

    def laplacian(self, position: ArrayLike) -> NDArray[np.float64] | float:
        local = self._local_coordinates(position)
        exponent = -0.5 * (
            (local[..., 0] / self.sigma[0]) ** 2 + (local[..., 1] / self.sigma[1]) ** 2
        )
        gaussian_part = self.amplitude * np.exp(exponent)
        curvature = (
            local[..., 0] ** 2 / self.sigma[0] ** 4
            - 1.0 / self.sigma[0] ** 2
            + local[..., 1] ** 2 / self.sigma[1] ** 4
            - 1.0 / self.sigma[1] ** 2
        )
        result = gaussian_part * curvature
        return float(result) if np.ndim(result) == 0 else result

    def value_and_gradient(
        self, position: ArrayLike
    ) -> tuple[NDArray[np.float64] | float, NDArray[np.float64]]:
        local = self._local_coordinates(position)
        exponent = -0.5 * (
            (local[..., 0] / self.sigma[0]) ** 2 + (local[..., 1] / self.sigma[1]) ** 2
        )
        gaussian_part = self.amplitude * np.exp(exponent)
        eta = self.baseline + gaussian_part
        local_gradient = np.stack(
            (
                -gaussian_part * local[..., 0] / self.sigma[0] ** 2,
                -gaussian_part * local[..., 1] / self.sigma[1] ** 2,
            ),
            axis=-1,
        )
        gradient = local_gradient @ self.rotation.T
        value = float(eta) if np.ndim(eta) == 0 else eta
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

    def laplacian(self, position: ArrayLike) -> NDArray[np.float64] | float:
        points = positions_array(position)
        return np.zeros(points.shape[:-1], dtype=float)


@dataclass(frozen=True)
class PeriodicField(ChemicalField):
    """A single cosine plane wave, ``eta = baseline + A cos(q.r + phase)``.

    One landscape mode, so every spectral statement in the paper reads off
    directly: the mean-square contrast is ``A^2/2``, the curvature is
    ``-|q|^2`` times the modulation, and the diffusive lifetime of the mode is
    ``1/(D_eff |q|^2)``.  This is the landscape shape used by the periodic
    area-restricted-search literature, kept here so the same model can be run
    on it.
    """

    baseline: float
    amplitude: float
    wavevector: tuple[float, float]
    phase: float = 0.0

    def __post_init__(self) -> None:
        values = (self.baseline, self.amplitude, *self.wavevector, self.phase)
        if not np.all(np.isfinite(values)):
            raise ValueError("periodic field parameters must be finite")
        if not np.any(np.asarray(self.wavevector)):
            raise ValueError("a periodic field requires a nonzero wavevector")
        object.__setattr__(self, "reference_value", float(self.baseline))

    @property
    def wavenumber(self) -> float:
        return float(np.hypot(*self.wavevector))

    @property
    def wavelength(self) -> float:
        return float(2.0 * np.pi / self.wavenumber)

    def _phase(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sum(points * np.asarray(self.wavevector), axis=-1) + self.phase

    def value(self, position: ArrayLike) -> NDArray[np.float64] | float:
        points = positions_array(position)
        result = self.baseline + self.amplitude * np.cos(self._phase(points))
        return float(result) if np.ndim(result) == 0 else result

    def gradient(self, position: ArrayLike) -> NDArray[np.float64]:
        _, gradient = self.value_and_gradient(position)
        return gradient

    def value_and_gradient(
        self, position: ArrayLike
    ) -> tuple[NDArray[np.float64] | float, NDArray[np.float64]]:
        points = positions_array(position)
        angle = self._phase(points)
        value = self.baseline + self.amplitude * np.cos(angle)
        slope = -self.amplitude * np.sin(angle)
        gradient = slope[..., None] * np.asarray(self.wavevector)
        return (float(value) if np.ndim(value) == 0 else value), gradient

    def laplacian(self, position: ArrayLike) -> NDArray[np.float64] | float:
        points = positions_array(position)
        result = -(self.wavenumber**2) * self.amplitude * np.cos(self._phase(points))
        return float(result) if np.ndim(result) == 0 else result


@dataclass(frozen=True)
class MultiGaussianField(ChemicalField):
    """A sum of Gaussian sources sharing one background eta."""

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

    def laplacian(self, position: ArrayLike) -> NDArray[np.float64] | float:
        points = positions_array(position)
        result = np.zeros(points.shape[:-1], dtype=float)
        for source in self.sources:
            result += np.asarray(source.laplacian(points), dtype=float)
        return float(result) if np.ndim(result) == 0 else result

    def value_and_gradient(
        self, position: ArrayLike
    ) -> tuple[NDArray[np.float64] | float, NDArray[np.float64]]:
        """Both fields in one pass over the sources.

        Without this override the base class calls ``value`` and then
        ``gradient``, and every source's ``gradient`` recomputes its value and
        discards it -- four Gaussians evaluated eight times instead of four.
        The arithmetic per source is unchanged, so results are bit for bit
        identical to the two-call path.
        """

        points = positions_array(position)
        value = np.full(points.shape[:-1], self.baseline, dtype=float)
        gradient = np.zeros_like(points, dtype=float)
        for source in self.sources:
            source_value, source_gradient = source.value_and_gradient(points)
            value += np.asarray(source_value) - source.baseline
            gradient += source_gradient
        return (float(value) if np.ndim(value) == 0 else value), gradient
