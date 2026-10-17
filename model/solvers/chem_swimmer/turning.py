"""Achiral, rotation-invariant tumble kernels from the manuscript.

Every kernel samples a *relative* turn angle ``psi``.  Adding ``psi`` to the
pre-tumble heading makes the law rotation invariant.  Achirality is the
requirement that ``psi`` and ``-psi`` have the same law.  The first two cosine
moments are exposed with the manuscript's notation because they set the
polarity and nematic relaxation rates,

    P_turn   = E[cos(psi)],
    P_2_turn = E[cos(2 psi)].

The built-in wrapped-normal, discrete, and tabulated-density laws cover the
usual cases, including deterministic turns.  ``kind = "user"`` accepts any
Python object implementing :class:`TurningKernel`; custom laws use the
readable Python simulator, while built-in wrapped-normal laws can also use the
Numba engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

import numpy as np

TAU = 2.0 * np.pi


@runtime_checkable
class TurningKernel(Protocol):
    """Interface for an achiral, rotation-invariant turning probability law."""

    P_turn: float
    P_2_turn: float
    achiral: bool
    supports_numba: bool

    def sample_turn_angle(
        self, normal_innovation: float, uniform_innovation: float
    ) -> float:
        """Draw the signed relative turn angle ``psi`` in radians."""

    def cosine_moment(self, harmonic: int) -> float:
        """Return ``E[cos(n psi)]``, the damping a tumble applies to harmonic n.

        ``cosine_moment(1)`` and ``cosine_moment(2)`` are ``P_turn`` and
        ``P_2_turn``.  Higher harmonics are needed whenever the angular
        hierarchy is carried past nematic order, as in the finite-lag
        navigation channel of ``information.py``.
        """


def _validate_moments(kernel: TurningKernel) -> TurningKernel:
    if not bool(kernel.achiral):
        raise ValueError("the manuscript requires an achiral turning kernel")
    for name in ("P_turn", "P_2_turn"):
        value = float(getattr(kernel, name))
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"turning-kernel {name} must lie in [-1, 1]")
    if not callable(kernel.sample_turn_angle):
        raise TypeError("a turning kernel must define sample_turn_angle(z, u)")
    return kernel


class WrappedNormalTurningKernel:
    """Even wrapped-normal law selected by its first cosine moment."""

    achiral = True
    supports_numba = True

    def __init__(self, P_turn: float) -> None:
        P_turn = float(P_turn)
        if not np.isfinite(P_turn) or not -1.0 <= P_turn <= 1.0:
            raise ValueError("turning.P_turn must lie in [-1, 1]")
        self.P_turn = P_turn
        # For a wrapped normal, E[cos(k psi)] = cos(k mu) exp(-k^2 sigma^2/2).
        self.P_2_turn = abs(P_turn) ** 4

    def cosine_moment(self, harmonic: int) -> float:
        """``E[cos(n psi)] = cos(n mu) |P_turn|^{n^2}`` for a wrapped normal."""

        n = int(harmonic)
        magnitude = abs(self.P_turn) ** (n * n)
        # A negative P_turn is the mean turn mu = pi, so cos(n mu) = (-1)^n.
        sign = 1.0 if self.P_turn >= 0.0 or n % 2 == 0 else -1.0
        return float(sign * magnitude)

    def sample_turn_angle(
        self, normal_innovation: float, uniform_innovation: float
    ) -> float:
        if self.P_turn == 0.0:
            return TAU * (float(uniform_innovation) - 0.5)
        if abs(self.P_turn) == 1.0:
            return 0.0 if self.P_turn > 0.0 else np.pi
        center = 0.0 if self.P_turn > 0.0 else np.pi
        sigma = np.sqrt(-2.0 * np.log(abs(self.P_turn)))
        return float(center + sigma * float(normal_innovation))


class DiscreteTurningKernel:
    """Arbitrary symmetric discrete probability measure for ``|psi|``.

    ``absolute_angles`` lie in ``[0, pi]``.  For an interior angle the two
    signed turns are equiprobable; the sign is immaterial at 0 and pi.
    """

    achiral = True
    supports_numba = False

    def __init__(
        self, absolute_angles: list[float], probabilities: list[float]
    ) -> None:
        angles = np.asarray(absolute_angles, dtype=float)
        weights = np.asarray(probabilities, dtype=float)
        if angles.ndim != 1 or weights.ndim != 1 or len(angles) != len(weights):
            raise ValueError(
                "turning.absolute_angles and turning.probabilities must be equal-length lists"
            )
        if len(angles) == 0 or not np.all(np.isfinite(angles)):
            raise ValueError("the discrete turning kernel must contain finite angles")
        if np.any(angles < 0.0) or np.any(angles > np.pi):
            raise ValueError("turning.absolute_angles must lie in [0, pi]")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("turning.probabilities must be finite and nonnegative")
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("turning.probabilities must have positive total mass")
        order = np.argsort(angles)
        self.absolute_angles = angles[order]
        self.probabilities = weights[order] / total
        self._cdf = np.cumsum(self.probabilities)
        self._cdf[-1] = 1.0
        self.P_turn = float(np.sum(self.probabilities * np.cos(self.absolute_angles)))
        self.P_2_turn = float(
            np.sum(self.probabilities * np.cos(2.0 * self.absolute_angles))
        )

    def cosine_moment(self, harmonic: int) -> float:
        return float(
            np.sum(self.probabilities * np.cos(float(harmonic) * self.absolute_angles))
        )

    def sample_turn_angle(
        self, normal_innovation: float, uniform_innovation: float
    ) -> float:
        index = min(
            len(self.absolute_angles) - 1,
            int(np.searchsorted(self._cdf, float(uniform_innovation), side="right")),
        )
        magnitude = float(self.absolute_angles[index])
        return -magnitude if float(normal_innovation) < 0.0 else magnitude


class TabulatedTurningKernel:
    """Even piecewise-linear density for ``|psi|`` on ``[0, pi]``.

    Sampling inverts the exact quadratic segment CDF, and the two reported
    cosine moments are integrated analytically over each linear segment.
    Thus this is a precisely defined kernel, not a grid-dependent rejection
    sampler or a midpoint approximation.
    """

    achiral = True
    supports_numba = False

    def __init__(self, absolute_angles: list[float], density: list[float]) -> None:
        angles = np.asarray(absolute_angles, dtype=float)
        values = np.asarray(density, dtype=float)
        if angles.ndim != 1 or values.ndim != 1 or len(angles) != len(values):
            raise ValueError(
                "turning.absolute_angles and turning.density must be equal-length lists"
            )
        if len(angles) < 2 or not np.all(np.isfinite(angles)):
            raise ValueError(
                "a tabulated turning density needs at least two finite angles"
            )
        if np.any(np.diff(angles) <= 0.0) or angles[0] < 0.0 or angles[-1] > np.pi:
            raise ValueError(
                "turning.absolute_angles must increase strictly within [0, pi]"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("turning.density must be finite and nonnegative")
        widths = np.diff(angles)
        increments = 0.5 * (values[:-1] + values[1:]) * widths
        total = float(np.sum(increments))
        if total <= 0.0:
            raise ValueError("turning.density must have positive integral")
        self.absolute_angles = angles
        self.density = values / total
        self._segment_mass = increments / total
        self._cdf = np.concatenate(([0.0], np.cumsum(self._segment_mass)))
        self._cdf[-1] = 1.0
        self.P_turn = float(np.clip(self._cosine_moment(1), -1.0, 1.0))
        self.P_2_turn = float(np.clip(self._cosine_moment(2), -1.0, 1.0))

    def cosine_moment(self, harmonic: int) -> float:
        return float(np.clip(self._cosine_moment(harmonic), -1.0, 1.0))

    def _cosine_moment(self, harmonic: int) -> float:
        result = 0.0
        k = float(harmonic)
        for left, right, y_left, y_right in zip(
            self.absolute_angles[:-1],
            self.absolute_angles[1:],
            self.density[:-1],
            self.density[1:],
            strict=True,
        ):
            slope = (y_right - y_left) / (right - left)
            intercept = y_left - slope * left

            def primitive(
                x: float,
                intercept: float = float(intercept),
                slope: float = float(slope),
            ) -> float:
                return intercept * np.sin(k * x) / k + slope * (
                    x * np.sin(k * x) / k + np.cos(k * x) / (k * k)
                )

            result += primitive(float(right)) - primitive(float(left))
        return float(result)

    def sample_turn_angle(
        self, normal_innovation: float, uniform_innovation: float
    ) -> float:
        u = min(1.0, max(0.0, float(uniform_innovation)))
        index = min(
            len(self._segment_mass) - 1,
            int(np.searchsorted(self._cdf, u, side="right") - 1),
        )
        left = float(self.absolute_angles[index])
        width = float(self.absolute_angles[index + 1] - left)
        y_left = float(self.density[index])
        slope = float((self.density[index + 1] - self.density[index]) / width)
        target = u - float(self._cdf[index])
        if abs(slope) <= 1.0e-14:
            offset = target / y_left if y_left > 0.0 else 0.0
        else:
            discriminant = max(0.0, y_left * y_left + 2.0 * slope * target)
            denominator = y_left + np.sqrt(discriminant)
            offset = 2.0 * target / denominator if denominator > 0.0 else width
        magnitude = left + min(width, max(0.0, offset))
        return -magnitude if float(normal_innovation) < 0.0 else magnitude


def turning_cosine_moment(kernel: TurningKernel, harmonic: int) -> float:
    """``E[cos(n psi)]`` for any kernel, including a user law without the method.

    The first two harmonics are always available through ``P_turn`` and
    ``P_2_turn``; a custom law that does not expose the general moment can
    therefore still be used everywhere the manuscript needs only polarity and
    nematic relaxation.
    """

    n = int(harmonic)
    if n < 0:
        raise ValueError("a cosine moment needs a nonnegative harmonic")
    if n == 0:
        return 1.0
    moment = getattr(kernel, "cosine_moment", None)
    if callable(moment):
        return float(moment(n))
    if n == 1:
        return float(kernel.P_turn)
    if n == 2:
        return float(kernel.P_2_turn)
    raise ValueError(
        f"turning kernel {type(kernel).__name__} exposes only the first two "
        f"cosine moments; harmonic {n} requires a cosine_moment(n) method"
    )


def create_turning_kernel(settings: Mapping[str, Any]) -> TurningKernel:
    """Construct and validate a turning law from the ``[turning]`` TOML table."""

    data = dict(settings)
    kind = str(data.pop("kind", "wrapped_normal")).lower()
    if kind == "wrapped_normal":
        unknown = sorted(set(data) - {"P_turn"})
        if unknown:
            raise ValueError(
                f"unknown wrapped-normal turning key(s): {', '.join(unknown)}"
            )
        if "P_turn" not in data:
            raise ValueError("turning.P_turn is required for kind='wrapped_normal'")
        kernel: TurningKernel = WrappedNormalTurningKernel(data["P_turn"])
    elif kind == "discrete":
        unknown = sorted(set(data) - {"absolute_angles", "probabilities"})
        if unknown:
            raise ValueError(f"unknown discrete turning key(s): {', '.join(unknown)}")
        kernel = DiscreteTurningKernel(
            list(data.get("absolute_angles", [])),
            list(data.get("probabilities", [])),
        )
    elif kind == "tabulated":
        unknown = sorted(set(data) - {"absolute_angles", "density"})
        if unknown:
            raise ValueError(f"unknown tabulated turning key(s): {', '.join(unknown)}")
        kernel = TabulatedTurningKernel(
            list(data.get("absolute_angles", [])),
            list(data.get("density", [])),
        )
    elif kind == "user":
        module_name = str(data.pop("module", ""))
        factory_name = str(data.pop("factory", "create_turning_kernel"))
        if not module_name:
            raise ValueError("turning.module is required for kind='user'")
        factory = getattr(import_module(module_name), factory_name)
        kernel = factory(data)
        if not isinstance(kernel, TurningKernel):
            raise TypeError(
                "a user turning factory must return an object implementing TurningKernel"
            )
    else:
        raise ValueError(
            "turning.kind must be wrapped_normal, discrete, tabulated, or user"
        )
    return _validate_moments(kernel)


def sample_tumble_heading(
    old_heading: float,
    kernel: TurningKernel,
    normal_innovation: float,
    uniform_innovation: float,
) -> float:
    """Sample ``theta' = theta + psi`` and wrap it to ``[0, 2 pi)``."""

    return float(
        (old_heading + kernel.sample_turn_angle(normal_innovation, uniform_innovation))
        % TAU
    )
