"""Interface shared by every prescribed chemical field."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ChemicalField(ABC):
    """A static scalar field c(r) with an analytic spatial gradient."""

    reference_value: float

    @abstractmethod
    def value(self, position: ArrayLike) -> NDArray[np.float64] | float:
        """Return c(r); ``position`` may have shape ``(2,)`` or ``(..., 2)``."""

    @abstractmethod
    def gradient(self, position: ArrayLike) -> NDArray[np.float64]:
        """Return grad(c); the output has the same leading shape plus ``(2,)``."""

    def value_and_gradient(
        self, position: ArrayLike
    ) -> tuple[NDArray[np.float64] | float, NDArray[np.float64]]:
        return self.value(position), self.gradient(position)


def positions_array(position: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(position, dtype=float)
    if array.ndim == 0 or array.shape[-1] != 2:
        raise ValueError("positions must have final dimension 2")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must contain only finite numbers")
    return array
