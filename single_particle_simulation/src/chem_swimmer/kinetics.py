"""The linear response law in equation (2) and its case-by-case selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .angles import heading_vector
from .config import BackgroundKinetics, Responses


class NonPhysicalParameterError(ValueError):
    """Raised instead of silently clipping a negative speed, rate, or diffusivity."""


@dataclass(frozen=True)
class KineticState:
    concentration: float
    delta_c: float
    gradient: NDArray[np.float64]
    alignment: float
    v: float
    D_T: float
    D_R: float
    tumble_rate: float


@dataclass(frozen=True)
class KineticLaw:
    background: BackgroundKinetics
    responses: Responses
    reference_concentration: float

    def evaluate(
        self,
        concentration: float,
        gradient: NDArray[np.float64],
        heading: float,
        position: NDArray[np.float64] | None = None,
    ) -> KineticState:
        delta_c = float(concentration - self.reference_concentration)
        alignment = float(heading_vector(heading) @ gradient)
        response = self.responses
        base = self.background

        values = {
            "v": base.v0 + response.v_c * delta_c + response.v_S * alignment,
            "D_T": base.D_T0 + response.D_T_c * delta_c + response.D_T_S * alignment,
            "D_R": base.D_R0 + response.D_R_c * delta_c + response.D_R_S * alignment,
            "lambda": base.lambda0
            + response.lambda_c * delta_c
            + response.lambda_S * alignment,
        }
        negative = {name: value for name, value in values.items() if value < 0.0}
        if negative:
            location = "" if position is None else f" at r={np.asarray(position)}"
            details = ", ".join(
                f"{name}={value:.6g}" for name, value in negative.items()
            )
            raise NonPhysicalParameterError(
                f"linear response produced a negative kinetic parameter{location}: "
                f"{details}. Reduce the corresponding sensitivity or field amplitude."
            )

        return KineticState(
            concentration=float(concentration),
            delta_c=delta_c,
            gradient=np.asarray(gradient, dtype=float),
            alignment=alignment,
            v=values["v"],
            D_T=values["D_T"],
            D_R=values["D_R"],
            tumble_rate=values["lambda"],
        )


def macroscopic_coefficients(
    background: BackgroundKinetics, responses: Responses
) -> tuple[float, float, float]:
    """Return ``(gamma0, D_eff, kappa)`` from equations (13), (17), and (18)."""

    gamma0 = background.D_R0 + background.lambda0 * (1.0 - background.a1)
    if gamma0 == 0.0:
        # The stochastic dynamics remain valid, but the long-time isotropic
        # closure used to derive D_eff and kappa assumes orientational relaxation.
        return gamma0, float("nan"), float("nan")
    effective_diffusivity = background.D_T0 + background.v0**2 / (2.0 * gamma0)
    mobility = (
        responses.v_S / 2.0
        - responses.D_T_c
        - background.v0
        * (responses.v_c + responses.D_R_S + (1.0 - background.a1) * responses.lambda_S)
        / (2.0 * gamma0)
    )
    return gamma0, effective_diffusivity, mobility
