"""The manuscript linear response law and its case-by-case selection.

Symbol map to ``manuscript/main.tex``: ``gamma1`` is the polarity relaxation
rate ``D_R^(0)`` of Eq. (gamma), ``effective_diffusivity`` is
``D_eff^(0)`` of Eq. (macroscopic-effective-diffusivity), and ``mobility``
is the chemotactic mobility ``kappa`` of Eq. (drift-mobility), defined by
the long-time density flux ``J = -D_eff grad(rho) + kappa rho grad(eta)``.
"""

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
    eta: float
    delta_eta: float
    gradient: NDArray[np.float64]
    zeta: float
    v: float
    D_T: float
    D_R: float
    tumble_rate: float


@dataclass(frozen=True)
class KineticLaw:
    background: BackgroundKinetics
    responses: Responses
    eta0: float

    def evaluate(
        self,
        eta: float,
        gradient: NDArray[np.float64],
        heading: float,
        position: NDArray[np.float64] | None = None,
    ) -> KineticState:
        delta_eta = float(eta - self.eta0)
        zeta = float(heading_vector(heading) @ gradient)
        response = self.responses
        base = self.background

        values = {
            "v": base.v0 + response.v_eta * delta_eta + response.v_zeta * zeta,
            "D_T": base.D_T0 + response.D_T_eta * delta_eta + response.D_T_zeta * zeta,
            "D_R": base.D_R0 + response.D_R_eta * delta_eta + response.D_R_zeta * zeta,
            "lambda": base.lambda0
            + response.lambda_eta * delta_eta
            + response.lambda_zeta * zeta,
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
            eta=float(eta),
            delta_eta=delta_eta,
            gradient=np.asarray(gradient, dtype=float),
            zeta=zeta,
            v=values["v"],
            D_T=values["D_T"],
            D_R=values["D_R"],
            tumble_rate=values["lambda"],
        )


def macroscopic_coefficients(
    background: BackgroundKinetics, responses: Responses, P_turn: float
) -> tuple[float, float, float]:
    """Return ``(gamma1, D_eff, kappa)``; see the module docstring for the map.

    ``kappa`` collects both blocks of Eq. (drift-mobility): the gradient
    responses ``v_zeta``, ``D_R_zeta``, ``lambda_zeta`` and ``chi_theta``,
    and the level responses ``D_T_eta`` and ``v_eta``, which drive density
    even without directional sensing because a position-dependent
    diffusivity or speed transports it.
    """

    gamma1 = background.D_R0 + background.lambda0 * (1.0 - P_turn)
    if gamma1 == 0.0:
        # The stochastic dynamics remain valid, but the long-time isotropic
        # closure used to derive D_eff and kappa assumes orientational relaxation.
        return gamma1, float("nan"), float("nan")
    effective_diffusivity = background.D_T0 + background.v0**2 / (2.0 * gamma1)
    mobility = (
        responses.v_zeta / 2.0
        - responses.D_T_eta
        + background.v0 * responses.chi_theta / (2.0 * gamma1)
        - background.v0
        * (
            responses.v_eta
            + responses.D_R_zeta
            + (1.0 - P_turn) * responses.lambda_zeta
        )
        / (2.0 * gamma1)
    )
    return gamma1, effective_diffusivity, mobility


def delta_D_eff_coefficient(
    background: BackgroundKinetics, responses: Responses, P_turn: float
) -> float:
    """Return ``delta D_eff'``: the shift of the effective diffusivity per unit
    ``eta_1``, i.e. the bracket of the manuscript's Eq. (diffusivity-spectral-
    shift-explicit), so that ``D_eff(r) = D_eff^(0) + eps * this * eta_1(r)``.
    """

    gamma1 = background.D_R0 + background.lambda0 * (1.0 - P_turn)
    if gamma1 == 0.0:
        return float("nan")
    gamma_eta = responses.D_R_eta + (1.0 - P_turn) * responses.lambda_eta
    return (
        responses.D_T_eta
        + background.v0 * responses.v_eta / gamma1
        - background.v0**2 * gamma_eta / (2.0 * gamma1**2)
    )


def pulse_centroid_drift(
    background: BackgroundKinetics, responses: Responses, P_turn: float
) -> float:
    """Return ``V = kappa + delta D_eff'`` of the manuscript's Eq. (pulse-drift).

    ``kappa`` is the mobility that fixes the accumulated state; ``V`` is what
    the centroid of a spreading release actually moves at, because a spatially
    varying ``D_eff`` displaces the centroid on its own.  The two differ, and
    ``D_T_eta`` cancels from ``V`` exactly: a pulse spreading in a linearly
    varying Ito diffusivity does not move its centroid.
    """

    _, _, mobility = macroscopic_coefficients(background, responses, P_turn)
    return mobility + delta_D_eff_coefficient(background, responses, P_turn)


def angular_steering_rate(
    responses: Responses,
    gradient: NDArray[np.float64],
    heading: float,
) -> float:
    """Return ``chi_theta e_perp(theta) . grad(eta)`` for Case 9.

    With ``e_perp=(-sin(theta), cos(theta))``, a positive ``chi_theta``
    increases alignment with ``grad(eta)`` because
    ``d(e.grad(eta))/dt = chi_theta (e_perp.grad(eta))**2`` when the
    gradient is locally frozen.
    """

    direction = heading_vector(heading)
    perpendicular = np.asarray((-direction[1], direction[0]), dtype=float)
    return float(responses.chi_theta * (perpendicular @ gradient))
