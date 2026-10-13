"""Analytic P1 (dipole) estimate of the sliding-wall angular boundary layer.

The manuscript's closed bulk equations show that Cases 2, 4, and 5 have zero chemotactic
mobility ``kappa`` in the closed bulk equations because a uniform, isotropic reference state has
zero polarity, ``p^(0) = 0``. This cancellation does not hold at the disk
boundary: the sliding wall
(``boundary.py``) absorbs the attempted outward normal displacement but
never rotates the heading, so a swimmer heading into the wall is held there
until rotational diffusion or a tumble releases it. That produces a nonzero background polarity layer,
``p_wall^(0)``, which is what actually resurrects Cases 2, 4, and 5 in a
bounded disk.

This module derives and evaluates a closed-form estimate of that layer for
the *background* (chemistry-free) dynamics alone -- the same calculation
that sets the isotropic reference state used throughout section 5 of the
manuscript. The derivation is a boundary-layer version of the bulk closure
already used for the closed density--polarity system: truncate the angular
hierarchy at the
dipole (drop the nematic moment ``Q``), but, unlike the bulk closure, do not
discard second derivatives in space -- the whole point of a boundary layer
is that gradients are steep on the layer width, not on the field length
``L``. The derivation and result are retained here beside their implementation.

Local half-space coordinates: let ``y >= 0`` be the distance from the wall
into the disk and ``phi`` the heading angle measured from the inward normal
(``phi = 0`` points into the bulk). Flattening the wall (valid once the
layer width is small compared to ``R``) and dropping chemistry, the
steady dipole-truncated moment equations are

    v0 * p(y)  = D_T0 * rho'(y)                              (flux-free bulk)
    D_T0 * p''(y) - gamma1 * p(y) - (v0/2) * rho'(y) = 0      (polarity)

with ``p`` the wall-normal polarity component. Together these give
``p'' = kappa^2 p`` with

    kappa = sqrt(D_T0 * gamma1 + v0^2/2) / D_T0.

The sliding wall imposes a zero-flux condition *for every heading*,
``v0 cos(phi) P(0, phi) = D_T0 d/dy P(0, phi)``. Its dipole (cos-phi)
moment, dropping the nematic correction, is a Marshak-type boundary
condition

    v0 * rho(0) / 2 = D_T0 * p'(0).

Solving the two-point problem (decay as y -> infinity, this boundary
condition at y = 0) gives the wall dipole, oriented from the bulk toward
the wall:

    |p_wall^(0)| / rho_bulk = sqrt(Pe + Pe^2 / 2) / 2,   Pe = v0^2 / (D_T0 * gamma1)

and the layer decays as ``p(y) = p_wall^(0) * exp(-kappa * y)``, width
``1 / kappa``.

This is a *P1/dipole estimate*, not an exact solution: it truncates the
nematic and higher angular harmonics everywhere in the layer, and it
neglects disk curvature. It is quantitatively most trustworthy when
``Pe = O(1)`` or smaller; the default configuration has ``Pe ~ 2``, so the
estimate should be read as an order-of-magnitude prediction to be checked
against the directly measured polarity field (``render_polarity_report``),
not as a validated closed form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import BackgroundKinetics


@dataclass(frozen=True)
class SlidingWallBoundaryLayer:
    """Dipole (P1) estimate of the background sliding-wall polarity layer."""

    gamma1: float
    peclet: float
    decay_rate: float
    width: float
    wall_polarity_over_bulk_density: float


def sliding_wall_boundary_layer(
    background: BackgroundKinetics, P_turn: float
) -> SlidingWallBoundaryLayer:
    """Return the dipole-truncated sliding-wall boundary-layer estimate.

    All quantities are dimensionless, in the same units as the rest of the
    package (length in ``R``, time in ``R^2/D_T0``). ``gamma1`` and
    ``peclet`` are ``nan`` when ``gamma1 = 0`` (no orientational relaxation,
    so the layer never reaches a steady width); ``decay_rate`` is ``inf``
    and ``wall_polarity_over_bulk_density`` is ``0`` when ``v0 = 0`` (no
    self-propulsion, so nothing accumulates at the wall).
    """

    v0 = background.v0
    D_T0 = background.D_T0
    gamma1 = background.D_R0 + background.lambda0 * (1.0 - P_turn)
    if gamma1 <= 0.0:
        return SlidingWallBoundaryLayer(
            gamma1=gamma1,
            peclet=float("nan"),
            decay_rate=float("nan"),
            width=float("nan"),
            wall_polarity_over_bulk_density=float("nan"),
        )
    if v0 == 0.0:
        decay_rate = math.sqrt(gamma1 / D_T0)
        return SlidingWallBoundaryLayer(
            gamma1=gamma1,
            peclet=0.0,
            decay_rate=decay_rate,
            width=1.0 / decay_rate,
            wall_polarity_over_bulk_density=0.0,
        )

    peclet = v0**2 / (D_T0 * gamma1)
    decay_rate = math.sqrt(D_T0 * gamma1 + 0.5 * v0**2) / D_T0
    ratio = math.sqrt(peclet + 0.5 * peclet**2) / 2.0
    return SlidingWallBoundaryLayer(
        gamma1=gamma1,
        peclet=peclet,
        decay_rate=decay_rate,
        width=1.0 / decay_rate,
        wall_polarity_over_bulk_density=ratio,
    )
