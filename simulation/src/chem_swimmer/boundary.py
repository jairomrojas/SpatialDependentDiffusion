"""Sliding (no-flux) contact at the disk wall, with the heading unchanged.

The manuscript's wall is impermeable but does not reorient a cell: the
outward normal component of an attempted displacement is removed, and the
cell continues along the boundary through the surviving tangential motion.
The heading is an internal state and is never touched, so a cell pressed
into the wall stays pressed there until rotational diffusion or a tumble
turns it away.  This is the microscopic realization of the phase-space
condition ``nu . J_sp = 0`` imposed separately for every heading, and it is
the wall rule assumed by ``boundary_layer.py``.

Specular reflection of the displacement would instead send such a cell back
into the bulk while its heading still points outward, which is neither the
Skorokhod normal-constraint process nor the condition the closure uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .angles import wrap_angle


@dataclass(frozen=True)
class BoundaryResult:
    position: NDArray[np.float64]
    heading: float
    hits: int


def _exit_fraction(
    position: NDArray[np.float64],
    displacement: NDArray[np.float64],
    radius: float,
) -> float:
    """Fraction of a straight increment at which it exits a circle."""

    a = float(displacement @ displacement)
    if a == 0.0:
        return 0.0
    b = 2.0 * float(position @ displacement)
    c = float(position @ position - radius**2)
    discriminant = max(0.0, b * b - 4.0 * a * c)
    roots = (
        (-b - np.sqrt(discriminant)) / (2.0 * a),
        (-b + np.sqrt(discriminant)) / (2.0 * a),
    )
    admissible = [root for root in roots if -1.0e-12 <= root <= 1.0 + 1.0e-12]
    if not admissible:
        raise RuntimeError("could not locate the disk-boundary intersection")
    return float(np.clip(max(admissible), 0.0, 1.0))


def advance_in_disk(
    position: NDArray[np.float64],
    displacement: NDArray[np.float64],
    heading: float,
    radius: float,
    max_reflections: int = 64,
) -> BoundaryResult:
    """Apply one positional increment under the manuscript's sliding wall.

    A straight increment that stays inside the disk is taken as it is.  One
    that would leave the disk is advanced to the wall; the outward normal
    part of what remains is absorbed by the impermeability constraint, and
    the tangential part is travelled as arc length along the boundary.  A
    single contact therefore consumes the whole increment, and the heading is
    returned unchanged.  ``max_reflections`` is accepted for backward
    compatibility and is unused.
    """

    del max_reflections
    current = np.asarray(position, dtype=float)
    increment = np.asarray(displacement, dtype=float)
    angle = wrap_angle(float(heading))
    tolerance = 2.0e-13 * max(radius, 1.0)

    trial = current + increment
    if float(trial @ trial) <= radius**2 * (1.0 + 1.0e-13):
        norm = float(np.linalg.norm(trial))
        if norm > radius:
            trial = trial * ((radius - tolerance) / norm)
        return BoundaryResult(trial, angle, 0)

    fraction = _exit_fraction(current, increment, radius)
    contact = current + fraction * increment
    contact_norm = float(np.linalg.norm(contact))
    if contact_norm == 0.0:
        raise RuntimeError("degenerate wall contact at the disk center")
    normal = contact / contact_norm
    tangent = np.array((-normal[1], normal[0]), dtype=float)

    remaining = (1.0 - fraction) * increment
    # Absorb the outward normal component; travel the tangential component as
    # arc length so that sliding conserves the tangential speed exactly.
    arc = float(remaining @ tangent)
    rotation = arc / radius
    cosine, sine = np.cos(rotation), np.sin(rotation)
    slid = np.array(
        (
            normal[0] * cosine - normal[1] * sine,
            normal[0] * sine + normal[1] * cosine,
        ),
        dtype=float,
    )
    return BoundaryResult(slid * (radius - tolerance), angle, 1)
