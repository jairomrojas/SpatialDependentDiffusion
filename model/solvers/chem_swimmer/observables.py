"""Macroscopic transport observables of a release experiment.

Three scalars summarize a release: the average polarity ``P = int p d2r / N``,
the centroid ``X = int r rho d2r / N``, and the second moment
``S = int |r|^2 rho d2r / N``.  Integrating the polar-closed system of
``manuscript/main.tex`` over the plane turns its two field equations into a
small closed set of ordinary differential equations for these scalars, exact to
first order in the landscape amplitude.  The auxiliary scalar
``Omega = int r . p d2r / N`` closes the set; it is the outward polarity
moment that feeds spreading.

Three independent paths produce the same four curves:

``particle_observables``
    plain ensemble means over the random walk -- no binning, no bandwidth;
``finite_volume_observables``
    grid moments of the conservative density--polarity solver;
``predicted_observables``
    the closed moment system driven by the uniform-background propagator.

The prediction is linear in the response coefficients, so it also states which
mechanism reaches which observable, and when.

Two further observables measure what the population *gets* rather than where it
goes, and are the ones a foraging experiment reports:

``uptake``
    the mean nutrient a cell experiences, ``<eta>``, whose rate law
    ``d<eta>/dt = <D_eff lap eta> + kappa <|grad eta|^2>`` splits passive
    smoothing from the chemotactic harvest.  The mobility ``kappa`` is
    therefore a foraging efficiency, not only a transport coefficient;
``discrimination``
    the Kullback--Leibler divergence of one strategy's population from
    another's, quadratic in the response and hence the quantity that fixes how
    many cells must be tracked to tell them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .config import ProjectConfig
from .density import (
    PDF_CASES,
    PositionPDFResult,
    resolve_initial_gaussian_sigma,
)
from .fields import ChemicalField, create_field, disk_area_average
from .numerics import inferred_characteristic_length
from .turning import create_turning_kernel

DEFAULT_BOX_SIZE = 12.0
DEFAULT_BOX_POINTS = 384


@dataclass(frozen=True)
class TransportObservables:
    """Net polarity, centroid, second moment, and outward polarity moment.

    Every array is indexed ``[case, time]`` with a trailing component axis for
    the two vector observables.  The ``*_error`` fields are filled only by the
    particle path, where they are the paired errors of the case-minus-baseline
    response and therefore already carry the common-random-number variance
    reduction.
    """

    method: str
    times: NDArray[np.float64]
    case_numbers: tuple[int, ...]
    net_polarity: NDArray[np.float64]
    centroid: NDArray[np.float64]
    spread: NDArray[np.float64]
    shape: NDArray[np.float64] | None = None
    outward_polarity: NDArray[np.float64] | None = None
    mean_squared_displacement: NDArray[np.float64] | None = None
    uptake: NDArray[np.float64] | None = None
    discrimination: NDArray[np.float64] | None = None
    net_polarity_error: NDArray[np.float64] | None = None
    centroid_error: NDArray[np.float64] | None = None
    spread_error: NDArray[np.float64] | None = None
    shape_error: NDArray[np.float64] | None = None

    def response(self, name: str) -> NDArray[np.float64]:
        """Case-minus-baseline response of ``name``; Case 0 is the baseline."""

        values = getattr(self, name)
        if values is None:
            raise ValueError(f"{self.method} observables do not carry {name}")
        return values - values[0:1]

    def interpolated(self, times: NDArray[np.float64]) -> TransportObservables:
        """Resample onto ``times`` so paths reported on different grids align.

        Every observable is a smooth function of time on a grid that already
        resolves the fastest rate, so linear interpolation is well below the
        discretization error of the solver that produced it.
        """

        target = np.asarray(times, dtype=float)
        if target.min() < self.times.min() or target.max() > self.times.max():
            raise ValueError("cannot extrapolate transport observables")

        def resample(values: NDArray[np.float64] | None) -> NDArray[np.float64] | None:
            if values is None:
                return None
            flat = values.reshape(values.shape[0], values.shape[1], -1)
            moved = np.stack(
                [
                    [
                        np.interp(target, self.times, flat[case, :, column])
                        for column in range(flat.shape[2])
                    ]
                    for case in range(flat.shape[0])
                ]
            )
            return np.moveaxis(moved, -1, 1).reshape(
                (values.shape[0], len(target), *values.shape[2:])
            )

        return replace(
            self,
            times=target,
            net_polarity=resample(self.net_polarity),
            centroid=resample(self.centroid),
            spread=resample(self.spread),
            outward_polarity=resample(self.outward_polarity),
            mean_squared_displacement=resample(self.mean_squared_displacement),
            uptake=resample(self.uptake),
            discrimination=resample(self.discrimination),
            net_polarity_error=resample(self.net_polarity_error),
            centroid_error=resample(self.centroid_error),
            spread_error=resample(self.spread_error),
        )


def polarity_relaxation_rate(config: ProjectConfig) -> float:
    """``gamma1 = D_R0 + lambda0 (1 - P_turn)`` of Eq. (gamma)."""

    turning = create_turning_kernel(config.turning)
    rate = config.kinetics.D_R0 + config.kinetics.lambda0 * (1.0 - turning.P_turn)
    if rate <= 0.0:
        raise ValueError("the macroscopic observables require gamma1 > 0")
    return float(rate)


def effective_diffusivity(config: ProjectConfig) -> float:
    """``D_eff = D_T0 + v0^2 / (2 gamma1)`` of Eq. (macroscopic-effective-diffusivity)."""

    return float(
        config.kinetics.D_T0
        + config.kinetics.v0**2 / (2.0 * polarity_relaxation_rate(config))
    )


def green_amplitudes(
    config: ProjectConfig,
    wavenumber: NDArray[np.float64] | float,
    time: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-cell Green amplitudes ``A_k(t)`` and ``B_k(t)`` of Eq. (dirac-amplitudes).

    Evaluated through complex ``sigma_k``, so the same expression covers the
    relaxational branch ``k < k_c`` and the oscillatory branch above it.
    """

    k = np.asarray(wavenumber, dtype=float)
    gamma1 = polarity_relaxation_rate(config)
    v0 = config.kinetics.v0
    mu = config.kinetics.D_T0 * k**2 + 0.5 * gamma1
    sigma = np.sqrt((gamma1**2 - 2.0 * v0**2 * k**2).astype(complex))
    half = 0.5 * sigma * time
    # Never form exp(-mu t) and cosh(sigma t / 2) separately: each overflows on
    # its own well before their product does, and the product is what is
    # bounded.  Away from the double root the two decaying branches are written
    # out; near it the removable singularity is taken by series.
    slow = np.exp(-(mu - 0.5 * sigma) * time)
    fast = np.exp(-(mu + 0.5 * sigma) * time)
    small = np.abs(half) < 1.0e-6
    safe = np.where(small, 1.0, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosh_term = 0.5 * (slow + fast)
        sinch_term = np.where(
            small,
            np.exp(-mu * time) * time * (1.0 + half**2 / 6.0),
            (slow - fast) / safe,
        )
    density_amplitude = np.real(cosh_term + 0.5 * gamma1 * sinch_term)
    polarity_amplitude = np.real(0.5 * v0 * k * sinch_term)
    return density_amplitude, polarity_amplitude


def baseline_outward_polarity(
    config: ProjectConfig, times: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``Omega^(0)(t) = (v0/gamma1)(1 - e^{-gamma1 t})`` for an isotropic release."""

    gamma1 = polarity_relaxation_rate(config)
    times = np.asarray(times, dtype=float)
    return (config.kinetics.v0 / gamma1) * (1.0 - np.exp(-gamma1 * times))


def baseline_spread(
    config: ProjectConfig, times: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Uniform-background spread ``S^(0)(t)``, ballistic then diffusive.

    ``4 D_T0 t + (2 v0^2/gamma1^2)(gamma1 t - 1 + e^{-gamma1 t})``: the
    integral of ``4 D_T0 + 2 v0 Omega^(0)``.
    """

    gamma1 = polarity_relaxation_rate(config)
    times = np.asarray(times, dtype=float)
    active = (2.0 * config.kinetics.v0**2 / gamma1**2) * (
        gamma1 * times - 1.0 + np.exp(-gamma1 * times)
    )
    return 4.0 * config.kinetics.D_T0 * times + active


@dataclass(frozen=True)
class BackgroundOverlaps:
    """Landscape overlaps of the uniform-background fields.

    Each is an integral of a zeroth-order field against the landscape offset
    ``psi = eta - eta_0``; together they drive every first-order observable.
    ``level`` and ``gradient`` are the density overlaps, ``polarity`` and
    ``alignment`` the polarity overlaps, ``dilation`` and ``outward_level``
    their first radial moments.  The three ``shear_*`` entries are the
    traceless companions of ``dilation``, ``outward_polarity`` and
    ``outward_level``; they drive the shape tensor and vanish identically in a
    uniform background.
    """

    times: NDArray[np.float64]
    level: NDArray[np.float64]  # E   = int rho^(0) psi
    gradient: NDArray[np.float64]  # G_i = int rho^(0) d_i psi
    polarity: NDArray[np.float64]  # P_i = int psi p^(0)_i
    alignment: NDArray[np.float64]  # L   = int p^(0) . grad psi
    dilation: NDArray[np.float64]  # C   = int rho^(0) r . grad psi
    outward_level: NDArray[np.float64]  # K   = int psi (r . p^(0))
    outward_polarity: NDArray[np.float64]  # Omega^(0) = int r . p^(0)
    mass: NDArray[np.float64]
    # Traceless, stored as the two independent components (xx, xy).
    shear: NDArray[np.float64]  # C^_ij = int rho^(0)(x_i d_j psi + x_j d_i psi)^0
    shear_level: NDArray[np.float64]  # K^_ij = int psi (x_i p_j + x_j p_i)^0
    shear_alignment: NDArray[np.float64]  # L^_ij = int (p_i d_j psi + p_j d_i psi)^0


def background_overlaps(
    config: ProjectConfig,
    field: ChemicalField,
    times: NDArray[np.float64],
    *,
    box_size: float = DEFAULT_BOX_SIZE,
    box_points: int = DEFAULT_BOX_POINTS,
) -> BackgroundOverlaps:
    """Overlap integrals of the uniform-background release with the landscape.

    The zeroth-order density and polarity are synthesized spectrally from
    Eq. (dirac-amplitudes) on a periodic box centred on the release point.  The
    box only has to be wide enough that the released pulse never reaches its
    edge; the landscape itself is sampled exactly.
    """

    times = np.asarray(times, dtype=float)
    center = np.asarray(config.initial.position, dtype=float)
    box_size = _commensurate_box(field, box_size)
    spacing = box_size / box_points
    axis = np.fft.fftfreq(box_points) * box_size
    offset_x, offset_y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((offset_x + center[0], offset_y + center[1]), axis=-1)
    eta, gradient = field.value_and_gradient(points)
    psi = np.asarray(eta, dtype=float) - disk_area_average(field, config.domain.radius)
    grad_psi = np.asarray(gradient, dtype=float)
    radial_gradient = offset_x * grad_psi[..., 0] + offset_y * grad_psi[..., 1]

    wave_axis = 2.0 * np.pi * np.fft.fftfreq(box_points, d=spacing)
    wave_x, wave_y = np.meshgrid(wave_axis, wave_axis, indexing="ij")
    wavenumber = np.hypot(wave_x, wave_y)
    positive = wavenumber > 0.0
    unit_x = np.divide(
        wave_x, wavenumber, out=np.zeros_like(wavenumber), where=positive
    )
    unit_y = np.divide(
        wave_y, wavenumber, out=np.zeros_like(wavenumber), where=positive
    )
    window = _release_window(config, field, wavenumber)
    _check_box(config, field, times, psi, box_size, spacing)

    area = spacing**2
    shape = (len(times),)
    level = np.empty(shape)
    alignment = np.empty(shape)
    dilation = np.empty(shape)
    outward_level = np.empty(shape)
    outward = np.empty(shape)
    mass = np.empty(shape)
    gradient_overlap = np.empty((len(times), 2))
    polarity_overlap = np.empty((len(times), 2))
    shear = np.empty((len(times), 2))
    shear_level = np.empty((len(times), 2))
    shear_alignment = np.empty((len(times), 2))
    shear_field = np.stack(
        (
            offset_x * grad_psi[..., 0] - offset_y * grad_psi[..., 1],
            offset_x * grad_psi[..., 1] + offset_y * grad_psi[..., 0],
        ),
        axis=-1,
    )
    for index, time in enumerate(times):
        density_amplitude, polarity_amplitude = green_amplitudes(
            config, wavenumber, float(time)
        )
        density = np.real(np.fft.ifft2(density_amplitude * window)) / area
        weighted = polarity_amplitude * window
        polarity_x = np.real(np.fft.ifft2(-1j * unit_x * weighted)) / area
        polarity_y = np.real(np.fft.ifft2(-1j * unit_y * weighted)) / area
        mass[index] = density.sum() * area
        level[index] = (density * psi).sum() * area
        gradient_overlap[index, 0] = (density * grad_psi[..., 0]).sum() * area
        gradient_overlap[index, 1] = (density * grad_psi[..., 1]).sum() * area
        polarity_overlap[index, 0] = (psi * polarity_x).sum() * area
        polarity_overlap[index, 1] = (psi * polarity_y).sum() * area
        alignment[index] = (
            polarity_x * grad_psi[..., 0] + polarity_y * grad_psi[..., 1]
        ).sum() * area
        dilation[index] = (density * radial_gradient).sum() * area
        radial_polarity = offset_x * polarity_x + offset_y * polarity_y
        outward[index] = radial_polarity.sum() * area
        outward_level[index] = (psi * radial_polarity).sum() * area
        shear[index] = (density[..., None] * shear_field).sum(axis=(0, 1)) * area
        shear_moment = np.stack(
            (
                offset_x * polarity_x - offset_y * polarity_y,
                offset_x * polarity_y + offset_y * polarity_x,
            ),
            axis=-1,
        )
        shear_level[index] = (psi[..., None] * shear_moment).sum(axis=(0, 1)) * area
        shear_alignment[index, 0] = (
            polarity_x * grad_psi[..., 0] - polarity_y * grad_psi[..., 1]
        ).sum() * area
        shear_alignment[index, 1] = (
            polarity_x * grad_psi[..., 1] + polarity_y * grad_psi[..., 0]
        ).sum() * area

    return BackgroundOverlaps(
        times=times,
        level=level,
        gradient=gradient_overlap,
        polarity=polarity_overlap,
        alignment=alignment,
        dilation=dilation,
        outward_level=outward_level,
        outward_polarity=outward,
        mass=mass,
        shear=shear,
        shear_level=shear_level,
        shear_alignment=shear_alignment,
    )


def _commensurate_box(field: ChemicalField, box_size: float) -> float:
    """Snap the periodic box to whole landscape periods where that is possible.

    A single cosine mode is periodic in the synthesis box only if the box holds
    a whole number of its periods along every axis.  Otherwise the box wraps a
    discontinuity, which leaks power all the way to the Nyquist edge and makes
    the spectral synthesis inexact no matter how fine the grid.  For an
    axis-aligned mode the repair is exact and costs a few per cent of the box
    width.  An oblique mode has no square box that works in general, so the
    caller is left to supply one and ``_check_box`` reports the failure.
    """

    wavevector = getattr(field, "wavevector", None)
    if wavevector is None:
        return box_size
    components = [abs(float(value)) for value in wavevector if float(value) != 0.0]
    if len(components) != 1:
        return box_size
    period = 2.0 * np.pi / components[0]
    return max(1, round(box_size / period)) * period


def _check_box(
    config: ProjectConfig,
    field: ChemicalField,
    times: NDArray[np.float64],
    psi: NDArray[np.float64],
    box_size: float,
    spacing: float,
) -> None:
    """Two conditions make the periodic synthesis exact for these integrals.

    The released pulse must not reach the edge of the box, and the landscape
    must be resolved on the grid.  Nothing is required of the pulse's own
    resolution: the overlaps are spectral sums against the landscape, whose
    spectrum is what has to fit.  A mass check would be vacuous, since the
    trapezoid sum of an inverse transform returns its ``k=0`` coefficient
    exactly at any resolution.
    """

    horizon = float(np.max(times))
    reach = config.kinetics.v0 * horizon + np.sqrt(4.0 * config.kinetics.D_T0 * horizon)
    if config.density.initial_position_distribution == "gaussian":
        reach += 4.0 * resolve_initial_gaussian_sigma(
            config.density, inferred_characteristic_length(config, field)
        )
    if 1.5 * reach > 0.45 * box_size:
        raise ValueError(
            f"the release spreads to {reach:.3g} by t={horizon:.3g}, too far for "
            f"a periodic box of size {box_size:.3g}; widen box_size"
        )
    spectrum = np.abs(np.fft.fft2(psi))
    edge = max(
        spectrum[len(spectrum) // 2, :].max(), spectrum[:, len(spectrum) // 2].max()
    )
    if edge > 1.0e-6 * spectrum.max():
        raise ValueError(
            "the nutrient landscape is not resolved on the spectral box "
            f"(Nyquist amplitude {edge / spectrum.max():.3g} of the peak); "
            "refine box_points"
        )


def _release_window(
    config: ProjectConfig,
    field: ChemicalField,
    wavenumber: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fourier transform of the prescribed initial density, normalized to one."""

    distribution = config.density.initial_position_distribution
    if distribution == "point":
        return np.ones_like(wavenumber)
    if distribution != "gaussian":
        raise ValueError(
            "the macroscopic-observable prediction is written for a compact "
            f"central release; initial_position_distribution={distribution!r} "
            "is not one"
        )
    sigma = resolve_initial_gaussian_sigma(
        config.density, inferred_characteristic_length(config, field)
    )
    return np.exp(-0.5 * sigma**2 * wavenumber**2)


def _relaxing_integral(
    times: NDArray[np.float64],
    source: NDArray[np.float64],
    rate: float,
) -> NDArray[np.float64]:
    """Solve ``y' = -rate y + source`` from ``y(0)=0`` with an integrating factor."""

    weight = np.exp(rate * times)
    increments = 0.5 * (weight[1:] * source[1:] + weight[:-1] * source[:-1])
    integral = np.concatenate(([0.0], np.cumsum(increments * np.diff(times))))
    return integral / weight


def _cumulative(
    times: NDArray[np.float64], source: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.concatenate(
        ([0.0], np.cumsum(0.5 * (source[1:] + source[:-1]) * np.diff(times)))
    )


def predicted_observables(
    config: ProjectConfig,
    field: ChemicalField | None = None,
    times: NDArray[np.float64] | None = None,
    *,
    overlaps: BackgroundOverlaps | None = None,
    box_size: float = DEFAULT_BOX_SIZE,
    box_points: int = DEFAULT_BOX_POINTS,
) -> TransportObservables:
    """First-order prediction of the four macroscopic observables.

    Solves, for each case,

    ``dP/dt   = -gamma1 P + (chi - DR_zeta) G/2 - DR_eta Pi``
    ``dOmega/dt = -gamma1 Omega + v0 + v_eta E + v_zeta L - DR_eta K + (chi - DR_zeta) C/2``
    ``dX/dt   = v0 P + v_eta Pi + v_zeta G/2``
    ``dS/dt   = 4 D_T0 + 2 v0 Omega + 2 v_eta K + v_zeta C + 4 D_T_eta E + 4 D_T_zeta L``
    ``dOmega^/dt = -gamma1 Omega^ + v_zeta L^/2 - DR_eta K^ + (chi - DR_zeta) C^/2``
    ``dT/dt   = v0 Omega^ + v_eta K^ + v_zeta C^/2``

    with the overlaps of :func:`background_overlaps` as the sources.  Case 0
    reproduces the uniform-background closed forms exactly.
    """

    field = create_field(config.field) if field is None else field
    if overlaps is None:
        if times is None:
            raise ValueError("predicted_observables needs times or overlaps")
        overlaps = background_overlaps(
            config, field, times, box_size=box_size, box_points=box_points
        )
    times = overlaps.times
    gamma1 = polarity_relaxation_rate(config)
    turning = create_turning_kernel(config.turning)
    v0 = config.kinetics.v0
    baseline_outward = baseline_outward_polarity(config, times)
    baseline = baseline_spread(config, times)

    net_polarity = np.zeros((len(PDF_CASES), len(times), 2))
    centroid = np.zeros_like(net_polarity)
    spread = np.zeros((len(PDF_CASES), len(times)))
    outward = np.zeros_like(spread)
    shape = np.zeros((len(PDF_CASES), len(times), 2))
    for case_index, case_number in enumerate(PDF_CASES):
        response = config.responses.only_case(case_number)
        rotational_level = (
            response.D_R_eta + (1.0 - turning.P_turn) * response.lambda_eta
        )
        rotational_gradient = (
            response.D_R_zeta + (1.0 - turning.P_turn) * response.lambda_zeta
        )
        steering = response.chi_theta - rotational_gradient
        polarity = np.stack(
            [
                _relaxing_integral(
                    times,
                    0.5 * steering * overlaps.gradient[:, axis]
                    - rotational_level * overlaps.polarity[:, axis],
                    gamma1,
                )
                for axis in (0, 1)
            ],
            axis=-1,
        )
        outward_response = _relaxing_integral(
            times,
            response.v_eta * overlaps.level
            + response.v_zeta * overlaps.alignment
            - rotational_level * overlaps.outward_level
            + 0.5 * steering * overlaps.dilation,
            gamma1,
        )
        net_polarity[case_index] = polarity
        outward[case_index] = baseline_outward + outward_response
        centroid[case_index] = np.stack(
            [
                _cumulative(
                    times,
                    v0 * polarity[:, axis]
                    + response.v_eta * overlaps.polarity[:, axis]
                    + 0.5 * response.v_zeta * overlaps.gradient[:, axis],
                )
                for axis in (0, 1)
            ],
            axis=-1,
        )
        shape_polarity = np.stack(
            [
                _relaxing_integral(
                    times,
                    0.5 * response.v_zeta * overlaps.shear_alignment[:, axis]
                    - rotational_level * overlaps.shear_level[:, axis]
                    + 0.5 * steering * overlaps.shear[:, axis],
                    gamma1,
                )
                for axis in (0, 1)
            ],
            axis=-1,
        )
        shape[case_index] = np.stack(
            [
                _cumulative(
                    times,
                    v0 * shape_polarity[:, axis]
                    + response.v_eta * overlaps.shear_level[:, axis]
                    + 0.5 * response.v_zeta * overlaps.shear[:, axis],
                )
                for axis in (0, 1)
            ],
            axis=-1,
        )
        spread[case_index] = baseline + _cumulative(
            times,
            2.0 * v0 * outward_response
            + 2.0 * response.v_eta * overlaps.outward_level
            + response.v_zeta * overlaps.dilation
            + 4.0 * response.D_T_eta * overlaps.level
            + 4.0 * response.D_T_zeta * overlaps.alignment,
        )
    return TransportObservables(
        method="prediction",
        times=times,
        case_numbers=PDF_CASES,
        net_polarity=net_polarity,
        centroid=centroid,
        spread=spread,
        shape=shape,
        outward_polarity=outward,
    )


# ---------------------------------------------------------------------------
# Foraging observables: what the population gets, not where it goes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeTransfer:
    """How much of one landscape mode a release sees over its whole lifetime.

    Every saturated observable is a time integral of the population's overlap
    with the mode, and each such integral is the Laplace transform of the
    uniform-background propagator at zero frequency.  That transform is exact
    at first order in the landscape and elementary:

    ``int_0^inf <cos(q.x)> dt = W(q) / [D_eff(q) q^2]``,
    ``D_eff(q) = D_T0 + (v0)^2 / [2 (D_T0 q^2 + gamma1)]``,

    with ``W(q) = exp(-q^2 sigma_0^2/2)`` the transform of the release itself.
    ``D_eff(q)`` reduces to ``D_eff^(0)`` and ``W`` to one when the landscape is
    coarse compared with a run and with the release; ``derivative`` is
    ``d/dq`` of the same integral, which is what the dilation overlap
    ``C = int rho x.grad(eta)`` needs.
    """

    wavenumber: float
    release_transform: float
    effective_diffusivity: float
    integral: float
    derivative: float
    polarity_rate: float


def mode_transfer(
    config: ProjectConfig, wavenumber: float, release_sigma: float
) -> ModeTransfer:
    """Zero-frequency transfer of one landscape mode; see :class:`ModeTransfer`."""

    q = float(wavenumber)
    if q <= 0.0:
        raise ValueError("a landscape mode needs a positive wavenumber")
    gamma1 = polarity_relaxation_rate(config)
    D_T0 = float(config.kinetics.D_T0)
    v0 = float(config.kinetics.v0)
    sigma = float(release_sigma)

    polarity_rate = gamma1 + D_T0 * q**2
    D_eff_q = D_T0 + v0**2 / (2.0 * polarity_rate)
    transform = np.exp(-0.5 * q**2 * sigma**2)
    integral = transform / (D_eff_q * q**2)
    d_transform = -q * sigma**2 * transform
    d_D_eff = -(v0**2) * D_T0 * q / polarity_rate**2
    derivative = (
        d_transform / (D_eff_q * q**2)
        - transform * d_D_eff / (D_eff_q**2 * q**2)
        - 2.0 * transform / (D_eff_q * q**3)
    )
    return ModeTransfer(
        wavenumber=q,
        release_transform=float(transform),
        effective_diffusivity=float(D_eff_q),
        integral=float(integral),
        derivative=float(derivative),
        polarity_rate=float(polarity_rate),
    )


def _mode_overlap_integrals(
    config: ProjectConfig, transfer: ModeTransfer, amplitude: float
) -> dict[str, float]:
    """Lifetime integrals of the six overlaps on one landscape mode.

    Returned per released cell, for a unit-amplitude mode of the stated parity:
    ``E, C, K, L`` for an even mode ``cos(q.x)`` and ``G, Pi`` for an odd mode
    ``sin(q.x)``, both about the release point.  Each follows from
    Eq. (observable-overlaps) evaluated on the uniform-background fields, whose
    time integral is :class:`ModeTransfer`.
    """

    q = transfer.wavenumber
    v0 = float(config.kinetics.v0)
    D_T0 = float(config.kinetics.D_T0)
    adiabatic = v0 / (2.0 * transfer.polarity_rate)
    level = amplitude * transfer.integral
    dilation = amplitude * q * transfer.derivative
    # The two overlaps that weight the polarity by position need the derivative
    # of the polarity amplitude, not just its adiabatic value: the extra term
    # is the wavenumber dependence of the polarity relaxation rate itself.
    curvature = v0 * D_T0 * q**2 * level / transfer.polarity_rate**2
    return {
        "E": level,
        "C": dilation,
        "K": adiabatic * (2.0 * level + dilation) - curvature,
        "L": -adiabatic * q**2 * level,
        "G": amplitude * q * transfer.integral,
        "Pi": adiabatic * amplitude * q * transfer.integral,
        # Traceless companions along the mode direction.
        "C_shear": dilation,
        "K_shear": adiabatic * dilation - curvature,
        "L_shear": -adiabatic * q**2 * level,
    }


def centroid_saturation(
    config: ProjectConfig,
    case_number: int,
    wavenumber: float,
    amplitude: float,
    release_sigma: float,
) -> float:
    """Total centroid displacement on an odd mode ``amplitude sin(q.x)``.

    The displacement is finite because the mode decays once the pulse has
    spread across a wavelength, and its value is the pulse drift:

        ``X(inf) = amplitude * V * W(q) / [D_eff(q) q]``.

    The result is exact at first order -- no adiabatic elimination is used,
    because integrating ``dP/dt = -gamma1 P + source`` from zero to infinity
    gives ``int P = int source / gamma1`` whatever the transient does.
    """

    transfer = mode_transfer(config, wavenumber, release_sigma)
    overlaps = _mode_overlap_integrals(config, transfer, amplitude)
    responses = config.responses.only_case(case_number)
    turning = create_turning_kernel(config.turning)
    retained = 1.0 - turning.P_turn
    gamma_eta = responses.D_R_eta + retained * responses.lambda_eta
    gamma_zeta = responses.D_R_zeta + retained * responses.lambda_zeta
    gamma1 = polarity_relaxation_rate(config)

    net_polarity = (
        0.5 * (responses.chi_theta - gamma_zeta) * overlaps["G"]
        - gamma_eta * overlaps["Pi"]
    ) / gamma1
    return float(
        config.kinetics.v0 * net_polarity
        + responses.v_eta * overlaps["Pi"]
        + 0.5 * responses.v_zeta * overlaps["G"]
    )


def spread_saturation(
    config: ProjectConfig,
    case_number: int,
    wavenumber: float,
    amplitude: float,
    release_sigma: float,
) -> float:
    """Total change of the second moment on an even mode ``amplitude cos(q.x)``.

    In the coarse-landscape limit ``q sigma_0, q v0/gamma1 << 1`` this reduces
    to ``-4 amplitude kappa / (D_eff^(0) q^2)``: the spread saturates on the
    *mobility*, exactly as the centroid saturates on the drift, so one release
    in a structured landscape returns both coefficients separately.  The
    The law assumes the population eventually samples a landscape of zero mean,
    so that the level overlap ``E`` decays rather than leaving a constant
    residue; on a disk that means the reference level must be the landscape's
    own mean and not, say, a disk average that a partial wavelength has biased.
    The directional flux responses add a term the gradient expansion misses,
    ``-amplitude (v0/gamma1) [v0 v_zeta/gamma1 + 2 D_T_zeta] / D_eff``, which is
    the only place Case 5 appears at all and is the one piece that does not
    scale as ``q^-2``.
    """

    transfer = mode_transfer(config, wavenumber, release_sigma)
    overlaps = _mode_overlap_integrals(config, transfer, amplitude)
    responses = config.responses.only_case(case_number)
    turning = create_turning_kernel(config.turning)
    retained = 1.0 - turning.P_turn
    gamma_eta = responses.D_R_eta + retained * responses.lambda_eta
    gamma_zeta = responses.D_R_zeta + retained * responses.lambda_zeta
    gamma1 = polarity_relaxation_rate(config)

    outward = (
        responses.v_eta * overlaps["E"]
        + responses.v_zeta * overlaps["L"]
        - gamma_eta * overlaps["K"]
        + 0.5 * (responses.chi_theta - gamma_zeta) * overlaps["C"]
    ) / gamma1
    return float(
        2.0 * config.kinetics.v0 * outward
        + 2.0 * responses.v_eta * overlaps["K"]
        + responses.v_zeta * overlaps["C"]
        + 4.0 * responses.D_T_eta * overlaps["E"]
        + 4.0 * responses.D_T_zeta * overlaps["L"]
    )


def shape_saturation(
    config: ProjectConfig,
    case_number: int,
    wavenumber: float,
    amplitude: float,
    release_sigma: float,
) -> float:
    """Total ``T_parallel`` on an even mode ``amplitude cos(q.x)`` along ``x``.

    The traceless second moment saturates on the *pulse drift*,

        ``T(inf) = -2 amplitude V / (D_eff^(0) q^2)``,

    half the magnitude of the spread saturation and with ``V`` in place of
    ``-kappa``.  One even landscape mode therefore returns both coefficients:
    the trace of the second moment gives the mobility, its traceless part gives
    the drift, and no mode of the opposite parity is needed.  The shape has no
    uniform background at all, so the whole measured signal is the response,
    and no translational-diffusivity response reaches it -- a diffusive flux is
    isotropic, so Cases 1 and 5 cannot shape a pulse.
    """

    transfer = mode_transfer(config, wavenumber, release_sigma)
    overlaps = _mode_overlap_integrals(config, transfer, amplitude)
    responses = config.responses.only_case(case_number)
    turning = create_turning_kernel(config.turning)
    retained = 1.0 - turning.P_turn
    gamma_eta = responses.D_R_eta + retained * responses.lambda_eta
    gamma_zeta = responses.D_R_zeta + retained * responses.lambda_zeta
    gamma1 = polarity_relaxation_rate(config)

    shear_polarity = (
        0.5 * responses.v_zeta * overlaps["L_shear"]
        - gamma_eta * overlaps["K_shear"]
        + 0.5 * (responses.chi_theta - gamma_zeta) * overlaps["C_shear"]
    ) / gamma1
    return float(
        config.kinetics.v0 * shear_polarity
        + responses.v_eta * overlaps["K_shear"]
        + 0.5 * responses.v_zeta * overlaps["C_shear"]
    )


def pulse_drift_coefficient(config: ProjectConfig, case_number: int) -> float:
    """``V = kappa + delta D_eff'`` of Eq. (pulse-drift), for one case."""

    from .kinetics import delta_D_eff_coefficient, macroscopic_coefficients

    turning = create_turning_kernel(config.turning)
    responses = config.responses.only_case(case_number)
    _, _, mobility = macroscopic_coefficients(
        config.kinetics, responses, turning.P_turn
    )
    return float(
        mobility + delta_D_eff_coefficient(config.kinetics, responses, turning.P_turn)
    )


def invert_saturations(
    config: ProjectConfig,
    wavenumber: float,
    amplitude: float,
    release_sigma: float,
    shape: float,
    spread: float,
) -> tuple[float, float]:
    """Read ``(V, kappa)`` off one even mode's saturated moments.

    The traceless moment saturates on ``V int C`` and the trace on
    ``4 delta D_eff' int E + 2 V int C``, so

        ``V = T(inf) / int C``,
        ``kappa = V - [dS(inf) - 2 T(inf)] / (4 int E)``,

    with the two lifetime integrals supplied by :func:`mode_transfer`.  This is
    the whole inversion: one landscape mode, one second-moment tensor, both
    coefficients.  The only rule it misreads is Case 5, whose contribution to
    the spread does not scale as ``q^-2`` and therefore shows up as a spurious
    mobility -- which is itself how a two-wavelength experiment detects it,
    Eq. (observable-run-correction).
    """

    transfer = mode_transfer(config, wavenumber, release_sigma)
    level = amplitude * transfer.integral
    dilation = amplitude * transfer.wavenumber * transfer.derivative
    drift = shape / dilation
    return drift, drift - (spread - 2.0 * shape) / (4.0 * level)


def coarse_spread_saturation(
    config: ProjectConfig,
    case_number: int,
    wavenumber: float,
    amplitude: float,
) -> float:
    """``-4 A kappa / (D_eff q^2)`` plus the run-length term, the paper's form."""

    gamma1 = polarity_relaxation_rate(config)
    D_eff = effective_diffusivity(config)
    responses = config.responses.only_case(case_number)
    run = config.kinetics.v0 / gamma1
    mobility = chemotactic_mobility(config, case_number)
    return float(
        -4.0 * amplitude * mobility / (D_eff * wavenumber**2)
        - amplitude * run * (run * responses.v_zeta + 2.0 * responses.D_T_zeta) / D_eff
    )


def chemotactic_mobility(config: ProjectConfig, case_number: int) -> float:
    """``kappa`` of Eq. (drift-mobility) for one case."""

    from .kinetics import macroscopic_coefficients

    turning = create_turning_kernel(config.turning)
    return float(
        macroscopic_coefficients(
            config.kinetics, config.responses.only_case(case_number), turning.P_turn
        )[2]
    )


def even_mode_amplitude(
    field: ChemicalField, center: tuple[float, float] = (0.0, 0.0)
) -> float:
    """Amplitude of the ``cos(q.x)`` component of a periodic field about ``center``.

    Expanding ``A cos(q.r + phase)`` about the release point splits it into an
    even part ``A cos(phase + q.center) cos(q.x)`` and an odd part; the even
    part is the one the spread and the shape respond to, and its signed
    amplitude is what the saturation laws call ``epsilon``.
    """

    offset = float(
        np.dot(np.asarray(field.wavevector, dtype=float), np.asarray(center))
    )
    return float(field.amplitude * np.cos(field.phase + offset))


def chemotactic_peclet(
    config: ProjectConfig, case_number: int, wavenumber: float, amplitude: float
) -> float:
    """``Pe = kappa A / D_eff^(0)``: the one number that decides if a rule matters.

    It is the exponent of the steady accumulation, the fractional density
    contrast a weak landscape produces, and the fractional reduction in the
    time to find a target, all at once.  A response with ``|Pe| << 1`` is
    invisible however large its microscopic sensitivity looks.
    """

    return float(
        chemotactic_mobility(config, case_number)
        * amplitude
        / effective_diffusivity(config)
    )


def accumulation_ratio(
    config: ProjectConfig, case_number: int, wavenumber: float, amplitude: float
) -> float:
    """Steady density contrast ``rho_max/rho_min`` on one landscape mode.

    The zero-flux state of Eq. (long-time-drift-diffusion) is
    ``rho ~ exp(kappa eta / D_eff^(0))``, so a mode of amplitude ``A`` produces
    ``exp(2 kappa A / D_eff^(0)) = exp(2 Pe)``.  This is what a plate assay or a
    microfluidic partition measures, and unlike the centroid displacement it is
    set by ``kappa`` alone -- the diffusivity shift cancels because a
    steady state carries no flux.
    """

    return float(
        np.exp(2.0 * chemotactic_peclet(config, case_number, wavenumber, amplitude))
    )


def uptake_rate(
    config: ProjectConfig,
    case_number: int,
    curvature: NDArray[np.float64],
    squared_gradient: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Harvest law ``d<eta>/dt = D_eff <lap eta> + kappa <|grad eta|^2>``.

    Both averages are over the instantaneous density.  The first term is
    passive: diffusion erodes whatever nutrient contrast the release started
    with, whatever the cell does.  The second is the harvest, and it is
    strictly the chemotactic mobility times the mean-square gradient the
    population is sitting in -- so a rule with ``kappa = 0`` gains nothing at
    this order however much it reshapes the population.

    Valid once alignment is quasi-static (``t gamma1 >> 1``) and the landscape
    is coarse compared with a run, ``q v0 / gamma1 << 1``.
    """

    return effective_diffusivity(config) * np.asarray(curvature, dtype=float) + (
        chemotactic_mobility(config, case_number)
        * np.asarray(squared_gradient, dtype=float)
    )


def uptake_gain(config: ProjectConfig, field: ChemicalField, case_number: int) -> float:
    """Asymptotic uptake gain ``kappa <eta_1^2> / D_eff`` over a null forager.

    The mean-square landscape contrast is taken over the disk.  For a single
    landscape mode of amplitude ``A`` this is ``kappa A^2 / (2 D_eff)``, which
    does not depend on the mode's wavelength: a fine landscape and a coarse one
    are worth the same, they are only harvested at different speeds.
    """

    contrast = _landscape_contrast(field, config.domain.radius)
    return (
        chemotactic_mobility(config, case_number)
        * contrast
        / effective_diffusivity(config)
    )


def _landscape_contrast(
    field: ChemicalField, radius: float, points: int = 601
) -> float:
    """Disk-area variance of the landscape."""

    axis = np.linspace(-radius, radius, points)
    x_grid, y_grid = np.meshgrid(axis, axis)
    inside = x_grid**2 + y_grid**2 <= radius**2
    values = np.asarray(field.value(np.stack((x_grid, y_grid), axis=-1)), dtype=float)
    selected = values[inside]
    return float(np.mean((selected - selected.mean()) ** 2))


def landscape_uptake_time(config: ProjectConfig, wavenumber: float) -> float:
    """``tau_q = 1/(D_eff q^2)``: the time to harvest a landscape mode."""

    if wavenumber <= 0.0:
        raise ValueError("the uptake time needs a positive wavenumber")
    return 1.0 / (effective_diffusivity(config) * wavenumber**2)


FIRST_NEUMANN_DISK_WAVENUMBER = 1.8411837813406593


def domain_mixing_time(config: ProjectConfig) -> float:
    """``1/(D_eff k_1^2)``: the time to explore the whole disk.

    ``k_1`` is the first nonzero Neumann eigenvalue of the disk, so this is the
    slowest relaxation the geometry allows.  Its ratio to
    :func:`landscape_uptake_time` is the separation between local harvesting
    and global mixing.
    """

    return 1.0 / (
        effective_diffusivity(config)
        * (FIRST_NEUMANN_DISK_WAVENUMBER / config.domain.radius) ** 2
    )


def periodic_uptake_prediction(
    config: ProjectConfig,
    field: ChemicalField,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Closed-form ``<eta>(t)`` for every case on a single landscape mode.

    On ``eta = eta_0 + A cos(q.r + phi)`` the harvest law closes on two
    scalars.  The null forager's exposure is the propagator amplitude itself,
    ``w0 = cos(phi) A_q(t) W(q)``, and the gain solves

        ``dw1/dt = -D_eff q^2 w1 + (kappa A q^2 / 2)(1 - w2)``

    from ``w1(0) = 0``, with ``w2 = cos(2 phi) A_2q(t) W(2q)`` the exposure to
    the second harmonic -- the factor that switches the harvest on once the
    pulse has spread across a wavelength.  Returned as ``<eta>``, indexed
    ``[case, time]``.
    """

    from .fields.builtin import PeriodicField

    if not isinstance(field, PeriodicField):
        raise TypeError("the closed uptake prediction needs a single landscape mode")
    times = np.asarray(times, dtype=float)
    wavenumber = field.wavenumber
    amplitude = field.amplitude
    window = _release_window(config, field, np.asarray([wavenumber, 2.0 * wavenumber]))
    release = np.asarray(config.initial.position, dtype=float)
    phase = float(np.dot(release, field.wavevector) + field.phase)

    first = np.array([green_amplitudes(config, wavenumber, float(t))[0] for t in times])
    second = np.array(
        [green_amplitudes(config, 2.0 * wavenumber, float(t))[0] for t in times]
    )
    null = np.cos(phase) * first * window[0]
    harmonic = np.cos(2.0 * phase) * second * window[1]

    diffusivity = effective_diffusivity(config)
    decay = diffusivity * wavenumber**2
    result = np.empty((len(PDF_CASES), len(times)))
    for case_index, case_number in enumerate(PDF_CASES):
        source = (
            0.5
            * chemotactic_mobility(config, case_number)
            * amplitude
            * wavenumber**2
            * (1.0 - harmonic)
        )
        gain = _relaxing_integral(times, source, decay)
        result[case_index] = field.baseline + amplitude * (null + gain)
    return result


def grid_observables(
    times: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    density: NDArray[np.float64],
    polarity_x: NDArray[np.float64],
    polarity_y: NDArray[np.float64],
    case_numbers: tuple[int, ...],
    *,
    method: str,
    center: tuple[float, float] = (0.0, 0.0),
) -> TransportObservables:
    """Reduce gridded ``(rho, p)`` fields to the four macroscopic observables."""

    spacing = float(coordinates[1] - coordinates[0])
    area = spacing**2
    x_grid, y_grid = np.meshgrid(coordinates - center[0], coordinates - center[1])
    mass = density.sum(axis=(-2, -1)) * area
    net_polarity = (
        np.stack(
            (
                polarity_x.sum(axis=(-2, -1)) * area,
                polarity_y.sum(axis=(-2, -1)) * area,
            ),
            axis=-1,
        )
        / mass[..., None]
    )
    centroid = np.stack(
        (
            (density * x_grid).sum(axis=(-2, -1)) * area / mass,
            (density * y_grid).sum(axis=(-2, -1)) * area / mass,
        ),
        axis=-1,
    )
    spread = (density * (x_grid**2 + y_grid**2)).sum(axis=(-2, -1)) * area / mass
    shape = np.stack(
        (
            0.5 * (density * (x_grid**2 - y_grid**2)).sum(axis=(-2, -1)) * area / mass,
            (density * x_grid * y_grid).sum(axis=(-2, -1)) * area / mass,
        ),
        axis=-1,
    )
    outward = (
        (polarity_x * x_grid + polarity_y * y_grid).sum(axis=(-2, -1)) * area / mass
    )
    return TransportObservables(
        method=method,
        times=np.asarray(times, dtype=float),
        case_numbers=case_numbers,
        net_polarity=net_polarity,
        centroid=centroid,
        spread=spread,
        shape=shape,
        outward_polarity=outward,
    )


def finite_volume_observables(
    config: ProjectConfig,
    field: ChemicalField | None = None,
    *,
    show_progress: bool = False,
) -> TransportObservables:
    """Macroscopic observables from the conservative density--polarity solver.

    Reduced from the solver's own finite volumes, so no kernel-density
    smoothing enters the comparison.
    """

    from .moment import integrate_moment_fields

    solution = integrate_moment_fields(config, field, show_progress=show_progress)
    x = solution.points[:, 0] - config.initial.position[0]
    y = solution.points[:, 1] - config.initial.position[1]
    area = solution.spacing**2
    mass = solution.density.sum(axis=-1) * area
    net_polarity = (
        np.stack(
            (
                solution.polarity_x.sum(axis=-1) * area,
                solution.polarity_y.sum(axis=-1) * area,
            ),
            axis=-1,
        )
        / mass[:, None]
    )
    centroid = np.stack(
        (
            (solution.density * x).sum(axis=-1) * area / mass,
            (solution.density * y).sum(axis=-1) * area / mass,
        ),
        axis=-1,
    )
    spread = (solution.density * (x * x + y * y)).sum(axis=-1) * area / mass
    shape = np.stack(
        (
            0.5 * (solution.density * (x * x - y * y)).sum(axis=-1) * area / mass,
            (solution.density * x * y).sum(axis=-1) * area / mass,
        ),
        axis=-1,
    )
    outward = (
        (solution.polarity_x * x + solution.polarity_y * y).sum(axis=-1) * area / mass
    )
    resolved = create_field(config.field) if field is None else field
    landscape = np.asarray(resolved.value(solution.points), dtype=float)
    uptake = (solution.density * landscape).sum(axis=-1) * area / mass
    floor = 1.0e-14 * solution.density[0].max()
    divergence = (
        0.5
        * (
            (solution.density - solution.density[0][None]) ** 2
            / np.maximum(solution.density[0][None], floor)
        ).sum(axis=-1)
        * area
    )
    return TransportObservables(
        method="polar PDE",
        times=solution.times,
        case_numbers=PDF_CASES,
        net_polarity=net_polarity,
        centroid=centroid,
        spread=spread,
        shape=shape,
        outward_polarity=outward,
        uptake=uptake,
        discrimination=divergence,
    )


def particle_observables(
    result: PositionPDFResult,
    release_center: tuple[float, float] = (0.0, 0.0),
) -> TransportObservables:
    """Macroscopic observables from the particle ensemble means.

    Each entry is a plain average over the random walk, so the estimate
    involves no spatial binning and no bandwidth; the reported errors are the
    paired case-minus-baseline errors of the shared random numbers.
    """

    observables = result.microscopic_observables
    if observables is None:
        raise ValueError("the particle result carries no ensemble observables")
    missing = {
        "net_polarity_x",
        "net_polarity_y",
        "centroid_x",
        "centroid_y",
        "squared_radius",
        "shape_xx",
        "shape_xy",
    }.difference(observables.names)
    if missing:
        raise ValueError(
            "the particle result predates the transport observables; "
            "rerun the density computation (missing: "
            + ", ".join(sorted(missing))
            + ")"
        )
    index = {name: position for position, name in enumerate(observables.names)}

    def value(name: str) -> NDArray[np.float64]:
        return observables.values[index[name]]

    def error(name: str) -> NDArray[np.float64]:
        return observables.paired_standard_errors[index[name]]

    # Reported about the release point, as the continuum paths are, so that
    # absolute values line up across paths and not only their responses.
    center = np.asarray(release_center, dtype=float)
    absolute = np.stack((value("centroid_x"), value("centroid_y")), -1)
    centroid = absolute - center
    spread = (
        value("squared_radius")
        - 2.0 * (center * absolute).sum(-1)
        + float(center @ center)
    )
    # The traceless second moment about the release point.  Both components are
    # plain ensemble means once the release offset is removed.
    shape = np.stack(
        (
            0.5 * value("shape_xx")
            - center[0] * value("centroid_x")
            + center[1] * value("centroid_y")
            + 0.5 * (center[0] ** 2 - center[1] ** 2),
            value("shape_xy")
            - center[0] * value("centroid_y")
            - center[1] * value("centroid_x")
            + center[0] * center[1],
        ),
        -1,
    )
    return TransportObservables(
        method="particles",
        times=result.times,
        case_numbers=result.case_numbers,
        net_polarity=np.stack((value("net_polarity_x"), value("net_polarity_y")), -1),
        centroid=centroid,
        spread=spread,
        shape=shape,
        mean_squared_displacement=value("mean_squared_displacement"),
        uptake=value("mean_eta"),
        net_polarity_error=np.stack(
            (error("net_polarity_x"), error("net_polarity_y")), -1
        ),
        centroid_error=np.stack((error("centroid_x"), error("centroid_y")), -1),
        spread_error=error("squared_radius"),
        shape_error=np.stack((0.5 * error("shape_xx"), error("shape_xy")), -1),
    )


# Leading power of t in each case's response, read off the source terms of
# ``predicted_observables``.  Two overlaps start finite -- the release-averaged
# level E and gradient G -- while Pi, L and K each need a polarity the release
# has yet to build and so start at O(t).  The dilation overlap C is the one
# entry that depends on the release: C ~ <|x|^2> lap(eta)/2, finite for a
# compact release of nonzero width and O(t) only in the point limit.
#
# Leading power of t in each rule's response, ordered
# (net polarity, centroid, spread, shape).  SHORT_TIME_EXPONENTS is the
# compact-release counting, which is what an experiment and the simulations
# here see.  POINT_RELEASE_EXPONENTS is the idealized limit; the two differ
# only in the entries driven by the curvature overlap C (Cases 6-9 in the
# spread, and the same rules plus Case 7 in the shape), which a point release
# delays by one power of t.  Rows that share an entry share it because they
# share a microscopic combination, not because the observables are blunt.
# Verified against fitted log-log slopes in the tests.
SHORT_TIME_EXPONENTS: dict[int, tuple[int | None, int | None, int, int | None]] = {
    1: (None, None, 1, None),
    2: (2, 3, 3, 3),
    3: (None, 2, 2, 2),
    4: (2, 3, 3, 3),
    5: (None, None, 2, None),
    6: (1, 2, 2, 2),
    7: (None, 1, 1, 1),
    8: (1, 2, 2, 2),
    9: (1, 2, 2, 2),
}

POINT_RELEASE_EXPONENTS: dict[int, tuple[int | None, int | None, int, int | None]] = {
    **SHORT_TIME_EXPONENTS,
    6: (1, 2, 3, 3),
    7: (None, 1, 2, 2),
    8: (1, 2, 3, 3),
    9: (1, 2, 3, 3),
}

__all__ = [
    "POINT_RELEASE_EXPONENTS",
    "SHORT_TIME_EXPONENTS",
    "BackgroundOverlaps",
    "ModeTransfer",
    "TransportObservables",
    "accumulation_ratio",
    "background_overlaps",
    "baseline_outward_polarity",
    "baseline_spread",
    "centroid_saturation",
    "chemotactic_mobility",
    "chemotactic_peclet",
    "coarse_spread_saturation",
    "domain_mixing_time",
    "effective_diffusivity",
    "even_mode_amplitude",
    "finite_volume_observables",
    "green_amplitudes",
    "grid_observables",
    "invert_saturations",
    "landscape_uptake_time",
    "mode_transfer",
    "particle_observables",
    "periodic_uptake_prediction",
    "polarity_relaxation_rate",
    "predicted_observables",
    "pulse_drift_coefficient",
    "shape_saturation",
    "spread_saturation",
    "uptake_gain",
    "uptake_rate",
]
