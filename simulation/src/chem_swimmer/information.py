"""Projected navigation and the finite-lag information rate of a release.

The macroscopic observables of ``observables.py`` are one-time functionals of
the density and polarity fields.  They cannot say how much a *single* cell's
realized motion reveals about the nutrient geometry it started in, because that
is a two-time, heading-resolved question.  This module supplies the two
observables that answer it, both built from the same random variable so that
all nine response rules are compared on equal terms.

Fix a window ``[t, t+tau]`` and let ``S_t = (r_t, alpha_t)`` be the
nutrient-relative state at its start, with ``alpha = arg grad(eta) - theta`` the
signed heading error, so that ``zeta_par = g cos(alpha)`` and
``zeta_perp = g sin(alpha)`` with ``g = |grad eta|``.  The common observable is
the displacement actually made good along the gradient present at the start,

    Y^tau = (r_{t+tau} - r_t) . grad(eta)(r_t) / g(r_t),

whose conditional mean is a navigation velocity and whose conditional law is a
communication channel:

``projected_velocity``
    ``V_par = E[Y]/tau`` -- distance made good per unit time.  It counts speed,
    persistence, tumbling and steering, not an alignment score.
``information_rate``
    ``i = I(S_t; Y^tau)/tau`` in nats per unit time -- how much knowing where
    and how a cell was pointing reduces uncertainty about where it then went.

Three independent paths produce both, exactly as for the transport moments:

``particle_navigation``
    plain window statistics of the random walk, with a binned estimator for
    the channel that assumes nothing about its shape;
``harmonic_navigation``
    the angular-moment hierarchy of the *backward* generator, carried past
    polar order to ``harmonics`` angular modes and ``moments`` powers of ``Y``.
    The polar-closed field equations cannot produce these observables at all:
    a one-time density does not determine a two-time conditional law;
``navigation_prediction``
    the closed first-order expressions derived in the manuscript.

The prediction is the point of the module.  Writing ``u = gamma1 tau`` and
``V_j = kappa_j + delta D_eff'_j`` for the pulse drift, the whole
mechanism-resolved content of the conditional mean is three numbers,

    V_par(delta_eta, alpha; tau)
        = v0 cos(alpha) W(u)              (baseline, no response)
        + A(tau)                          (drift)
        + B(tau) cos(2 alpha)             (gradient turning and speed)
        + C(tau) delta_eta cos(alpha),    (level turning and speed)

with ``A_j = V_j g Phi_j(u)`` and three lag functions only:

    Phi_7 = 1                             Case 7 needs no polarity,
    Phi_{3,6,8,9} = 1 - W(u) ~ u/2        one polarity relaxation,
    Phi_{2,4} = Psi(u) ~ u^2/6            two,
    V_1 = V_5 = 0                         no drift at any lag.

So the navigation observable reproduces the manuscript's time-domain hierarchy
in the lag domain, and saturates at the pulse drift ``V_j g``.  The second
harmonic ``B`` carries ``chi + D_R^(zeta)`` where the drift carries
``chi - D_R^(zeta)``: resolving the heading therefore separates steering from
heading-dependent turning, the one degeneracy the population fields cannot
break, because that information sits in the nematic harmonic the polar closure
discards.

The information rate is a *quadratic* functional of the response, like the
divergence of ``observables.discrimination``.  In the short-lag limit the
channel is exactly Gaussian, ``Y = tau a(S) + sqrt(2 D_T(S) tau) Z``, with
``a = v(S) cos(alpha)``, giving

    i(0+) = Var[a]/(4 D_T0) + Var[d]/(4 tau),   D_T = D_T0 (1 + d),

so the speed responses (Cases 3 and 7) raise it at order ``tau^0``, the
translational-diffusivity responses (Cases 1 and 5) make it diverge as
``1/tau``, and the five turning responses leave it exactly at the baseline
``(v0)^2 / (8 D_T0)`` however well they navigate.  Navigation performance and
information are therefore dissociated: ``A`` never enters ``i`` at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ProjectConfig
from .density import PDF_CASES, resolve_initial_gaussian_sigma
from .fields import ChemicalField, create_field
from .kinetics import delta_D_eff_coefficient, macroscopic_coefficients
from .numerics import inferred_characteristic_length
from .turning import create_turning_kernel, turning_cosine_moment

DEFAULT_HARMONICS = 8
DEFAULT_MOMENTS = 6
DEFAULT_ALPHA_NODES = 32
DEFAULT_LEVEL_NODES = 16
DEFAULT_INFORMATION_ALPHA_BINS = 24
DEFAULT_INFORMATION_VALUE_BINS = 64
INFORMATION_VALUE_RANGE = 6.0

#: Leading power of the lag in each rule's projected-velocity response, from
#: the lag functions above.  ``None`` marks a rule whose drift vanishes at
#: every lag, not merely at short lag.
LAG_EXPONENTS: dict[int, int | None] = {
    1: None,
    2: 2,
    3: 1,
    4: 2,
    5: None,
    6: 1,
    7: 0,
    8: 1,
    9: 1,
}


@dataclass(frozen=True)
class LagFunctions:
    """The five universal lag shapes, evaluated on one lag grid.

    ``W`` is the running mean of the polarity autocorrelation, ``Phi = 1 - W``
    the drift build-up after one relaxation, ``Psi`` after two, ``Lambda`` the
    level-weighted first harmonic, and ``Theta`` the second-harmonic response,
    the only one that knows the nematic rate ``gamma2``.
    """

    lags: NDArray[np.float64]
    gamma1: float
    gamma2: float
    gamma3: float
    W: NDArray[np.float64]
    W2: NDArray[np.float64]
    Phi: NDArray[np.float64]
    Psi: NDArray[np.float64]
    Lam: NDArray[np.float64]
    Theta: NDArray[np.float64]
    Theta3: NDArray[np.float64]
    Xi: NDArray[np.float64]


def _relaxation_mean(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """``W(u) = (1 - e^-u)/u``, by series where the quotient cancels."""

    small = np.abs(u) < 1.0e-4
    safe = np.where(small, 1.0, u)
    series = 1.0 - u / 2.0 + u**2 / 6.0 - u**3 / 24.0
    return np.where(small, series, np.expm1(-safe) / -safe)


def _double_relaxation(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """``Psi(u) = 1 - [2 - (2+u) e^-u]/u ~ u^2/6``: two relaxations."""

    small = np.abs(u) < 1.0e-3
    safe = np.where(small, 1.0, u)
    series = u**2 / 6.0 - u**3 / 12.0 + u**4 / 40.0
    exact = 1.0 - (2.0 - (2.0 + safe) * np.exp(-safe)) / safe
    return np.where(small, series, exact)


def _level_weight(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """``Lambda(u) = [1 - (1+u) e^-u]/u ~ u/2``."""

    small = np.abs(u) < 1.0e-3
    safe = np.where(small, 1.0, u)
    series = u / 2.0 - u**2 / 3.0 + u**3 / 8.0
    exact = (1.0 - (1.0 + safe) * np.exp(-safe)) / safe
    return np.where(small, series, exact)


def _cross_relaxation(
    fast: float, slow: float, lags: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``[W(fast tau) - W(slow tau)] / (slow - fast)``: one harmonic driving another.

    This is the running mean of the convolution
    ``(e^{-fast tau} - e^{-slow tau})/(slow - fast)``, so it starts at
    ``tau/2`` however far apart the two rates are, and it is the removable
    ``Lambda/rate`` when they coincide.
    """

    separation = slow - fast
    if abs(separation) <= 1.0e-9 * max(abs(fast), abs(slow)):
        return _level_weight(fast * lags) / fast
    return (_relaxation_mean(fast * lags) - _relaxation_mean(slow * lags)) / separation


def lag_functions(
    gamma1: float, gamma2: float, gamma3: float, lags: NDArray[np.float64]
) -> LagFunctions:
    """Evaluate the universal lag shapes for the first three relaxation rates."""

    lags = np.asarray(lags, dtype=float)
    if np.any(lags < 0.0):
        raise ValueError("the observation lag must be nonnegative")
    if min(gamma1, gamma2, gamma3) <= 0.0:
        raise ValueError("the navigation observables require positive relaxation rates")
    u = gamma1 * lags
    W = _relaxation_mean(u)
    W2 = _relaxation_mean(gamma2 * lags)
    Lam = _level_weight(u)
    Theta = _cross_relaxation(gamma2, gamma1, lags)
    return LagFunctions(
        lags=lags,
        gamma1=float(gamma1),
        gamma2=float(gamma2),
        gamma3=float(gamma3),
        W=W,
        W2=W2,
        Phi=1.0 - W,
        Psi=_double_relaxation(u),
        Lam=Lam,
        Theta=Theta,
        Theta3=_cross_relaxation(gamma3, gamma2, lags),
        Xi=Theta - Lam / gamma1,
    )


@dataclass(frozen=True)
class NavigationObservables:
    """Finite-lag navigation and information, indexed ``[case, lag]``.

    ``projected_velocity`` is ``A`` plus the (vanishing) ensemble mean of the
    baseline; ``alignment_harmonic`` is ``B``, the coefficient of
    ``cos(2 alpha)``; ``level_harmonic`` is ``C``, the coefficient of
    ``delta_eta cos(alpha)`` per unit variance of the released level.  The
    ``*_error`` fields are filled by the particle path alone and are the paired
    errors of the case-minus-baseline response.
    """

    method: str
    lags: NDArray[np.float64]
    case_numbers: tuple[int, ...]
    projected_velocity: NDArray[np.float64]
    alignment_harmonic: NDArray[np.float64]
    level_harmonic: NDArray[np.float64]
    nutrient_gain: NDArray[np.float64] | None = None
    information: NDArray[np.float64] | None = None
    conditional_mean: NDArray[np.float64] | None = None
    conditional_variance: NDArray[np.float64] | None = None
    projected_velocity_error: NDArray[np.float64] | None = None
    alignment_harmonic_error: NDArray[np.float64] | None = None
    level_harmonic_error: NDArray[np.float64] | None = None
    information_error: NDArray[np.float64] | None = None

    def response(self, name: str) -> NDArray[np.float64]:
        """Case-minus-baseline response of ``name``; Case 0 is the baseline."""

        values = getattr(self, name)
        if values is None:
            raise ValueError(f"{self.method} navigation does not carry {name}")
        return values - values[0:1]


def _gamma_rates(config: ProjectConfig) -> tuple[float, float, float]:
    """Relaxation of the first three angular harmonics, ``n^2 D_R0 + lambda0(1-P_n)``."""

    turning = create_turning_kernel(config.turning)
    rates = tuple(
        float(
            harmonic**2 * config.kinetics.D_R0
            + config.kinetics.lambda0 * (1.0 - turning_cosine_moment(turning, harmonic))
        )
        for harmonic in (1, 2, 3)
    )
    if min(rates) <= 0.0:
        raise ValueError("the navigation observables require positive relaxation rates")
    return rates


def uniform_gradient(
    config: ProjectConfig, field: ChemicalField
) -> NDArray[np.float64]:
    """Return the constant nutrient gradient, refusing a landscape that varies.

    Every closed form in this module freezes the gradient direction over the
    observation window, which is exact only for a landscape with a constant
    gradient.  Requiring one here keeps the three paths comparing the same
    quantity rather than three different approximations to it.
    """

    probe = (
        np.array(
            [[0.0, 0.0], [0.3, 0.0], [0.0, 0.3], [-0.25, 0.2], [0.15, -0.35]],
            dtype=float,
        )
        * config.domain.radius
    )
    gradients = np.asarray(field.gradient(probe), dtype=float)
    reference = gradients[0]
    scale = float(np.linalg.norm(reference))
    if scale <= 0.0:
        raise ValueError("the navigation observables need a nonzero nutrient gradient")
    if np.max(np.linalg.norm(gradients - reference, axis=-1)) > 1.0e-9 * scale:
        raise ValueError(
            "the finite-lag navigation observables are defined against a frozen "
            "gradient direction; use field.kind = 'linear'"
        )
    return reference


def _response_terms(config: ProjectConfig, case_number: int) -> dict[str, float]:
    """Response sensitivities and the two turning combinations, for one case."""

    turning = create_turning_kernel(config.turning)
    responses = config.responses.only_case(case_number)
    retained = 1.0 - turning.P_turn
    return {
        "v_eta": float(responses.v_eta),
        "v_zeta": float(responses.v_zeta),
        "D_T_eta": float(responses.D_T_eta),
        "D_T_zeta": float(responses.D_T_zeta),
        "chi": float(responses.chi_theta),
        "gamma_eta": float(responses.D_R_eta + retained * responses.lambda_eta),
        "gamma_zeta": float(responses.D_R_zeta + retained * responses.lambda_zeta),
    }


def pulse_drift(config: ProjectConfig, case_number: int) -> float:
    """``V = kappa + delta D_eff'`` of Eq. (pulse-drift), for one case."""

    turning = create_turning_kernel(config.turning)
    responses = config.responses.only_case(case_number)
    _, _, mobility = macroscopic_coefficients(
        config.kinetics, responses, turning.P_turn
    )
    return float(
        mobility + delta_D_eff_coefficient(config.kinetics, responses, turning.P_turn)
    )


def navigation_coefficients(
    config: ProjectConfig,
    case_number: int,
    shapes: LagFunctions,
    gradient_magnitude: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the closed-form ``(A, B, C)`` of one rule on a lag grid.

    ``A`` is the drift ``V_j g Phi_j(u)``, assembled term by term so that a
    configuration with several sensitivities on at once is handled too.
    """

    v0 = float(config.kinetics.v0)
    gamma1 = shapes.gamma1
    terms = _response_terms(config, case_number)
    g = float(gradient_magnitude)

    # Drift, k = 0.  Each bracket is that rule's V_j; each shape is its Phi_j.
    drift = (
        0.5 * terms["v_zeta"] * np.ones_like(shapes.Phi)
        + (v0 * terms["v_eta"] / (2.0 * gamma1)) * shapes.Phi
        + (v0 * (terms["chi"] - terms["gamma_zeta"]) / (2.0 * gamma1)) * shapes.Phi
        - (v0**2 * terms["gamma_eta"] / (2.0 * gamma1**2)) * shapes.Psi
    )

    # Second angular harmonic.  Note chi + gamma_zeta, not chi - gamma_zeta.
    alignment = (
        0.5 * terms["v_zeta"] * shapes.W2
        + (v0 * terms["v_eta"] / 2.0) * shapes.Theta
        - (v0 * (terms["chi"] + terms["gamma_zeta"]) / 2.0) * shapes.Theta
        - (v0**2 * terms["gamma_eta"] / (2.0 * (gamma1 - shapes.gamma2))) * shapes.Xi
    )

    # First harmonic, weighted by the released nutrient level.
    level = terms["v_eta"] * shapes.W - (v0 * terms["gamma_eta"] / gamma1) * shapes.Lam

    return g * drift, g * alignment, level


def _conditional_moments(
    config: ProjectConfig,
    case_number: int,
    shapes: LagFunctions,
    gradient_magnitude: float,
    alpha: NDArray[np.float64],
    level: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Closed-form conditional mean and variance of ``Y``, shape ``[lag, state]``.

    The mean is complete at first order in the response.  The variance keeps the
    exact uniform background, the two translational-diffusivity channels -- the
    only rules whose leading information contribution is a variance modulation
    -- and the speed rescaling of the run term.  The remaining first-order
    corrections to the velocity autocorrelation are dropped; ``harmonic_navigation``
    retains them, which is how their size is measured.
    """

    v0 = float(config.kinetics.v0)
    D_T0 = float(config.kinetics.D_T0)
    gamma1 = shapes.gamma1
    terms = _response_terms(config, case_number)
    g = float(gradient_magnitude)
    tau = shapes.lags[:, None]

    cos1 = np.cos(alpha)[None, :]
    cos2 = np.cos(2.0 * alpha)[None, :]
    level_row = level[None, :]

    drift, alignment, level_gain = navigation_coefficients(
        config, case_number, shapes, gradient_magnitude
    )
    mean = tau * (
        v0 * cos1 * shapes.W[:, None]
        + drift[:, None]
        + alignment[:, None] * cos2
        + level_gain[:, None] * level_row * cos1
    )

    run = (v0**2) * tau * (shapes.Phi[:, None] / gamma1 + shapes.Theta[:, None] * cos2)
    second = 2.0 * D_T0 * tau + run
    # Case 3 rescales the run term with the local speed, v0 -> v0 + v_eta delta_eta.
    second = second + (terms["v_eta"] * level_row / v0) * run
    # Case 7 makes the run itself alignment dependent: a cell heading up-gradient
    # both moves farther and spreads farther, and the two nearly cancel in the
    # conditional variance.  Dropping this term would leave the cancellation out.
    second = second + terms["v_zeta"] * g * v0 * tau * (
        np.cos(alpha)[None, :]
        * (shapes.Phi[:, None] / gamma1 + 0.5 * shapes.Theta[:, None])
        + 0.5 * np.cos(3.0 * alpha)[None, :] * shapes.Theta3[:, None]
    )
    # Case 1: an Ito diffusivity that follows the level shifts the spread by the
    # level the cell released at, plus the level its own run has since gained.
    second = second + 2.0 * terms["D_T_eta"] * (
        level_row * tau + g * v0 * cos1 * tau * shapes.Phi[:, None] / gamma1
    )
    # Case 5: the same, conditional on alignment rather than level.
    second = second + 2.0 * terms["D_T_zeta"] * g * cos1 * tau * shapes.W[:, None]

    # Keep the variance consistent to first order: subtract 2 m0 dm, not m^2,
    # so no second-order piece of the mean leaks into the conditional spread.
    baseline_mean = tau * v0 * cos1 * shapes.W[:, None]
    variance = second - baseline_mean**2 - 2.0 * baseline_mean * (mean - baseline_mean)
    return mean, np.maximum(variance, 1.0e-300)


@dataclass(frozen=True)
class StateGrid:
    """The resolution at which the nutrient-relative state is read.

    A finite-lag information rate is only defined once the state has a stated
    resolution, and the same one must be used for every rule.  The heading error
    is cut into uniform bins and the released level into equal-probability bins
    of its own Gaussian law, so all three paths -- particles, the kinetic
    hierarchy, and the closed prediction -- report the information carried by
    the identical coarse-grained state.
    """

    alpha: NDArray[np.float64]
    level: NDArray[np.float64]
    weights: NDArray[np.float64]
    alpha_bins: int
    level_bins: int
    level_edges: NDArray[np.float64]
    level_sigma: float

    @property
    def size(self) -> int:
        return int(self.alpha.size)

    def assign(
        self, alpha: NDArray[np.float64], level: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        """Flat cell index of each sampled ``(alpha, delta_eta)`` pair."""

        wrapped = np.mod(np.asarray(alpha, dtype=float), 2.0 * np.pi)
        angular = np.minimum(
            (wrapped * self.alpha_bins / (2.0 * np.pi)).astype(np.int64),
            self.alpha_bins - 1,
        )
        radial = np.searchsorted(self.level_edges, np.asarray(level, dtype=float))
        radial = np.clip(radial - 1, 0, self.level_bins - 1)
        return angular * self.level_bins + radial


def _equal_probability_normal_bins(
    bins: int, sigma: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Conditional means and edges of ``bins`` equal-probability normal cells."""

    if bins < 1:
        raise ValueError("the level state needs at least one bin")
    if sigma <= 0.0 or bins == 1:
        return np.zeros(1), np.array([-np.inf, np.inf])
    from math import erf, sqrt

    quantiles = np.arange(1, bins) / bins

    # Invert the standard normal CDF by bisection: no SciPy dependency, and the
    # bin edges only need to be exact to the width of a histogram cell.
    def cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    edges = [-np.inf]
    for target in quantiles:
        low, high = -12.0, 12.0
        for _ in range(200):
            middle = 0.5 * (low + high)
            if cdf(middle) < target:
                low = middle
            else:
                high = middle
        edges.append(0.5 * (low + high))
    edges.append(np.inf)
    edges = np.asarray(edges, dtype=float)
    density = np.exp(-0.5 * np.clip(edges, -40.0, 40.0) ** 2) / np.sqrt(2.0 * np.pi)
    density[0] = 0.0
    density[-1] = 0.0
    means = (density[:-1] - density[1:]) * bins
    return sigma * means, sigma * edges


def navigation_state_grid(
    config: ProjectConfig,
    field: ChemicalField,
    gradient_magnitude: float,
    *,
    alpha_bins: int = DEFAULT_ALPHA_NODES,
    level_bins: int = DEFAULT_LEVEL_NODES,
) -> StateGrid:
    """Common ``(alpha, delta_eta)`` reading grid for all three paths."""

    sigma = _release_level_sigma(config, field, gradient_magnitude)
    if sigma <= 0.0:
        level_bins = 1
    level_nodes, level_edges = _equal_probability_normal_bins(level_bins, sigma)
    alpha_nodes = 2.0 * np.pi * (np.arange(alpha_bins) + 0.5) / alpha_bins
    alpha_grid, level_grid = np.meshgrid(alpha_nodes, level_nodes, indexing="ij")
    weights = np.full(alpha_grid.size, 1.0 / alpha_grid.size)
    return StateGrid(
        alpha=alpha_grid.ravel(),
        level=level_grid.ravel(),
        weights=weights,
        alpha_bins=int(alpha_bins),
        level_bins=int(level_bins),
        level_edges=level_edges,
        level_sigma=float(sigma),
    )


def _release_level_sigma(
    config: ProjectConfig, field: ChemicalField, gradient_magnitude: float
) -> float:
    """Standard deviation of ``delta_eta`` over the released ensemble.

    For a constant gradient the released level is ``g`` times the displacement
    along it, so an isotropic Gaussian release of width ``sigma_0`` gives
    ``delta_eta ~ N(0, (g sigma_0)^2)``.
    """

    if config.density.initial_position_distribution != "gaussian":
        return 0.0
    sigma = resolve_initial_gaussian_sigma(
        config.density, inferred_characteristic_length(config, field)
    )
    return float(gradient_magnitude * sigma)


def gaussian_channel_information(
    mean: NDArray[np.float64],
    variance: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    value_nodes: int = 513,
    value_range: float = 9.0,
) -> NDArray[np.float64]:
    """Mutual information of a Gaussian channel with state-dependent moments.

    ``mean`` and ``variance`` are ``[lag, state]``; ``weights`` is the state
    law.  The marginal is a Gaussian mixture, so the outer integral is done by
    quadrature on a grid scaled to the mixture's own spread.  No small
    signal-to-noise expansion is made, which matters as soon as the run
    dominates the step.
    """

    mean = np.asarray(mean, dtype=float)
    variance = np.asarray(variance, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    centre = np.sum(weights * mean, axis=-1, keepdims=True)
    spread = np.sqrt(
        np.sum(weights * (variance + (mean - centre) ** 2), axis=-1, keepdims=True)
    )
    grid = np.linspace(-value_range, value_range, value_nodes)
    y = centre + spread * grid[None, :]

    # p(y | s) for every lag, state and quadrature point.
    exponent = -0.5 * (y[:, None, :] - mean[..., None]) ** 2 / variance[..., None]
    density = np.exp(exponent) / np.sqrt(2.0 * np.pi * variance[..., None])
    marginal = np.sum(weights[None, :, None] * density, axis=1)

    safe_marginal = np.maximum(marginal, 1.0e-300)
    safe_density = np.maximum(density, 1.0e-300)
    integrand = np.sum(
        weights[None, :, None]
        * density
        * np.log(safe_density / safe_marginal[:, None, :]),
        axis=1,
    )
    step = (y[:, 1] - y[:, 0])[:, None]
    return np.maximum(np.sum(integrand, axis=-1) * step[:, 0], 0.0)


def navigation_prediction(
    config: ProjectConfig,
    lags: NDArray[np.float64],
    *,
    field: ChemicalField | None = None,
    case_numbers: tuple[int, ...] = PDF_CASES,
    alpha_nodes: int = DEFAULT_ALPHA_NODES,
    level_nodes: int = DEFAULT_LEVEL_NODES,
) -> NavigationObservables:
    """Closed first-order prediction of the finite-lag navigation channel."""

    field = create_field(config.field) if field is None else field
    gradient = uniform_gradient(config, field)
    g = float(np.linalg.norm(gradient))
    gamma1, gamma2, gamma3 = _gamma_rates(config)
    lags = np.asarray(lags, dtype=float)
    shapes = lag_functions(gamma1, gamma2, gamma3, lags)
    grid = navigation_state_grid(
        config, field, g, alpha_bins=alpha_nodes, level_bins=level_nodes
    )

    means = np.zeros((len(case_numbers), len(lags), grid.size))
    variances = np.zeros_like(means)
    for row, case_number in enumerate(case_numbers):
        means[row], variances[row] = _conditional_moments(
            config, case_number, shapes, g, grid.alpha, grid.level
        )
    return _reduce_conditional_moments(
        "prediction", lags, tuple(case_numbers), means, variances, grid, g
    )


def short_lag_information(
    config: ProjectConfig,
    case_number: int,
    *,
    field: ChemicalField | None = None,
) -> tuple[float, float]:
    """Return ``(rate, divergent)``: the ``tau -> 0`` information rate.

    The channel is exactly Gaussian at short lag, ``Y = tau a(S) +
    sqrt(2 D_T(S) tau) Z`` with ``a = v(S) cos(alpha)``, so

        i(0+) = Var[a] / (4 D_T0)  +  Var[d] / (4 tau),

    with ``D_T = D_T0 (1 + d)``.  The first return value is the finite part; the
    second is the coefficient of ``1/tau``, nonzero only for the two
    translational-diffusivity rules, whose channel is a scale mixture and whose
    information does not vanish with the window.
    """

    field = create_field(config.field) if field is None else field
    gradient = uniform_gradient(config, field)
    g = float(np.linalg.norm(gradient))
    v0 = float(config.kinetics.v0)
    D_T0 = float(config.kinetics.D_T0)
    terms = _response_terms(config, case_number)
    level_sigma = _release_level_sigma(config, field, g)

    # a = (v0 + v_eta delta_eta + v_zeta g cos alpha) cos alpha, with alpha
    # uniform and delta_eta zero-mean and independent of it.
    mean_a = 0.5 * terms["v_zeta"] * g
    speed_square = v0**2 + terms["v_eta"] ** 2 * level_sigma**2
    second_a = 0.5 * speed_square + 0.375 * (terms["v_zeta"] * g) ** 2
    finite = (second_a - mean_a**2) / (4.0 * D_T0)

    # d = (D_T_eta delta_eta + D_T_zeta g cos alpha)/D_T0, zero mean.
    variance_d = (
        terms["D_T_eta"] ** 2 * level_sigma**2 + 0.5 * (terms["D_T_zeta"] * g) ** 2
    ) / D_T0**2
    return float(finite), float(variance_d / 4.0)


def _harmonic_damping(
    config: ProjectConfig, harmonics: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotational and tumble damping of every angular harmonic ``|k| <= K``.

    Rotational diffusion damps harmonic ``k`` as ``k^2``; a tumble damps it by
    ``1 - E[cos(k psi)]``.  Their sum at ``k = 1`` is ``gamma1`` and at ``k = 2``
    is ``gamma2``; the higher harmonics are what the polar closure discards.
    """

    kernel = create_turning_kernel(config.turning)
    k = np.arange(-harmonics, harmonics + 1, dtype=float)
    retained = np.array(
        [1.0 - turning_cosine_moment(kernel, abs(int(value))) for value in k]
    )
    return k**2, retained


def _moment_rhs(
    c: NDArray[np.complex128],
    k: NDArray[np.float64],
    k_squared: NDArray[np.float64],
    retained: NDArray[np.float64],
    coefficients: dict[str, NDArray[np.float64] | float],
) -> NDArray[np.complex128]:
    """Right-hand side of ``d/dtau E[Y^n e^{ik psi}]``; ``c`` is ``[n, k, state]``."""

    def up(values: NDArray[np.complex128], shift: int) -> NDArray[np.complex128]:
        out = np.zeros_like(values)
        out[:-shift] = values[shift:]
        return out

    def down(values: NDArray[np.complex128], shift: int) -> NDArray[np.complex128]:
        out = np.zeros_like(values)
        out[shift:] = values[:-shift]
        return out

    speed = coefficients["speed"]
    speed_level = coefficients["v_eta_g"]
    speed_zeta = coefficients["v_zeta_g"]
    diffusivity = coefficients["D_T"]
    diffusivity_level = coefficients["D_T_eta_g"]
    diffusivity_zeta = coefficients["D_T_zeta_g"]
    rotational = coefficients["D_R"]
    rotational_level = coefficients["D_R_eta_g"]
    rotational_zeta = coefficients["D_R_zeta_g"]
    tumble = coefficients["lambda"]
    tumble_level = coefficients["lambda_eta_g"]
    tumble_zeta = coefficients["lambda_zeta_g"]
    steering = coefficients["chi_g"]

    kx = k[:, None]
    kk = k_squared[:, None]
    damp = retained[:, None]

    out = np.zeros_like(c)
    moments = c.shape[0]
    for n in range(moments):
        row = c[n]
        neighbour = 0.5 * (up(row, 1) + down(row, 1))
        term = -kx * steering * 0.5 * (up(row, 1) - down(row, 1))
        term = term - kk * (rotational * row + rotational_zeta * neighbour)
        term = term - damp * (tumble * row + tumble_zeta * neighbour)
        if n >= 1:
            previous = c[n - 1]
            term = term + n * (
                speed * 0.5 * (up(previous, 1) + down(previous, 1))
                + speed_level * neighbour
                + speed_zeta
                * 0.25
                * (up(previous, 2) + 2.0 * previous + down(previous, 2))
            )
        if n >= 2:
            older = c[n - 2]
            term = term + n * (n - 1) * (
                diffusivity * older
                + diffusivity_level * c[n - 1]
                + diffusivity_zeta * 0.5 * (up(older, 1) + down(older, 1))
            )
        if n + 1 < moments:
            # A level-dependent turning rate couples upward in the moment index,
            # because the level a cell has reached is itself a power of Y.
            higher = c[n + 1]
            term = term - kk * rotational_level * higher - damp * tumble_level * higher
        out[n] = term
    return out


def harmonic_navigation(
    config: ProjectConfig,
    lags: NDArray[np.float64],
    *,
    field: ChemicalField | None = None,
    case_numbers: tuple[int, ...] = PDF_CASES,
    harmonics: int = DEFAULT_HARMONICS,
    moments: int = DEFAULT_MOMENTS,
    alpha_nodes: int = DEFAULT_ALPHA_NODES,
    level_nodes: int = DEFAULT_LEVEL_NODES,
    stability: float = 0.25,
) -> NavigationObservables:
    """Integrate the angular-moment hierarchy of the backward generator.

    For a landscape with a constant gradient the pair ``(Y, psi)`` is Markov and
    the level a cell has reached is ``delta_eta_0 + g Y``, so the joint moments
    ``E[Y^n e^{ik psi}]`` close: the hierarchy is banded in ``k`` and triangular
    in ``n`` except for the level-dependent turning rates, which couple one
    moment upward and are truncated at ``moments``.  Nothing is expanded in the
    response amplitude, and the angular hierarchy is carried to ``harmonics``
    modes, well past the polar order at which the field equations close.  This
    is therefore an independent solution of the same kinetic problem, not a
    reading of the same closure.
    """

    field = create_field(config.field) if field is None else field
    gradient = uniform_gradient(config, field)
    g = float(np.linalg.norm(gradient))
    lags = np.asarray(lags, dtype=float)
    if lags[0] != 0.0 or np.any(np.diff(lags) <= 0.0):
        raise ValueError("harmonic_navigation needs an increasing lag grid from zero")
    if moments < 3:
        raise ValueError("the conditional variance needs at least three moments")

    grid = navigation_state_grid(
        config, field, g, alpha_bins=alpha_nodes, level_bins=level_nodes
    )
    alpha_flat = grid.alpha
    level_flat = grid.level

    k = np.arange(-harmonics, harmonics + 1, dtype=float)
    k_squared, retained = _harmonic_damping(config, harmonics)
    kinetics = config.kinetics
    max_damping = float(
        np.max(k_squared) * kinetics.D_R0 + kinetics.lambda0 * float(np.max(retained))
    )
    step_target = stability / max(max_damping, np.finfo(float).tiny)

    means = np.zeros((len(case_numbers), len(lags), alpha_flat.size))
    variances = np.zeros_like(means)
    for row, case_number in enumerate(case_numbers):
        responses = config.responses.only_case(case_number)
        coefficients: dict[str, NDArray[np.float64] | float] = {
            "speed": kinetics.v0 + responses.v_eta * level_flat,
            "v_eta_g": responses.v_eta * g,
            "v_zeta_g": responses.v_zeta * g,
            "D_T": kinetics.D_T0 + responses.D_T_eta * level_flat,
            "D_T_eta_g": responses.D_T_eta * g,
            "D_T_zeta_g": responses.D_T_zeta * g,
            "D_R": kinetics.D_R0 + responses.D_R_eta * level_flat,
            "D_R_eta_g": responses.D_R_eta * g,
            "D_R_zeta_g": responses.D_R_zeta * g,
            "lambda": kinetics.lambda0 + responses.lambda_eta * level_flat,
            "lambda_eta_g": responses.lambda_eta * g,
            "lambda_zeta_g": responses.lambda_zeta * g,
            "chi_g": responses.chi_theta * g,
        }
        state = np.zeros((moments, k.size, alpha_flat.size), dtype=complex)
        state[0] = np.exp(1j * k[:, None] * alpha_flat[None, :])
        means[row, 0] = 0.0
        variances[row, 0] = 0.0
        zero_index = harmonics
        for index in range(1, len(lags)):
            span = float(lags[index] - lags[index - 1])
            substeps = max(1, int(np.ceil(span / step_target)))
            dt = span / substeps
            for _ in range(substeps):
                k1 = _moment_rhs(state, k, k_squared, retained, coefficients)
                k2 = _moment_rhs(
                    state + 0.5 * dt * k1, k, k_squared, retained, coefficients
                )
                k3 = _moment_rhs(
                    state + 0.5 * dt * k2, k, k_squared, retained, coefficients
                )
                k4 = _moment_rhs(state + dt * k3, k, k_squared, retained, coefficients)
                state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            mean = state[1, zero_index].real
            second = state[2, zero_index].real
            means[row, index] = mean
            variances[row, index] = np.maximum(second - mean**2, 1.0e-300)

    return _reduce_conditional_moments(
        "kinetic", lags, tuple(case_numbers), means, variances, grid, g
    )


def _reduce_conditional_moments(
    method: str,
    lags: NDArray[np.float64],
    case_numbers: tuple[int, ...],
    means: NDArray[np.float64],
    variances: NDArray[np.float64],
    grid: StateGrid,
    gradient_magnitude: float,
    *,
    errors: dict[str, NDArray[np.float64]] | None = None,
) -> NavigationObservables:
    """Project conditional moments onto the three reported harmonics.

    ``A`` is the state average, ``B`` twice the projection on ``cos(2 alpha)``
    and ``C`` twice the projection on ``delta_eta cos(alpha)`` per unit level
    variance -- each an ordinary ensemble mean, so a particle run estimates them
    without binning the projections at all.
    """

    weights = grid.weights / np.sum(grid.weights)
    positive = lags > 0.0
    lag_column = np.where(positive, lags, 1.0)
    level_variance = float(np.sum(weights * grid.level**2))

    drift = np.einsum("s,cts->ct", weights, means) / lag_column
    alignment = (
        2.0
        * np.einsum("s,cts->ct", weights * np.cos(2.0 * grid.alpha), means)
        / lag_column
    )
    if level_variance > 0.0:
        level_harmonic = (
            2.0
            * np.einsum("s,cts->ct", weights * grid.level * np.cos(grid.alpha), means)
            / (lag_column * level_variance)
        )
    else:
        level_harmonic = np.zeros_like(drift)

    information = np.empty_like(drift)
    for row in range(len(case_numbers)):
        information[row, positive] = (
            gaussian_channel_information(
                means[row][positive], variances[row][positive], weights
            )
            / lags[positive]
        )
    for values in (drift, alignment, level_harmonic, information):
        values[:, ~positive] = np.nan

    errors = errors or {}
    return NavigationObservables(
        method=method,
        lags=lags,
        case_numbers=case_numbers,
        projected_velocity=drift,
        alignment_harmonic=alignment,
        level_harmonic=level_harmonic,
        nutrient_gain=gradient_magnitude * drift,
        information=information,
        conditional_mean=means,
        conditional_variance=variances,
        projected_velocity_error=errors.get("projected_velocity"),
        alignment_harmonic_error=errors.get("alignment_harmonic"),
        level_harmonic_error=errors.get("level_harmonic"),
    )


@dataclass
class _WindowAccumulator:
    """Running sums of the window statistics, held per case and lag."""

    displacement: NDArray[np.float64]
    alignment: NDArray[np.float64]
    level: NDArray[np.float64]
    paired_squares: NDArray[np.float64]
    paired_alignment_squares: NDArray[np.float64]
    paired_level_squares: NDArray[np.float64]
    cell_sum: NDArray[np.float64]
    cell_square_sum: NDArray[np.float64]
    cell_count: NDArray[np.float64]
    half_cell_sum: NDArray[np.float64]
    half_cell_square_sum: NDArray[np.float64]
    half_cell_count: NDArray[np.float64]


def _new_accumulator(
    cases: int, lags: int, information_lags: int, cells: int
) -> _WindowAccumulator:
    shape = (cases, lags)
    return _WindowAccumulator(
        displacement=np.zeros(shape),
        alignment=np.zeros(shape),
        level=np.zeros(shape),
        paired_squares=np.zeros(shape),
        paired_alignment_squares=np.zeros(shape),
        paired_level_squares=np.zeros(shape),
        cell_sum=np.zeros((cases, information_lags, cells)),
        cell_square_sum=np.zeros((cases, information_lags, cells)),
        cell_count=np.zeros(cells),
        half_cell_sum=np.zeros((cases, information_lags, cells)),
        half_cell_square_sum=np.zeros((cases, information_lags, cells)),
        half_cell_count=np.zeros(cells),
    )


def _accumulate_windows(
    accumulator: _WindowAccumulator,
    projected: NDArray[np.float64],
    alpha: NDArray[np.float64],
    level: NDArray[np.float64],
    cells: NDArray[np.int64],
    information_slice: NDArray[np.int64],
    cell_count: int,
    half_stop: int,
) -> None:
    """Reduce one batch of windows; ``projected`` is ``[particle, case, lag]``.

    ``half_stop`` marks how many of this batch's windows belong to the first
    independent half of the ensemble.  That half is accumulated separately so
    the plug-in information estimator, whose bias falls as ``1/M``, can be
    extrapolated to infinite sample size.
    """

    cos2 = np.cos(2.0 * alpha)[:, None, None]
    weight = (level * np.cos(alpha))[:, None, None]

    accumulator.displacement += np.sum(projected, axis=0)
    accumulator.alignment += np.sum(projected * cos2, axis=0)
    accumulator.level += np.sum(projected * weight, axis=0)

    paired = projected - projected[:, 0:1, :]
    accumulator.paired_squares += np.sum(paired * paired, axis=0)
    accumulator.paired_alignment_squares += np.sum((paired * cos2) ** 2, axis=0)
    accumulator.paired_level_squares += np.sum((paired * weight) ** 2, axis=0)

    reduced = projected[:, :, information_slice]
    cases, lags = reduced.shape[1], reduced.shape[2]
    columns = np.arange(cases * lags)[None, :]
    length = cell_count * cases * lags

    def scatter(
        target: NDArray[np.float64],
        values: NDArray[np.float64],
        index: NDArray[np.int64],
    ) -> None:
        flat = (index[:, None] * (cases * lags) + columns).ravel()
        totals = np.bincount(flat, weights=values.reshape(-1), minlength=length)
        target += totals.reshape(cell_count, cases, lags).transpose(1, 2, 0)

    scatter(accumulator.cell_sum, reduced, cells)
    scatter(accumulator.cell_square_sum, reduced * reduced, cells)
    accumulator.cell_count += np.bincount(cells, minlength=cell_count)
    if half_stop > 0:
        head = reduced[:half_stop]
        scatter(accumulator.half_cell_sum, head, cells[:half_stop])
        scatter(accumulator.half_cell_square_sum, head * head, cells[:half_stop])
        accumulator.half_cell_count += np.bincount(
            cells[:half_stop], minlength=cell_count
        )


def particle_navigation(
    config: ProjectConfig,
    *,
    field: ChemicalField | None = None,
    ensemble_size: int | None = None,
    case_numbers: tuple[int, ...] = PDF_CASES,
    alpha_bins: int = DEFAULT_ALPHA_NODES,
    level_bins: int = DEFAULT_LEVEL_NODES,
    information_lags: int = 26,
    show_progress: bool = False,
) -> NavigationObservables:
    """Measure the finite-lag channel from windows of the random walk.

    Every window starts at the release, so the heading is isotropic and the
    reported state law is the prescribed one.  The three harmonics are plain
    ensemble means -- no binning enters them -- and their errors are paired
    across cases, so the common random numbers cancel almost all of the
    variance in the case-minus-baseline response.  The conditional mean and
    variance that feed the channel are read on the common state grid, so the
    information is compared at the same state resolution as the two theory
    paths.
    """

    from .accelerated import numba_available_for, state_batch_numba
    from .density import _density_config, _saved_times
    from .numerics import resolve_numerics

    field = create_field(config.field) if field is None else field
    gradient = uniform_gradient(config, field)
    g = float(np.linalg.norm(gradient))
    direction = gradient / g
    config, _ = resolve_numerics(config, field, case_numbers)
    run_config = _density_config(config)
    configured = run_config.density.ensemble_size
    count = int(configured if ensemble_size is None else ensemble_size)
    if isinstance(configured, str) and ensemble_size is None:
        raise ValueError(
            "the navigation observables need an explicit density.ensemble_size"
        )
    if not numba_available_for(field, create_turning_kernel(run_config.turning)):
        raise ValueError(
            "the finite-lag navigation observables need the native engine; "
            "use a built-in chemical field and turning kernel"
        )

    lags = _saved_times(run_config)
    grid = navigation_state_grid(
        config, field, g, alpha_bins=alpha_bins, level_bins=level_bins
    )
    # The three harmonics are plain means and need no occupancy floor, but the
    # conditional moments are read cell by cell, so the ensemble must fill the
    # state grid several hundred times over before their variances are usable.
    if count < 200 * grid.size:
        raise ValueError(
            f"M={count} leaves fewer than 200 windows in each of the "
            f"{grid.size} state cells; raise density.ensemble_size or lower "
            "the state resolution"
        )
    stride = max(1, int(np.ceil((len(lags) - 1) / max(1, information_lags - 1))))
    information_slice = np.arange(0, len(lags), stride, dtype=np.int64)
    if information_slice[-1] != len(lags) - 1:
        information_slice = np.append(information_slice, len(lags) - 1)

    accumulator = _new_accumulator(
        len(case_numbers), len(lags), len(information_slice), grid.size
    )
    reference_level = float(field.value(np.asarray(config.initial.position)))
    from .fields import disk_area_average

    eta0 = disk_area_average(field, config.domain.radius)
    del reference_level

    seed_sequence = np.random.SeedSequence(run_config.simulation.seed)
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(count)
    ]
    from .progress import progress_bar

    bar = progress_bar(
        total=count,
        description="Finite-lag navigation windows",
        unit="particle",
        enabled=show_progress,
    )
    batch_size = min(run_config.density.batch_size, count)
    for start in range(0, count, batch_size):
        batch_seeds = seeds[start : start + batch_size]
        positions, headings, _ = state_batch_numba(
            run_config,
            field,
            batch_seeds,
            isotropic_headings=(
                run_config.density.initial_heading_distribution == "isotropic"
            ),
            uniform_positions=(
                run_config.density.initial_position_distribution == "uniform"
            ),
            gaussian_positions=(
                run_config.density.initial_position_distribution == "gaussian"
            ),
            gaussian_sigma=resolve_initial_gaussian_sigma(
                run_config.density,
                inferred_characteristic_length(run_config, field),
            ),
            case_numbers=case_numbers,
        )
        release = positions[:, 0, 0, :] * config.domain.radius
        level = np.asarray(field.value(release), dtype=float) - eta0
        alpha = np.arctan2(direction[1], direction[0]) - headings[:, 0, 0]
        projected = (
            (positions - positions[:, :, 0:1, :]) @ direction
        ) * config.domain.radius
        cells = grid.assign(alpha, level)
        _accumulate_windows(
            accumulator,
            projected,
            alpha,
            level,
            cells,
            information_slice,
            grid.size,
            min(len(batch_seeds), max(0, count // 2 - start)),
        )
        bar.update(len(batch_seeds))
    bar.close()

    return _finalize_particle_navigation(
        accumulator, count, lags, information_slice, tuple(case_numbers), grid, g
    )


def _paired_error(
    sums: NDArray[np.float64], squares: NDArray[np.float64], count: int
) -> NDArray[np.float64]:
    """Standard error of the case-minus-baseline mean of a paired sample."""

    means = sums / count
    differences = means - means[0:1]
    centred = np.maximum(squares - count * differences**2, 0.0)
    variance = centred / max(count - 1, 1)
    return np.sqrt(variance / count)


def _finalize_particle_navigation(
    accumulator: _WindowAccumulator,
    count: int,
    lags: NDArray[np.float64],
    information_slice: NDArray[np.int64],
    case_numbers: tuple[int, ...],
    grid: StateGrid,
    gradient_magnitude: float,
) -> NavigationObservables:
    positive = lags > 0.0
    lag_column = np.where(positive, lags, 1.0)
    level_variance = float(np.sum(grid.weights * grid.level**2) / np.sum(grid.weights))

    drift = accumulator.displacement / (count * lag_column)
    alignment = 2.0 * accumulator.alignment / (count * lag_column)
    level = (
        2.0 * accumulator.level / (count * lag_column * level_variance)
        if level_variance > 0.0
        else np.zeros_like(drift)
    )
    drift_error = (
        _paired_error(accumulator.displacement, accumulator.paired_squares, count)
        / lag_column
    )
    alignment_error = (
        2.0
        * _paired_error(
            accumulator.alignment, accumulator.paired_alignment_squares, count
        )
        / lag_column
    )
    level_error = (
        2.0
        * _paired_error(accumulator.level, accumulator.paired_level_squares, count)
        / (lag_column * max(level_variance, np.finfo(float).tiny))
    )

    reduced_lags = lags[information_slice]
    live = reduced_lags > 0.0

    def plug_in(
        sums: NDArray[np.float64],
        squares: NDArray[np.float64],
        counts: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        occupancy = np.maximum(counts, 1.0)
        cell_mean = sums / occupancy
        cell_variance = np.maximum(squares / occupancy - cell_mean**2, 1.0e-300)
        weights = counts / np.sum(counts)
        values = np.empty((len(case_numbers), int(np.sum(live))))
        for row in range(len(case_numbers)):
            values[row] = gaussian_channel_information(
                cell_mean[row][live], cell_variance[row][live], weights
            )
        return values, cell_mean, cell_variance

    full, cell_mean, cell_variance = plug_in(
        accumulator.cell_sum, accumulator.cell_square_sum, accumulator.cell_count
    )
    half, _, _ = plug_in(
        accumulator.half_cell_sum,
        accumulator.half_cell_square_sum,
        accumulator.half_cell_count,
    )
    # The plug-in channel information is biased upward by the sampling noise of
    # each cell mean, and that bias falls as 1/M.  Two nested sample sizes
    # remove it to leading order, exactly as a first-order jackknife does.
    information = np.full((len(case_numbers), len(lags)), np.nan)
    information[:, information_slice[live]] = (2.0 * full - half) / reduced_lags[live]

    for values in (drift, alignment, level):
        values[:, ~positive] = np.nan

    return NavigationObservables(
        method="particles",
        lags=lags,
        case_numbers=case_numbers,
        projected_velocity=drift,
        alignment_harmonic=alignment,
        level_harmonic=level,
        nutrient_gain=gradient_magnitude * drift,
        information=information,
        conditional_mean=cell_mean,
        conditional_variance=cell_variance,
        projected_velocity_error=drift_error,
        alignment_harmonic_error=alignment_error,
        level_harmonic_error=level_error,
    )
