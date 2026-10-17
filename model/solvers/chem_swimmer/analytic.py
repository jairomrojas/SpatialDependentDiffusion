"""Analytical tools for the polar-closed theory in ``manuscript/main.tex``.

The public :func:`compute_analytic_pdf` entry point evaluates the manuscript's
short-time, infinite-plane density--polarity solution: the exact uniform-field
Fourier Green matrix plus the first-order weak-landscape Duhamel term.  Thus
``Q=0`` while ``p`` remains a dynamical variable.

The Neumann--Bessel utilities and :func:`compute_weighted_spectral_pdf` are
retained as explicitly named helpers for the optional scalar adiabatic
reduction. They are not used by the public manuscript comparison.

The macroscopic equation and no-flux condition are

    partial_t rho = div(D_eff grad(rho) - kappa rho grad(eta)),
    n . (D_eff grad(rho) - kappa rho grad(eta)) = 0.

With

    w = exp[kappa (eta - eta0) / D_eff],    f = rho / w,

the detailed-balance transformation for this auxiliary scalar PDE is exact:

    w partial_t f = D_eff div(w grad(f)),    partial_n f = 0.

For ``f = phi exp(-mu t)`` this gives the self-adjoint weighted problem

    -D_eff div(w grad(phi)) = mu w phi.

Projection onto the real Neumann--Bessel basis produces the symmetric
generalized eigenproblem ``A c = mu B c`` with

    A_ij = D_eff integral[w grad(q_i).grad(q_j)],
    B_ij =       integral[w q_i q_j].

The matrices are assembled by disk quadrature and diagonalized directly.
For that legacy scalar problem the only spectral approximation is basis truncation; no spurious
position-dependent diffusivity and no additional expansion in ``kappa`` are
introduced.  The finite-volume and spectral solvers consequently solve the
same closed PDE by independent numerical methods.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, pi, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy import special
from scipy.linalg import eigh as generalized_eigh

from .config import ProjectConfig
from .density import (
    PDF_CASES,
    PositionPDFResult,
    prescribed_initial_density_grid,
    resolve_difference_bandwidth,
    resolve_initial_gaussian_sigma,
    smooth_density_stack,
)
from .fields import ChemicalField, create_field, disk_area_average
from .kinetics import macroscopic_coefficients
from .numerics import inferred_characteristic_length, resolve_numerics
from .provenance import build_provenance


@dataclass(frozen=True)
class BesselMode:
    """One Neumann eigenfunction of the disk Laplacian, equation (5)."""

    m: int
    n: int
    parity: str  # "const" | "radial" | "cos" | "sin"
    k: float
    norm: (
        float  # sqrt(N_mn); q(r,theta) = radial_shape(r) * angular_shape(theta) / norm
    )


@dataclass(frozen=True)
class NeumannBesselBasis:
    radius: float
    modes: tuple[BesselMode, ...]

    @property
    def size(self) -> int:
        return len(self.modes)


def build_neumann_bessel_basis(
    radius: float,
    m_max: int,
    n_max: int,
) -> NeumannBesselBasis:
    """The real orthonormal basis of equation (5), truncated at ``m_max``, ``n_max``.

    ``zeta_mn`` is the ``n``-th positive zero of ``J_m'``; modes are the
    constant mode, the axisymmetric radial modes, and cosine/sine pairs for
    each angular order ``m in [1, m_max]``.
    """

    modes: list[BesselMode] = [
        BesselMode(m=0, n=0, parity="const", k=0.0, norm=sqrt(pi * radius**2))
    ]
    zeta_0 = special.jnp_zeros(0, n_max)
    for n in range(1, n_max + 1):
        zeta = float(zeta_0[n - 1])
        k = zeta / radius
        normalization = pi * radius**2 * special.jv(0, zeta) ** 2
        modes.append(
            BesselMode(m=0, n=n, parity="radial", k=k, norm=sqrt(normalization))
        )
    for m in range(1, m_max + 1):
        zeta_m = special.jnp_zeros(m, n_max)
        for n in range(1, n_max + 1):
            zeta = float(zeta_m[n - 1])
            k = zeta / radius
            normalization = (
                0.5
                * pi
                * radius**2
                * (1.0 - (m / zeta) ** 2)
                * special.jv(m, zeta) ** 2
            )
            root_norm = sqrt(normalization)
            modes.append(BesselMode(m=m, n=n, parity="cos", k=k, norm=root_norm))
            modes.append(BesselMode(m=m, n=n, parity="sin", k=k, norm=root_norm))
    return NeumannBesselBasis(radius=radius, modes=tuple(modes))


def evaluate_basis(
    basis: NeumannBesselBasis,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``q_i(x, y)`` for every mode, shape ``(n_modes, *x.shape)``."""

    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    out = np.empty((basis.size,) + x.shape, dtype=np.float64)
    for index, mode in enumerate(basis.modes):
        if mode.parity == "const":
            out[index] = 1.0 / mode.norm
            continue
        radial = special.jv(mode.m, mode.k * r) / mode.norm
        if mode.parity == "radial":
            out[index] = radial
        elif mode.parity == "cos":
            out[index] = radial * np.cos(mode.m * theta)
        else:
            out[index] = radial * np.sin(mode.m * theta)
    return out


def evaluate_basis_gradient(
    basis: NeumannBesselBasis,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Cartesian gradients of every ``q_i``, shape ``(n_modes, *x.shape, 2)``."""

    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    positive_r = r > 1.0e-14
    out = np.zeros((basis.size,) + x.shape + (2,), dtype=np.float64)
    for index, mode in enumerate(basis.modes):
        if mode.parity == "const":
            continue
        radial = special.jv(mode.m, mode.k * r) / mode.norm
        radial_derivative = mode.k * special.jvp(mode.m, mode.k * r, 1) / mode.norm
        if mode.parity == "radial":
            angular = np.ones_like(theta)
            angular_derivative = np.zeros_like(theta)
        elif mode.parity == "cos":
            angular = np.cos(mode.m * theta)
            angular_derivative = -mode.m * np.sin(mode.m * theta)
        else:
            angular = np.sin(mode.m * theta)
            angular_derivative = mode.m * np.cos(mode.m * theta)

        derivative_r = radial_derivative * angular
        derivative_theta_over_r = np.zeros_like(r)
        np.divide(
            radial * angular_derivative,
            r,
            out=derivative_theta_over_r,
            where=positive_r,
        )
        out[index, ..., 0] = cosine * derivative_r - sine * derivative_theta_over_r
        out[index, ..., 1] = sine * derivative_r + cosine * derivative_theta_over_r

        # J_1(k r) ~ k r / 2 supplies the only nonzero Cartesian gradient at r=0.
        if mode.m == 1 and np.any(~positive_r):
            central_slope = mode.k / (2.0 * mode.norm)
            if mode.parity == "cos":
                out[index, ..., 0][~positive_r] = central_slope
                out[index, ..., 1][~positive_r] = 0.0
            elif mode.parity == "sin":
                out[index, ..., 0][~positive_r] = 0.0
                out[index, ..., 1][~positive_r] = central_slope
    return out


def _radial_gauss_legendre(
    radius: float, count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nodes, weights = np.polynomial.legendre.leggauss(count)
    r = 0.5 * radius * (nodes + 1.0)
    w = 0.5 * radius * weights
    return r, w


def compute_overlap_matrix(
    basis: NeumannBesselBasis,
    field: ChemicalField,
    *,
    radial_points: int = 96,
    angular_points: int = 256,
) -> NDArray[np.float64]:
    """``M_ij`` of equation (6), by direct quadrature over the disk.

    Gauss-Legendre in ``r`` (exact for smooth integrands well beyond the
    included Bessel oscillation frequencies) and a uniform grid in ``theta``
    (exponentially convergent for smooth periodic integrands). This
    evaluates the defining integral directly rather than replicating the
    manuscript's closed-form special cases term by term, which is validated
    against those special cases in the test suite instead.
    """

    r_nodes, r_weights = _radial_gauss_legendre(basis.radius, radial_points)
    theta = np.linspace(0.0, 2.0 * np.pi, angular_points, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_nodes, theta, indexing="ij")
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    eta = np.asarray(field.value(np.stack((x, y), axis=-1)), dtype=float)
    eta0 = disk_area_average(
        field,
        basis.radius,
    )
    eta1 = eta - eta0

    q = evaluate_basis(basis, x, y).reshape(basis.size, -1)
    radial_weight = r_weights * r_nodes
    angular_weight = 2.0 * np.pi / angular_points
    total_weight = (radial_weight[:, None] * angular_weight * eta1).reshape(-1)
    return (q * total_weight[None, :]) @ q.T


@dataclass(frozen=True)
class AnalyticSpectrum:
    """Decay rates and Neumann--Bessel coefficients of weighted modes ``phi``."""

    mu: NDArray[np.float64]
    coefficients: NDArray[np.float64]


def solve_analytic_spectrum(
    basis: NeumannBesselBasis,
    field: ChemicalField,
    kappa: float,
    effective_diffusivity: float,
    *,
    eta0: float | None = None,
    radial_points: int = 96,
    angular_points: int = 256,
) -> AnalyticSpectrum:
    """Solve ``-D_eff div(w grad(phi)) = mu w phi`` in the truncated basis."""

    n = basis.size

    r_nodes, r_weights = _radial_gauss_legendre(basis.radius, radial_points)
    theta = np.linspace(0.0, 2.0 * np.pi, angular_points, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_nodes, theta, indexing="ij")
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    eta = np.asarray(field.value(np.stack((x, y), axis=-1)), dtype=float)
    if eta0 is None:
        eta0 = disk_area_average(field, basis.radius)
    weight = np.exp(kappa * (eta - eta0) / effective_diffusivity)

    q = evaluate_basis(basis, x, y).reshape(n, -1)
    gradient_q = evaluate_basis_gradient(basis, x, y).reshape(n, -1, 2)
    radial_weight = r_weights * r_nodes
    angular_weight = 2.0 * np.pi / angular_points
    total_weight = (radial_weight[:, None] * angular_weight * weight).reshape(-1)
    mass = (q * total_weight[None, :]) @ q.T
    stiffness = effective_diffusivity * (
        (gradient_q[..., 0] * total_weight[None, :]) @ gradient_q[..., 0].T
        + (gradient_q[..., 1] * total_weight[None, :]) @ gradient_q[..., 1].T
    )
    # Quadrature roundoff can leave O(eps) asymmetry; enforce the mathematical
    # symmetry before calling the symmetric-definite eigensolver.
    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    mu, c = generalized_eigh(stiffness, mass)
    tolerance = 1.0e-10 * max(1.0, float(np.max(mu)))
    if float(np.min(mu)) < -tolerance:
        raise RuntimeError(
            "the weighted spectral operator produced a negative decay rate"
        )
    mu[np.abs(mu) <= tolerance] = 0.0
    return AnalyticSpectrum(mu=mu, coefficients=c.T.copy())


def _constant_mode_index(basis: NeumannBesselBasis) -> int:
    for index, mode in enumerate(basis.modes):
        if mode.parity == "const":
            return index
    raise ValueError("basis has no constant mode")


def uniform_projection(
    spectrum: AnalyticSpectrum,
    *,
    constant_mode_index: int,
    disk_area: float,
) -> NDArray[np.float64]:
    """The mode-projection coefficients for the uniform initial condition rho_in = 1/|omega_R|.

    This is the special case of :func:`delta_projection`'s general
    derivation for a spatially constant ``rho_in`` (see that function for the
    ``phi_i`` orthogonality this relies on): since the constant mode
    ``q_0 = 1/sqrt(disk_area)`` and ``integral[q_j] = sqrt(disk_area) *
    delta_{j,0}``, ``integral[phi_i] = C_{i,0} * sqrt(disk_area)``, giving
    ``projection_i = integral[rho_in * phi_i] = C_{i,0} / sqrt(disk_area)``.
    """

    return spectrum.coefficients[:, constant_mode_index] / sqrt(disk_area)


def delta_projection(
    spectrum: AnalyticSpectrum,
    basis: NeumannBesselBasis,
    position: tuple[float, float],
) -> NDArray[np.float64]:
    """The mode-projection coefficients for a Dirac delta at ``position``.

    The manuscript reconstructs
    ``rho = w sum_i projection_i phi_i exp(-mu_i t)`` and normalizes modes by
    ``integral[w phi_i phi_j] = delta_ij``.  Since
    ``f(x,0)=rho_in(x)/w(x)``, weighted projection cancels ``w`` and gives
    ``projection_i = integral[rho_in(x) phi_i(x) dx]``.  For a point source
    this reduces to ``projection_i = phi_i(position)``.
    """

    x0 = np.asarray([position[0]], dtype=np.float64)
    y0 = np.asarray([position[1]], dtype=np.float64)
    q_at_position = evaluate_basis(basis, x0, y0)[:, 0]
    return spectrum.coefficients @ q_at_position


def gaussian_projection(
    spectrum: AnalyticSpectrum,
    basis: NeumannBesselBasis,
    position: tuple[float, float],
    sigma: float,
    *,
    radial_points: int = 96,
    angular_points: int = 256,
) -> NDArray[np.float64]:
    """The mode-projection coefficients for a Gaussian of width ``sigma`` at ``position``.

    Same derivation as :func:`delta_projection`
    (``projection_i = integral[rho_in(x) phi_i(x) dx]``), but for a
    finite-width target the integral has no closed form, so it is evaluated
    by the same direct quadrature :func:`compute_overlap_matrix` uses.
    ``rho_in`` is explicitly renormalized to integrate to 1 *over the disk*
    (not the untruncated Gaussian formula's own normalization
    ``1/(2 pi sigma^2)``): for ``sigma`` a non-negligible fraction of the
    disk radius, a real fraction of an unclipped Gaussian's mass would fall
    outside the disk, and skipping this step would silently feed a
    sub-unit-mass target into the projection.

    Unlike :func:`delta_projection`, this is not a numerical shortcut for
    convenience -- it is the intended way to represent "concentrated near
    the center" here: this module's whole derivation requires ``grad(rho) =
    O(epsilon)`` (see the module docstring and
    ``manuscript/main.tex``, which explicitly states "a point release
    or another sharply localized initial profile does not satisfy this
    regime"), so ``sigma`` should be chosen wide enough that the initial
    density does not vary on a finer scale than the field itself does (see
    ``density.resolve_initial_gaussian_sigma``), not narrowed toward zero to
    approximate a delta more closely.
    """

    r_nodes, r_weights = _radial_gauss_legendre(basis.radius, radial_points)
    theta = np.linspace(0.0, 2.0 * np.pi, angular_points, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_nodes, theta, indexing="ij")
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)

    radial_weight = r_weights * r_nodes
    angular_weight = 2.0 * np.pi / angular_points
    weight = radial_weight[:, None] * angular_weight

    squared_distance = (x - position[0]) ** 2 + (y - position[1]) ** 2
    rho_in = np.exp(-0.5 * squared_distance / sigma**2)
    rho_in = rho_in / np.sum(rho_in * weight)

    phi = np.tensordot(
        spectrum.coefficients, evaluate_basis(basis, x, y), axes=([1], [0])
    )
    return (phi.reshape(basis.size, -1) * (rho_in * weight).reshape(-1)).sum(axis=-1)


def evaluate_density(
    spectrum: AnalyticSpectrum,
    basis_values: NDArray[np.float64],
    equilibrium_weight: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    projection: NDArray[np.float64],
) -> NDArray[np.float64]:
    """rho(x, t) from equation (10), generalized to any initial condition.

    ``basis_values`` is ``q_i`` evaluated on the output grid, shape
    ``(n_modes, ny, nx)``; ``equilibrium_weight`` is ``w(x)``;
    ``projection`` is the mode-projection vector for the chosen initial
    condition (see :func:`uniform_projection`, :func:`delta_projection`).
    """

    phi = np.tensordot(spectrum.coefficients, basis_values, axes=([1], [0]))
    weights = projection[:, None] * np.exp(-np.outer(spectrum.mu, times))
    return np.tensordot(weights, phi, axes=([0], [0])) * equilibrium_weight[None, :, :]


def stationary_density(
    equilibrium_weight: NDArray[np.float64],
    mask: NDArray[np.bool_],
    cell_area: float,
) -> NDArray[np.float64]:
    """Return the manuscript's exact ``rho_eq = w / integral(w)``."""

    inside = np.where(mask, equilibrium_weight, 0.0)
    normalization = np.sum(inside) * cell_area
    return inside / normalization


@dataclass(frozen=True)
class AnalyticCaseDiagnostics:
    """Self-consistency diagnostics for one case's exact modal solution."""

    case_number: int
    kappa: float
    max_abs_log_weight: float
    min_equilibrium_weight: float
    max_equilibrium_weight: float
    raw_mass_at_t0: float
    raw_mass_drift: float


@dataclass(frozen=True)
class WeightedSpectralDiagnostics:
    """Validity diagnostics for the whole analytic solve.

        ``raw_mass_at_t0`` and ``raw_mass_drift`` measure finite-basis and output-
        grid quadrature error.  The generalized eigenproblem itself uses the exact
        exponential weight from the manuscript.  The returned
        :class:`~chem_swimmer.density.PositionPDFResult` is renormalized to
        integrate to 1 exactly, matching every other density source in the
    package; this object is where the *un-normalized* residual is reported.
    """

    m_max: int
    n_max: int
    eta0: float
    effective_diffusivity: float
    gamma1: float
    gamma2: float
    cases: tuple[AnalyticCaseDiagnostics, ...]


def format_weighted_spectral_diagnostics(
    diagnostics: WeightedSpectralDiagnostics,
) -> str:
    lines = [
        "Auxiliary adiabatic weighted-spectral solution validity",
        (
            f"  basis: m_max={diagnostics.m_max}, n_max={diagnostics.n_max} "
            f"({(diagnostics.m_max * diagnostics.n_max * 2 + diagnostics.n_max + 1)} modes)"
        ),
        f"  eta0=area_average(eta)={diagnostics.eta0:.6g}",
        (
            f"  D_eff={diagnostics.effective_diffusivity:.6g}, "
            f"gamma1={diagnostics.gamma1:.6g}, gamma2={diagnostics.gamma2:.6g}"
        ),
        "  case  kappa    max|log(w)|   min(w)   max(w)  raw_mass(t=0)  raw_mass_drift",
    ]
    for case in diagnostics.cases:
        lines.append(
            f"  {case.case_number:>4}  {case.kappa:+.4f}  "
            f"{case.max_abs_log_weight:>11.4f}  "
            f"{case.min_equilibrium_weight:>7.4f}  "
            f"{case.max_equilibrium_weight:>7.4f}  "
            f"{case.raw_mass_at_t0:>13.6f}  {case.raw_mass_drift:>14.2e}"
        )
    lines.append(
        "  w=exp[kappa(eta-eta0)/D_eff] is strictly positive.  max|log(w)| "
        "measures the weak-heterogeneity expansion parameter; the mass "
        "residuals measure finite basis/output-grid truncation."
    )
    return "\n".join(lines)


def compute_weighted_spectral_pdf(
    config: ProjectConfig,
    *,
    field: ChemicalField | None = None,
    m_max: int = 16,
    n_max: int = 12,
    radial_quadrature_points: int = 96,
    angular_quadrature_points: int = 256,
) -> tuple[PositionPDFResult, WeightedSpectralDiagnostics]:
    """Evaluate the analytic modal solution for Cases 0-9.

    Reuses the exact microscopic time schedule (same duration, dt
    resolution, and ``density.time_slices``) so the result is directly,
    interpolation-free comparable to :func:`density.compute_position_pdf`.
    The returned density is renormalized to integrate to 1 at every case and
    time, for compatibility with the rest of the package; the un-normalized
    self-consistency residual is reported in the accompanying
    :class:`WeightedSpectralDiagnostics`.
    """

    field = create_field(config.field) if field is None else field
    resolved_config, _ = resolve_numerics(config, field, PDF_CASES)
    if isinstance(resolved_config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")

    steps = round(resolved_config.simulation.duration / resolved_config.simulation.dt)
    interval = max(1, ceil(steps / (resolved_config.density.time_slices - 1)))
    save_steps = np.arange(0, steps + 1, interval, dtype=np.int64)
    if save_steps[-1] != steps:
        save_steps = np.append(save_steps, steps)
    time_scale = resolved_config.domain.radius**2 / resolved_config.kinetics.D_T0
    times = save_steps.astype(float) * resolved_config.simulation.dt / time_scale
    times[-1] = resolved_config.simulation.duration / time_scale

    grid_size = resolved_config.density.grid_size
    spacing = 2.0 / grid_size
    coordinates = -1.0 + (np.arange(grid_size) + 0.5) * spacing
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    disk_area = pi * resolved_config.domain.radius**2

    basis = build_neumann_bessel_basis(resolved_config.domain.radius, m_max, n_max)
    basis_values = evaluate_basis(basis, x_grid, y_grid)
    constant_index = _constant_mode_index(basis)

    from .turning import create_turning_kernel

    turning_kernel = create_turning_kernel(resolved_config.turning)
    gamma1, effective_diffusivity, _ = macroscopic_coefficients(
        resolved_config.kinetics,
        resolved_config.responses.zeros(),
        turning_kernel.P_turn,
    )
    if not (np.isfinite(gamma1) and gamma1 > 0.0):
        raise ValueError("gamma1 must be finite and positive for the analytic solution")
    gamma2 = 4.0 * resolved_config.kinetics.D_R0 + resolved_config.kinetics.lambda0 * (
        1.0 - turning_kernel.P_2_turn
    )
    if not (np.isfinite(gamma2) and gamma2 > 0.0):
        raise ValueError(
            "gamma2 = 4 D_R0 + lambda0(1-P_2_turn) must be positive for "
            "the manuscript's moment closure"
        )

    eta = np.asarray(field.value(np.stack((x_grid, y_grid), axis=-1)), dtype=float)
    eta0 = disk_area_average(field, resolved_config.domain.radius)
    eta1 = eta - eta0
    cell_area = spacing**2
    initial_position = resolved_config.initial.position

    initial_distribution = resolved_config.density.initial_position_distribution
    is_uniform = initial_distribution == "uniform"
    gaussian_sigma = resolve_initial_gaussian_sigma(
        resolved_config.density, inferred_characteristic_length(resolved_config, field)
    )
    raw_density = np.zeros(
        (len(PDF_CASES), len(times), grid_size, grid_size), dtype=np.float64
    )
    case_diagnostics: list[AnalyticCaseDiagnostics] = []
    for case_index, case_number in enumerate(PDF_CASES):
        _, _, kappa = macroscopic_coefficients(
            resolved_config.kinetics,
            resolved_config.responses.only_case(case_number),
            turning_kernel.P_turn,
        )
        log_weight = kappa * eta1 / effective_diffusivity
        equilibrium_weight = np.exp(log_weight)
        spectrum = solve_analytic_spectrum(
            basis,
            field,
            kappa,
            effective_diffusivity,
            eta0=eta0,
            radial_points=radial_quadrature_points,
            angular_points=angular_quadrature_points,
        )
        if is_uniform:
            projection = uniform_projection(
                spectrum, constant_mode_index=constant_index, disk_area=disk_area
            )
        elif initial_distribution == "gaussian":
            projection = gaussian_projection(
                spectrum,
                basis,
                initial_position,
                gaussian_sigma,
                radial_points=radial_quadrature_points,
                angular_points=angular_quadrature_points,
            )
        else:
            projection = delta_projection(spectrum, basis, initial_position)
        case_density = evaluate_density(
            spectrum,
            basis_values,
            equilibrium_weight,
            times,
            projection=projection,
        )
        case_density[:, ~mask] = 0.0

        mass = np.sum(case_density, axis=(-2, -1), dtype=np.float64) * cell_area
        case_diagnostics.append(
            AnalyticCaseDiagnostics(
                case_number=case_number,
                kappa=kappa,
                max_abs_log_weight=float(np.max(np.abs(log_weight[mask]))),
                min_equilibrium_weight=float(np.min(equilibrium_weight[mask])),
                max_equilibrium_weight=float(np.max(equilibrium_weight[mask])),
                raw_mass_at_t0=float(mass[0]),
                raw_mass_drift=float(np.max(np.abs(mass - mass[0]))),
            )
        )
        raw_density[case_index] = case_density

    diagnostics = WeightedSpectralDiagnostics(
        m_max=m_max,
        n_max=n_max,
        eta0=eta0,
        effective_diffusivity=effective_diffusivity,
        gamma1=gamma1,
        gamma2=gamma2,
        cases=tuple(case_diagnostics),
    )

    if is_uniform:
        # The uniform IC is exactly representable by the constant mode alone,
        # so there is no discretization roughness to smooth away; normalize
        # each case's own raw (near-unit) mass directly, as before.
        masses = np.sum(raw_density, axis=(-2, -1), dtype=np.float64) * cell_area
        density = raw_density / masses[..., None, None]
        difference_density = density
        difference_bandwidth_fraction = resolved_config.density.bandwidth_fraction
    else:
        # A delta IC, by contrast, leaves real mode-truncation roughness in
        # the raw solution (see the module docstring); smooth it exactly as
        # macroscopic.compute_macroscopic_pdf smooths its own raw,
        # unsmoothed finite-volume grid, so the two stay comparable.
        density, smoothed_coordinates = smooth_density_stack(
            raw_density, resolved_config.density
        )
        if not np.array_equal(coordinates, smoothed_coordinates):
            raise RuntimeError("analytic KDE produced an inconsistent grid")
        difference_bandwidth_fraction = resolve_difference_bandwidth(
            resolved_config.density,
            inferred_characteristic_length(resolved_config, field),
        )
        if np.isclose(
            difference_bandwidth_fraction,
            resolved_config.density.bandwidth_fraction,
            rtol=0.0,
            atol=1.0e-15,
        ):
            difference_density = density
        else:
            difference_density, difference_coordinates = smooth_density_stack(
                raw_density,
                replace(
                    resolved_config.density,
                    bandwidth_fraction=difference_bandwidth_fraction,
                ),
            )
            if not np.array_equal(coordinates, difference_coordinates):
                raise RuntimeError(
                    "analytic difference KDE produced an inconsistent grid"
                )

    grid_points = np.stack((x_grid, y_grid), axis=-1)
    initial_grid = prescribed_initial_density_grid(
        resolved_config,
        field,
        grid_points,
        mask,
        spacing,
    )
    initial_density, initial_coordinates = smooth_density_stack(
        initial_grid[None, None],
        resolved_config.density,
    )
    if not np.array_equal(coordinates, initial_coordinates):
        raise RuntimeError(
            "analytic initial-density estimator produced an inconsistent grid"
        )
    density[:, 0] = initial_density[0, 0]
    if difference_density is not density:
        initial_difference_density, initial_difference_coordinates = (
            smooth_density_stack(
                initial_grid[None, None],
                replace(
                    resolved_config.density,
                    bandwidth_fraction=difference_bandwidth_fraction,
                ),
            )
        )
        if not np.array_equal(coordinates, initial_difference_coordinates):
            raise RuntimeError(
                "analytic difference initial-density estimator produced an inconsistent grid"
            )
        difference_density[:, 0] = initial_difference_density[0, 0]

    provenance = build_provenance(str(resolved_config.source_path))
    result = PositionPDFResult(
        times=times,
        coordinates=coordinates,
        density=density,
        case_numbers=PDF_CASES,
        ensemble_size=0,
        bandwidth_fraction=resolved_config.density.bandwidth_fraction,
        difference_density=difference_density,
        difference_bandwidth_fraction=difference_bandwidth_fraction,
        config_source=str(resolved_config.source_path),
        seed=resolved_config.simulation.seed,
        dt=float(resolved_config.simulation.dt),
        method="analytic",
        schema_version=provenance.schema_version,
        software_version=provenance.software_version,
        config_text=provenance.config_text,
        config_sha256=provenance.config_sha256,
        runtime_versions=provenance.runtime_versions,
    )
    return result, diagnostics


# ---------------------------------------------------------------------------
# Current manuscript public API
# ---------------------------------------------------------------------------
# Keep the weighted Neumann--Bessel functions above importable for notebooks
# that study the *additional* scalar adiabatic approximation, but route the
# advertised ``analytic`` strategy to the dynamic-p solution actually derived
# in manuscript/main.tex.
from .polar_fourier import (
    PolarFourierDiagnostics as AnalyticDiagnostics,
)
from .polar_fourier import (
    compute_polar_fourier_pdf,
    format_polar_fourier_diagnostics,
)


def format_analytic_diagnostics(diagnostics: AnalyticDiagnostics) -> str:
    """Report the manuscript plane-solution validity conditions."""

    return format_polar_fourier_diagnostics(diagnostics)


def compute_analytic_pdf(
    config: ProjectConfig,
    *,
    field: ChemicalField | None = None,
    padding_factor: int = 3,
    **legacy_scalar_options,
) -> tuple[PositionPDFResult, AnalyticDiagnostics]:
    """Evaluate the dynamic ``(rho,p)`` Fourier/Green solution with ``Q=0``.

    ``m_max``, ``n_max`` and the old quadrature keywords belonged to the
    superseded scalar Neumann--Bessel reduction.  Supplying any of them is an
    error rather than silently comparing a different mathematical problem.
    """

    if legacy_scalar_options:
        names = ", ".join(sorted(legacy_scalar_options))
        raise TypeError(
            f"obsolete scalar-adiabatic analytic options: {names}; the current "
            "analytic strategy follows manuscript/main.tex and accepts only "
            "padding_factor"
        )
    field = create_field(config.field) if field is None else field
    return compute_polar_fourier_pdf(
        config,
        field,
        padding_factor=padding_factor,
    )
