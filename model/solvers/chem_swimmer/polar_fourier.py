"""Short-time Fourier solution of the manuscript's dynamic polar closure.

This module implements Sec. ``Short-time infinite-plane solution`` of
``manuscript/main.tex``.  The uniform-nutrient density--polarity system is advanced
with its exact Fourier Green matrix.  A heterogeneous nutrient landscape is
included to first order by midpoint quadrature of the manuscript's Duhamel
integral.  At no point is ``partial_t p`` set to zero: the only angular
closure is ``Q = 0``.

The FFT merely supplies a quadrature of the inverse Fourier transforms and
the convolution with the prescribed nutrient field.  A padded periodic box
keeps its artificial boundary away from the reported unit disk; the returned
diagnostics state whether the manuscript's wall-free and weak-response
conditions are actually satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, sqrt

import numpy as np
from numpy.typing import NDArray

from .config import ProjectConfig
from .density import (
    PDF_CASES,
    PositionPDFResult,
    _normalized_smoothed,
    prescribed_initial_density_grid,
    resolve_difference_bandwidth,
    resolve_initial_gaussian_sigma,
    smooth_weighted_stack,
)
from .fields import ChemicalField, disk_area_average
from .numerics import inferred_characteristic_length
from .provenance import build_provenance
from .turning import create_turning_kernel


@dataclass(frozen=True)
class PolarFourierCaseDiagnostics:
    """First-order self-consistency measures for one response case."""

    case_number: int
    max_relative_correction: float
    minimum_density_before_clipping: float
    clipped_negative_mass: float
    minimum_captured_disk_mass: float


@dataclass(frozen=True)
class PolarFourierDiagnostics:
    """Scope and numerical diagnostics for the manuscript plane solution."""

    spectral_grid_size: int
    padding_factor: int
    integration_dt: float
    integration_steps: int
    eta0: float
    gamma1: float
    gamma2: float
    initial_radius: float
    wall_margin: float
    wall_excursion_bound: float
    wall_free_ratio: float
    max_abs_delta_eta: float
    max_gradient: float
    cases: tuple[PolarFourierCaseDiagnostics, ...]


def _response_coefficients(config: ProjectConfig) -> NDArray[np.float64]:
    names = tuple(config.responses.__dataclass_fields__)
    return np.asarray(
        [
            [getattr(config.responses.only_case(case), name) for name in names]
            for case in PDF_CASES
        ],
        dtype=float,
    )


def _green_apply(
    rho_hat: NDArray[np.complex128],
    px_hat: NDArray[np.complex128],
    py_hat: NDArray[np.complex128],
    dt: float,
    kx: NDArray[np.float64],
    ky: NDArray[np.float64],
    k: NDArray[np.float64],
    diffusivity: float,
    speed: float,
    gamma1: float,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """Apply Eqs. (longitudinal-green-matrix) and (transverse-solution)."""

    if dt == 0.0:
        return rho_hat.copy(), px_hat.copy(), py_hat.copy()
    nonzero = k > 0.0
    khat_x = np.zeros_like(k)
    khat_y = np.zeros_like(k)
    khat_x[nonzero] = kx[nonzero] / k[nonzero]
    khat_y[nonzero] = ky[nonzero] / k[nonzero]
    p_longitudinal = khat_x * px_hat + khat_y * py_hat
    p_parallel_x = khat_x * p_longitudinal
    p_parallel_y = khat_y * p_longitudinal
    p_perp_x = px_hat - p_parallel_x
    p_perp_y = py_hat - p_parallel_y

    sigma = np.lib.scimath.sqrt(gamma1**2 - 2.0 * speed**2 * k**2)
    half_argument = 0.5 * sigma * dt
    sinh_over_sigma = np.empty_like(sigma, dtype=np.complex128)
    regular = np.abs(sigma) > 1.0e-12
    sinh_over_sigma[regular] = np.sinh(half_argument[regular]) / sigma[regular]
    sinh_over_sigma[~regular] = 0.5 * dt
    common = np.exp(-(diffusivity * k**2 + 0.5 * gamma1) * dt)
    cosine = np.cosh(half_argument)
    a = common * (cosine + gamma1 * sinh_over_sigma)
    b = common * (-2.0j * speed * k * sinh_over_sigma)
    c = common * (-1.0j * speed * k * sinh_over_sigma)
    d = common * (cosine - gamma1 * sinh_over_sigma)
    rho_new = a * rho_hat + b * p_longitudinal
    p_longitudinal_new = c * rho_hat + d * p_longitudinal
    transverse_decay = np.exp(-(diffusivity * k**2 + gamma1) * dt)
    px_new = khat_x * p_longitudinal_new + transverse_decay * p_perp_x
    py_new = khat_y * p_longitudinal_new + transverse_decay * p_perp_y

    # At k=0 density is conserved and each Cartesian polarity component
    # decays with gamma1; no arbitrary longitudinal direction is introduced.
    zero = ~nonzero
    rho_new[..., zero] = rho_hat[..., zero]
    zero_decay = np.exp(-gamma1 * dt)
    px_new[..., zero] = zero_decay * px_hat[..., zero]
    py_new[..., zero] = zero_decay * py_hat[..., zero]
    return rho_new, px_new, py_new


def _derivatives(
    values: NDArray[np.float64],
    ikx: NDArray[np.complex128],
    iky: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    spectrum = np.fft.fft2(values, axes=(-2, -1))
    dx = np.fft.ifft2(ikx * spectrum, axes=(-2, -1)).real
    dy = np.fft.ifft2(iky * spectrum, axes=(-2, -1)).real
    return dx, dy


def _weak_forcing(
    rho: NDArray[np.float64],
    px: NDArray[np.float64],
    py: NDArray[np.float64],
    eta1: NDArray[np.float64],
    gradient_x: NDArray[np.float64],
    gradient_y: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    one_minus_p_turn: float,
    ikx: NDArray[np.complex128],
    iky: NDArray[np.complex128],
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """Fourier forcing in Eqs. (weak-density/polarity-forcing)."""

    shape = (-1, 1, 1)
    D_T_eta = coefficients[:, 0].reshape(shape)
    D_R_eta = coefficients[:, 1].reshape(shape)
    v_eta = coefficients[:, 2].reshape(shape)
    lambda_eta = coefficients[:, 3].reshape(shape)
    D_T_zeta = coefficients[:, 4].reshape(shape)
    D_R_zeta = coefficients[:, 5].reshape(shape)
    v_zeta = coefficients[:, 6].reshape(shape)
    lambda_zeta = coefficients[:, 7].reshape(shape)
    chi_theta = coefficients[:, 8].reshape(shape)
    gamma_eta = D_R_eta + one_minus_p_turn * lambda_eta
    gamma_zeta = D_R_zeta + one_minus_p_turn * lambda_zeta

    # Density flux, Eq. (weak-density-flux), evaluated as the equivalent
    # physical-space product before Fourier transformation.
    density_potential = D_T_eta * eta1 * rho + D_T_zeta * (
        px * gradient_x + py * gradient_y
    )
    potential_x, potential_y = _derivatives(density_potential, ikx, iky)
    flux_rho_x = v_eta * eta1 * px + 0.5 * v_zeta * rho * gradient_x - potential_x
    flux_rho_y = v_eta * eta1 * py + 0.5 * v_zeta * rho * gradient_y - potential_y
    flux_rho_x_hat = np.fft.fft2(flux_rho_x, axes=(-2, -1))
    flux_rho_y_hat = np.fft.fft2(flux_rho_y, axes=(-2, -1))
    forcing_rho_hat = -(ikx * flux_rho_x_hat + iky * flux_rho_y_hat)

    # Polarity tensor flux, Eq. (weak-polarity-flux).  The first tensor index
    # labels the polarity component and the second the spatial flux direction.
    potential_px = D_T_eta * eta1 * px + 0.5 * D_T_zeta * rho * gradient_x
    potential_py = D_T_eta * eta1 * py + 0.5 * D_T_zeta * rho * gradient_y
    dpx_dx, dpx_dy = _derivatives(potential_px, ikx, iky)
    dpy_dx, dpy_dy = _derivatives(potential_py, ikx, iky)
    dot = px * gradient_x + py * gradient_y
    flux_p_xx = (
        0.5 * v_eta * eta1 * rho
        + 0.25 * v_zeta * (2.0 * px * gradient_x + dot)
        - dpx_dx
    )
    flux_p_xy = 0.25 * v_zeta * (px * gradient_y + gradient_x * py) - dpx_dy
    flux_p_yx = 0.25 * v_zeta * (py * gradient_x + gradient_y * px) - dpy_dx
    flux_p_yy = (
        0.5 * v_eta * eta1 * rho
        + 0.25 * v_zeta * (2.0 * py * gradient_y + dot)
        - dpy_dy
    )
    p_xx_hat = np.fft.fft2(flux_p_xx, axes=(-2, -1))
    p_xy_hat = np.fft.fft2(flux_p_xy, axes=(-2, -1))
    p_yx_hat = np.fft.fft2(flux_p_yx, axes=(-2, -1))
    p_yy_hat = np.fft.fft2(flux_p_yy, axes=(-2, -1))
    source_factor = 0.5 * (chi_theta - gamma_zeta) * rho
    source_x = -gamma_eta * eta1 * px + source_factor * gradient_x
    source_y = -gamma_eta * eta1 * py + source_factor * gradient_y
    forcing_px_hat = -(ikx * p_xx_hat + iky * p_xy_hat)
    forcing_py_hat = -(ikx * p_yx_hat + iky * p_yy_hat)
    forcing_px_hat += np.fft.fft2(source_x, axes=(-2, -1))
    forcing_py_hat += np.fft.fft2(source_y, axes=(-2, -1))
    return forcing_rho_hat, forcing_px_hat, forcing_py_hat


def _initial_radius(
    config: ProjectConfig, field: ChemicalField, spacing: float
) -> float:
    center_radius = float(np.linalg.norm(config.initial.position))
    distribution = config.density.initial_position_distribution
    if distribution == "fixed":
        return min(1.0, center_radius + spacing)
    if distribution == "uniform":
        return 1.0
    sigma = resolve_initial_gaussian_sigma(
        config.density, inferred_characteristic_length(config, field)
    )
    return min(1.0, center_radius + 3.0 * sigma)


def compute_polar_fourier_pdf(
    config: ProjectConfig,
    field: ChemicalField,
    *,
    padding_factor: int = 3,
) -> tuple[PositionPDFResult, PolarFourierDiagnostics]:
    """Evaluate the manuscript's dynamic ``(rho, p)`` plane solution."""

    if padding_factor < 3 or padding_factor % 2 == 0:
        raise ValueError("padding_factor must be an odd integer at least 3")
    # Match the finite-volume manuscript solver's field-resolution rule so
    # density/polarity comparisons do not add an avoidable grid discrepancy.
    from .macroscopic import _macro_grid_size

    grid_size = _macro_grid_size(config, field)
    spacing = 2.0 / grid_size
    padded_size = padding_factor * grid_size
    box_length = padded_size * spacing
    padded_coordinates = -0.5 * box_length + (np.arange(padded_size) + 0.5) * spacing
    x_pad, y_pad = np.meshgrid(padded_coordinates, padded_coordinates)
    padded_points = np.stack((x_pad, y_pad), axis=-1)
    eta, gradient = field.value_and_gradient(padded_points)
    eta = np.asarray(eta, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    eta0 = disk_area_average(field, config.domain.radius)
    eta1 = eta - eta0

    frequencies = 2.0 * np.pi * np.fft.fftfreq(padded_size, d=spacing)
    kx, ky = np.meshgrid(frequencies, frequencies)
    k = np.hypot(kx, ky)
    ikx = (1.0j * kx).astype(np.complex128)
    iky = (1.0j * ky).astype(np.complex128)

    turning = create_turning_kernel(config.turning)
    gamma1 = config.kinetics.D_R0 + config.kinetics.lambda0 * (1.0 - turning.P_turn)
    gamma2 = 4.0 * config.kinetics.D_R0 + config.kinetics.lambda0 * (
        1.0 - turning.P_2_turn
    )
    if gamma1 <= 0.0 or gamma2 <= 0.0:
        raise ValueError("the polar Fourier solution requires gamma1,gamma2 > 0")

    # Choose output times first, then enough midpoint substeps that gamma1*dt
    # is small.  The uniform evolution itself remains exact for every step.
    frame_intervals = max(1, config.density.time_slices - 1)
    substeps = max(
        1,
        ceil(gamma1 * config.simulation.duration / (0.05 * frame_intervals)),
    )
    integration_steps = frame_intervals * substeps
    dt = config.simulation.duration / integration_steps
    save_steps = np.arange(0, integration_steps + 1, substeps, dtype=np.int64)
    times = save_steps.astype(float) * dt
    times[-1] = config.simulation.duration

    center = (padded_size - grid_size) // 2
    crop = (slice(center, center + grid_size), slice(center, center + grid_size))
    # Use the package's canonical cell-center expression exactly; the
    # algebraically equivalent padded slice can differ by a few ulps.
    coordinates = -1.0 + (np.arange(grid_size) + 0.5) * spacing
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    disk_mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    grid_points = np.stack((x_grid, y_grid), axis=-1)
    initial_crop = prescribed_initial_density_grid(
        config, field, grid_points, disk_mask, spacing
    )
    initial_density = np.zeros((padded_size, padded_size), dtype=float)
    initial_density[crop] = initial_crop
    if config.density.initial_heading_distribution == "isotropic":
        heading_x = 0.0
        heading_y = 0.0
    else:
        heading_x = float(np.cos(config.initial.heading))
        heading_y = float(np.sin(config.initial.heading))
    base_rho_hat = np.fft.fft2(initial_density)
    base_px_hat = heading_x * base_rho_hat
    base_py_hat = heading_y * base_rho_hat
    correction_rho_hat = np.zeros(
        (len(PDF_CASES), padded_size, padded_size), dtype=np.complex128
    )
    correction_px_hat = np.zeros_like(correction_rho_hat)
    correction_py_hat = np.zeros_like(correction_rho_hat)
    coefficients = _response_coefficients(config)

    saved_shape = (len(PDF_CASES), len(times), grid_size, grid_size)
    raw_density = np.zeros(saved_shape, dtype=np.float64)
    raw_px = np.zeros(saved_shape, dtype=np.float64)
    raw_py = np.zeros(saved_shape, dtype=np.float64)
    base_norms = np.zeros(len(times), dtype=float)
    correction_norms = np.zeros((len(PDF_CASES), len(times)), dtype=float)
    captured_mass = np.zeros((len(PDF_CASES), len(times)), dtype=float)

    def save(save_index: int) -> None:
        base_rho = np.fft.ifft2(base_rho_hat).real[crop]
        base_px = np.fft.ifft2(base_px_hat).real[crop]
        base_py = np.fft.ifft2(base_py_hat).real[crop]
        corr_rho = np.fft.ifft2(correction_rho_hat, axes=(-2, -1)).real[
            :, crop[0], crop[1]
        ]
        corr_px = np.fft.ifft2(correction_px_hat, axes=(-2, -1)).real[
            :, crop[0], crop[1]
        ]
        corr_py = np.fft.ifft2(correction_py_hat, axes=(-2, -1)).real[
            :, crop[0], crop[1]
        ]
        total_rho = base_rho[None] + corr_rho
        total_px = base_px[None] + corr_px
        total_py = base_py[None] + corr_py
        total_rho[:, ~disk_mask] = 0.0
        total_px[:, ~disk_mask] = 0.0
        total_py[:, ~disk_mask] = 0.0
        raw_density[:, save_index] = total_rho
        raw_px[:, save_index] = total_px
        raw_py[:, save_index] = total_py
        base_norms[save_index] = np.linalg.norm(base_rho[disk_mask])
        correction_norms[:, save_index] = np.linalg.norm(corr_rho[:, disk_mask], axis=1)
        captured_mass[:, save_index] = np.sum(total_rho, axis=(-2, -1)) * spacing**2

    save(0)
    save_index = 1
    for step in range(1, integration_steps + 1):
        mid_rho_hat, mid_px_hat, mid_py_hat = _green_apply(
            base_rho_hat,
            base_px_hat,
            base_py_hat,
            0.5 * dt,
            kx,
            ky,
            k,
            config.kinetics.D_T0,
            config.kinetics.v0,
            gamma1,
        )
        rho_mid = np.fft.ifft2(mid_rho_hat).real
        px_mid = np.fft.ifft2(mid_px_hat).real
        py_mid = np.fft.ifft2(mid_py_hat).real
        forcing = _weak_forcing(
            rho_mid,
            px_mid,
            py_mid,
            eta1,
            gradient[..., 0],
            gradient[..., 1],
            coefficients,
            1.0 - turning.P_turn,
            ikx,
            iky,
        )
        propagated_correction = _green_apply(
            correction_rho_hat,
            correction_px_hat,
            correction_py_hat,
            dt,
            kx,
            ky,
            k,
            config.kinetics.D_T0,
            config.kinetics.v0,
            gamma1,
        )
        midpoint_forcing = _green_apply(
            forcing[0],
            forcing[1],
            forcing[2],
            0.5 * dt,
            kx,
            ky,
            k,
            config.kinetics.D_T0,
            config.kinetics.v0,
            gamma1,
        )
        correction_rho_hat = propagated_correction[0] + dt * midpoint_forcing[0]
        correction_px_hat = propagated_correction[1] + dt * midpoint_forcing[1]
        correction_py_hat = propagated_correction[2] + dt * midpoint_forcing[2]
        base_rho_hat, base_px_hat, base_py_hat = _green_apply(
            base_rho_hat,
            base_px_hat,
            base_py_hat,
            dt,
            kx,
            ky,
            k,
            config.kinetics.D_T0,
            config.kinetics.v0,
            gamma1,
        )
        if step == save_steps[save_index]:
            save(save_index)
            save_index += 1

    minimum_before_clipping = np.min(raw_density, axis=(1, 2, 3))
    negative_mass = np.sum(np.maximum(-raw_density, 0.0), axis=(1, 2, 3)) * spacing**2
    # A first-order asymptotic density is not guaranteed to remain positive
    # outside its regime.  Keep the plotted probability valid and report the
    # exact amount changed by this observation-layer safeguard.
    negative = raw_density < 0.0
    raw_density[negative] = 0.0
    raw_px[negative] = 0.0
    raw_py[negative] = 0.0

    settings = replace(config.density, grid_size=grid_size)
    density, output_coordinates, normalization = _normalized_smoothed(
        raw_density, settings
    )
    polarity_x = smooth_weighted_stack(raw_px, settings, normalization)
    polarity_y = smooth_weighted_stack(raw_py, settings, normalization)
    if not np.allclose(coordinates, output_coordinates, rtol=0.0, atol=1.0e-14):
        raise RuntimeError("polar Fourier smoothing produced an inconsistent grid")
    difference_bandwidth = resolve_difference_bandwidth(
        settings, inferred_characteristic_length(config, field)
    )
    if np.isclose(difference_bandwidth, settings.bandwidth_fraction, atol=1.0e-15):
        difference_density = density
        difference_px = polarity_x
        difference_py = polarity_y
    else:
        difference_settings = replace(settings, bandwidth_fraction=difference_bandwidth)
        difference_density, difference_coordinates, difference_normalization = (
            _normalized_smoothed(raw_density, difference_settings)
        )
        if not np.allclose(coordinates, difference_coordinates, rtol=0.0, atol=1.0e-14):
            raise RuntimeError(
                "polar Fourier difference smoothing produced an inconsistent grid"
            )
        difference_px = smooth_weighted_stack(
            raw_px, difference_settings, difference_normalization
        )
        difference_py = smooth_weighted_stack(
            raw_py, difference_settings, difference_normalization
        )

    # Make the common prescribed initial observation literal and case exact.
    initial_observed, _, initial_normalization = _normalized_smoothed(
        initial_crop[None, None], settings
    )
    density[:, 0] = initial_observed[0, 0]
    polarity_x[:, 0] = heading_x * density[:, 0]
    polarity_y[:, 0] = heading_y * density[:, 0]
    del initial_normalization
    if difference_density is density:
        initial_difference = initial_observed
    else:
        initial_difference, _, _ = _normalized_smoothed(
            initial_crop[None, None], difference_settings
        )
        difference_density[:, 0] = initial_difference[0, 0]
    difference_px[:, 0] = heading_x * difference_density[:, 0]
    difference_py[:, 0] = heading_y * difference_density[:, 0]

    initial_radius = _initial_radius(config, field, spacing)
    wall_margin = max(0.0, 1.0 - initial_radius)
    disk_values = eta1[crop][disk_mask]
    disk_gradient = gradient[crop][disk_mask]
    gradient_norm = np.linalg.norm(disk_gradient, axis=-1)
    max_speed = config.kinetics.v0
    max_diffusivity = config.kinetics.D_T0
    for row in coefficients:
        local_speed = np.abs(config.kinetics.v0 + row[2] * disk_values)
        local_speed += abs(row[6]) * gradient_norm
        local_diffusivity = config.kinetics.D_T0 + row[0] * disk_values
        local_diffusivity += abs(row[4]) * gradient_norm
        max_speed = max(max_speed, float(np.max(local_speed)))
        max_diffusivity = max(max_diffusivity, float(np.max(local_diffusivity)))
    wall_excursion = max_speed * config.simulation.duration + sqrt(
        4.0 * max_diffusivity * config.simulation.duration
    )
    wall_ratio = np.inf if wall_margin == 0.0 else wall_excursion / wall_margin
    relative = correction_norms / np.maximum(base_norms[None], 1.0e-30)
    case_diagnostics = tuple(
        PolarFourierCaseDiagnostics(
            case_number=case,
            max_relative_correction=float(np.max(relative[index])),
            minimum_density_before_clipping=float(minimum_before_clipping[index]),
            clipped_negative_mass=float(negative_mass[index]),
            minimum_captured_disk_mass=float(np.min(captured_mass[index])),
        )
        for index, case in enumerate(PDF_CASES)
    )
    diagnostics = PolarFourierDiagnostics(
        spectral_grid_size=padded_size,
        padding_factor=padding_factor,
        integration_dt=dt,
        integration_steps=integration_steps,
        eta0=eta0,
        gamma1=gamma1,
        gamma2=gamma2,
        initial_radius=initial_radius,
        wall_margin=wall_margin,
        wall_excursion_bound=wall_excursion,
        wall_free_ratio=wall_ratio,
        max_abs_delta_eta=float(np.max(np.abs(disk_values))),
        max_gradient=float(np.max(gradient_norm)),
        cases=case_diagnostics,
    )
    provenance = build_provenance(str(config.source_path))
    result = PositionPDFResult(
        times=times,
        coordinates=coordinates,
        density=density,
        case_numbers=PDF_CASES,
        ensemble_size=0,
        bandwidth_fraction=settings.bandwidth_fraction,
        difference_density=difference_density,
        difference_bandwidth_fraction=difference_bandwidth,
        config_source=str(config.source_path),
        seed=config.simulation.seed,
        dt=dt,
        method="analytic",
        polarity_x=polarity_x,
        polarity_y=polarity_y,
        difference_polarity_x=difference_px,
        difference_polarity_y=difference_py,
        schema_version=provenance.schema_version,
        software_version=provenance.software_version,
        config_text=provenance.config_text,
        config_sha256=provenance.config_sha256,
        runtime_versions=provenance.runtime_versions,
    )
    return result, diagnostics


def format_polar_fourier_diagnostics(
    diagnostics: PolarFourierDiagnostics,
) -> str:
    ratio = (
        "infinity"
        if not np.isfinite(diagnostics.wall_free_ratio)
        else f"{diagnostics.wall_free_ratio:.4g}"
    )
    lines = [
        "Dynamic-p polar Fourier/Green solution (Q=0)",
        (
            f"  FFT quadrature={diagnostics.spectral_grid_size}^2 "
            f"(padding x{diagnostics.padding_factor}), "
            f"Duhamel dt={diagnostics.integration_dt:.6g}, "
            f"steps={diagnostics.integration_steps}"
        ),
        (
            f"  gamma1={diagnostics.gamma1:.6g}, "
            f"gamma2={diagnostics.gamma2:.6g}, eta0={diagnostics.eta0:.6g}"
        ),
        (
            "  wall-free check: "
            f"r0={diagnostics.initial_radius:.4g}, "
            f"excursion={diagnostics.wall_excursion_bound:.4g}, "
            f"margin={diagnostics.wall_margin:.4g}, ratio={ratio} "
            "(manuscript requires ratio << 1)"
        ),
        (
            f"  field scales: max|delta eta|={diagnostics.max_abs_delta_eta:.4g}, "
            f"max|grad eta|={diagnostics.max_gradient:.4g}"
        ),
        "  case  max||rho1||/||rho0||  min rho before clip  clipped mass  min disk mass",
    ]
    for case in diagnostics.cases:
        lines.append(
            f"  {case.case_number:>4}  {case.max_relative_correction:>20.3e}  "
            f"{case.minimum_density_before_clipping:>19.3e}  "
            f"{case.clipped_negative_mass:>12.3e}  "
            f"{case.minimum_captured_disk_mass:>13.6f}"
        )
    lines.append(
        "  Trust the heterogeneous analytic curve only when the correction, "
        "clipped mass, and wall-free ratio are all small."
    )
    return "\n".join(lines)
