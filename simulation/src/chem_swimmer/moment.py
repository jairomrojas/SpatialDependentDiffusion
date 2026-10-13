"""Density--polarity evolution with the polar closure ``Q = 0``.

This is the retained level of the angular hierarchy and the implementation
used by the public macroscopic finite-volume strategy: density and the two
components of polarity are evolved.  The traceless nematic tensor is set to
zero exactly as in the polar closure of ``manuscript/main.tex``; it is not
reconstructed from an adiabatic nematic balance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .config import ProjectConfig
from .density import (
    PDF_CASES,
    PositionPDFResult,
    _normalized_smoothed,
    prescribed_initial_density_grid,
    resolve_difference_bandwidth,
    smooth_weighted_stack,
)
from .fields import ChemicalField, create_field
from .fields.base import disk_area_average
from .macroscopic import _disk_grid, _macro_grid_size
from .numerics import inferred_characteristic_length
from .progress import progress_bar
from .provenance import build_provenance
from .turning import create_turning_kernel


@dataclass(frozen=True)
class MomentPlan:
    """Grid, coefficients, and explicit stability plan for the polar system."""

    grid_size: int
    spacing: float
    gamma1: float
    gamma2: float
    diffusion_dt: float
    wave_dt: float
    relaxation_dt: float
    safe_dt: float
    dt: float
    steps: int
    save_steps: NDArray[np.int64]
    max_diffusivity: float
    max_wave_speed: float
    max_relaxation_rate: float
    min_diffusivity: float
    cell_peclet: float

    @property
    def artificial_diffusivity(self) -> float:
        """Numerical diffusivity the hybrid face flux still has to add.

        Zero whenever ``cell_peclet <= 2``, i.e. whenever the physical
        diffusion already keeps the centred scheme monotone.  Plain Rusanov
        would instead add ``max_wave_speed * spacing / 2`` unconditionally.
        """

        return max(0.0, 0.5 * self.max_wave_speed * self.spacing - self.min_diffusivity)


def _case_coefficients(config: ProjectConfig) -> NDArray[np.float64]:
    names = tuple(config.responses.__dataclass_fields__)
    return np.asarray(
        [
            [getattr(config.responses.only_case(case), name) for name in names]
            for case in PDF_CASES
        ],
        dtype=float,
    )


def _compact_neighbors(active_ids: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return compact left/right/down/up neighbor ids for disk cells."""

    mask = active_ids >= 0
    neighbors = np.full((int(np.count_nonzero(mask)), 4), -1, dtype=np.int64)
    horizontal = mask[:, :-1] & mask[:, 1:]
    rows, columns = np.nonzero(horizontal)
    left = active_ids[rows, columns]
    right = active_ids[rows, columns + 1]
    neighbors[left, 1] = right
    neighbors[right, 0] = left
    vertical = mask[:-1, :] & mask[1:, :]
    rows, columns = np.nonzero(vertical)
    bottom = active_ids[rows, columns]
    top = active_ids[rows + 1, columns]
    neighbors[bottom, 3] = top
    neighbors[top, 2] = bottom
    return neighbors


def _compact_edges(active_ids: NDArray[np.int64]) -> tuple[NDArray[np.int64], ...]:
    """Return each internal face once, oriented in positive x or y."""

    mask = active_ids >= 0
    horizontal = mask[:, :-1] & mask[:, 1:]
    rows_x, columns_x = np.nonzero(horizontal)
    vertical = mask[:-1, :] & mask[1:, :]
    rows_y, columns_y = np.nonzero(vertical)
    left = active_ids[rows_x, columns_x]
    right = active_ids[rows_x, columns_x + 1]
    bottom = active_ids[rows_y, columns_y]
    top = active_ids[rows_y + 1, columns_y]
    edge_left = np.concatenate((left, bottom)).astype(np.int64)
    edge_right = np.concatenate((right, top)).astype(np.int64)
    edge_axis = np.concatenate(
        (np.zeros(len(left), dtype=np.int64), np.ones(len(bottom), dtype=np.int64))
    )
    return edge_left, edge_right, edge_axis


@njit(cache=True, parallel=True)
def _integrate_moment_system(
    initial_density: NDArray[np.float64],
    initial_polarity_x: NDArray[np.float64],
    initial_polarity_y: NDArray[np.float64],
    edge_left: NDArray[np.int64],
    edge_right: NDArray[np.int64],
    edge_axis: NDArray[np.int64],
    delta_eta: NDArray[np.float64],
    gradient_x: NDArray[np.float64],
    gradient_y: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    v0: float,
    D_T0: float,
    D_R0: float,
    lambda0: float,
    P_turn: float,
    spacing: float,
    dt: float,
    save_steps: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Advance all cases with conservative central + hybrid-dissipation fluxes."""

    case_count = coefficients.shape[0]
    active_count = len(initial_density)
    save_count = len(save_steps)
    saved_rho = np.empty((case_count, save_count, active_count), dtype=np.float64)
    saved_px = np.empty_like(saved_rho)
    saved_py = np.empty_like(saved_rho)
    for case_index in prange(case_count):
        # Responses order follows config.Responses.
        D_T_eta = coefficients[case_index, 0]
        D_R_eta = coefficients[case_index, 1]
        v_eta = coefficients[case_index, 2]
        lambda_eta = coefficients[case_index, 3]
        D_T_zeta = coefficients[case_index, 4]
        D_R_zeta = coefficients[case_index, 5]
        v_zeta = coefficients[case_index, 6]
        lambda_zeta = coefficients[case_index, 7]
        chi_theta = coefficients[case_index, 8]

        rho = initial_density.copy()
        px = initial_polarity_x.copy()
        py = initial_polarity_y.copy()
        next_rho = np.empty_like(rho)
        next_px = np.empty_like(px)
        next_py = np.empty_like(py)
        drho = np.empty_like(rho)
        dpx = np.empty_like(px)
        dpy = np.empty_like(py)
        density_potential = np.empty_like(rho)
        polarity_potential_x = np.empty_like(rho)
        polarity_potential_y = np.empty_like(rho)
        translational = np.empty_like(rho)

        for cell in range(active_count):
            saved_rho[case_index, 0, cell] = rho[cell]
            saved_px[case_index, 0, cell] = px[cell]
            saved_py[case_index, 0, cell] = py[cell]
        save_index = 1
        final_step = save_steps[-1]
        for step in range(1, final_step + 1):
            for cell in range(active_count):
                translational_factor = D_T0 + D_T_eta * delta_eta[cell]
                translational[cell] = translational_factor
                density_potential[cell] = translational_factor * rho[cell]
                density_potential[cell] += D_T_zeta * (
                    px[cell] * gradient_x[cell] + py[cell] * gradient_y[cell]
                )
                polarity_potential_x[cell] = translational_factor * px[cell]
                polarity_potential_x[cell] += (
                    0.5 * D_T_zeta * rho[cell] * gradient_x[cell]
                )
                polarity_potential_y[cell] = translational_factor * py[cell]
                polarity_potential_y[cell] += (
                    0.5 * D_T_zeta * rho[cell] * gradient_y[cell]
                )
                drho[cell] = 0.0
                dpx[cell] = 0.0
                dpy[cell] = 0.0

            for edge in range(len(edge_left)):
                a = edge_left[edge]
                b = edge_right[edge]
                axis = edge_axis[edge]
                if axis == 0:
                    pa = px[a]
                    pb = px[b]
                    ga_n = gradient_x[a]
                    gb_n = gradient_x[b]
                else:
                    pa = py[a]
                    pb = py[b]
                    ga_n = gradient_y[a]
                    gb_n = gradient_y[b]

                va = v0 + v_eta * delta_eta[a]
                vb = v0 + v_eta * delta_eta[b]
                # Third moment under the manuscript's polar closure Q=0:
                # M3_ijk=(p_i delta_jk+p_j delta_ik+p_k delta_ij)/4.
                if axis == 0:
                    m3x_a = (3.0 * px[a] * gradient_x[a] + py[a] * gradient_y[a]) / 4.0
                    m3x_b = (3.0 * px[b] * gradient_x[b] + py[b] * gradient_y[b]) / 4.0
                    m3y_a = (py[a] * gradient_x[a] + px[a] * gradient_y[a]) / 4.0
                    m3y_b = (py[b] * gradient_x[b] + px[b] * gradient_y[b]) / 4.0
                    prop_px_a = 0.5 * va * rho[a] + v_zeta * m3x_a
                    prop_px_b = 0.5 * vb * rho[b] + v_zeta * m3x_b
                    prop_py_a = v_zeta * m3y_a
                    prop_py_b = v_zeta * m3y_b
                else:
                    m3x_a = (px[a] * gradient_y[a] + py[a] * gradient_x[a]) / 4.0
                    m3x_b = (px[b] * gradient_y[b] + py[b] * gradient_x[b]) / 4.0
                    m3y_a = (px[a] * gradient_x[a] + 3.0 * py[a] * gradient_y[a]) / 4.0
                    m3y_b = (px[b] * gradient_x[b] + 3.0 * py[b] * gradient_y[b]) / 4.0
                    prop_px_a = v_zeta * m3x_a
                    prop_px_b = v_zeta * m3x_b
                    prop_py_a = 0.5 * va * rho[a] + v_zeta * m3y_a
                    prop_py_b = 0.5 * vb * rho[b] + v_zeta * m3y_b

                density_flux_a = va * pa + 0.5 * v_zeta * rho[a] * ga_n
                density_flux_b = vb * pb + 0.5 * v_zeta * rho[b] * gb_n
                wave_speed = max(
                    abs(va) + abs(v_zeta) * np.hypot(gradient_x[a], gradient_y[a]),
                    abs(vb) + abs(v_zeta) * np.hypot(gradient_x[b], gradient_y[b]),
                )
                # Hybrid (Rusanov-limited) dissipation.  Plain Rusanov adds a
                # numerical diffusivity wave_speed*spacing/2 unconditionally,
                # which at the production grid is a sizeable fraction of D_T0
                # and -- because wave_speed carries v_eta and v_zeta -- is
                # itself case dependent, contaminating the speed responses.
                # Central differencing is already monotone once the cell
                # Peclet number 2*face_diffusivity/(wave_speed*spacing) >= 1,
                # so add only the deficit the grid actually needs.
                face_diffusivity = min(translational[a], translational[b])
                dissipation = max(0.0, 0.5 * wave_speed - face_diffusivity / spacing)
                flux_rho = 0.5 * (density_flux_a + density_flux_b)
                flux_rho -= dissipation * (rho[b] - rho[a])
                flux_rho -= (density_potential[b] - density_potential[a]) / spacing
                flux_px = 0.5 * (prop_px_a + prop_px_b)
                flux_px -= dissipation * (px[b] - px[a])
                flux_px -= (polarity_potential_x[b] - polarity_potential_x[a]) / spacing
                flux_py = 0.5 * (prop_py_a + prop_py_b)
                flux_py -= dissipation * (py[b] - py[a])
                flux_py -= (polarity_potential_y[b] - polarity_potential_y[a]) / spacing

                inverse_spacing = 1.0 / spacing
                drho[a] -= flux_rho * inverse_spacing
                drho[b] += flux_rho * inverse_spacing
                dpx[a] -= flux_px * inverse_spacing
                dpx[b] += flux_px * inverse_spacing
                dpy[a] -= flux_py * inverse_spacing
                dpy[b] += flux_py * inverse_spacing

            for cell in range(active_count):
                local_gamma1 = D_R0 + D_R_eta * delta_eta[cell]
                local_gamma1 += (1.0 - P_turn) * (
                    lambda0 + lambda_eta * delta_eta[cell]
                )
                directional_loss = D_R_zeta + (1.0 - P_turn) * lambda_zeta
                polar_source = 0.5 * (chi_theta - directional_loss) * rho[cell]
                dpx[cell] -= local_gamma1 * px[cell]
                dpy[cell] -= local_gamma1 * py[cell]
                dpx[cell] += polar_source * gradient_x[cell]
                dpy[cell] += polar_source * gradient_y[cell]
                next_rho[cell] = rho[cell] + dt * drho[cell]
                next_px[cell] = px[cell] + dt * dpx[cell]
                next_py[cell] = py[cell] + dt * dpy[cell]

            swap = rho
            rho = next_rho
            next_rho = swap
            swap = px
            px = next_px
            next_px = swap
            swap = py
            py = next_py
            next_py = swap
            if save_index < save_count and step == save_steps[save_index]:
                for cell in range(active_count):
                    saved_rho[case_index, save_index, cell] = rho[cell]
                    saved_px[case_index, save_index, cell] = px[cell]
                    saved_py[case_index, save_index, cell] = py[cell]
                save_index += 1
    return saved_rho, saved_px, saved_py


def build_moment_plan(
    config: ProjectConfig,
    field: ChemicalField | None = None,
) -> tuple[
    MomentPlan,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Build the disk mesh and a conservative explicit stability bound."""

    field = create_field(config.field) if field is None else field
    grid_size = _macro_grid_size(config, field)
    coordinates, grid_points, mask, active_ids = _disk_grid(grid_size)
    spacing = 2.0 / grid_size
    eta, gradient = field.value_and_gradient(grid_points)
    eta = np.asarray(eta, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if eta.shape != mask.shape or gradient.shape != (*mask.shape, 2):
        raise ValueError("the chemical field returned incompatible grid shapes")
    if not np.all(np.isfinite(eta[mask])) or not np.all(np.isfinite(gradient[mask])):
        raise ValueError("the chemical field must be finite throughout the disk")

    turning = create_turning_kernel(config.turning)
    gamma1 = config.kinetics.D_R0 + config.kinetics.lambda0 * (1.0 - turning.P_turn)
    gamma2 = 4.0 * config.kinetics.D_R0 + config.kinetics.lambda0 * (
        1.0 - turning.P_2_turn
    )
    if gamma1 <= 0.0:
        raise ValueError("the polar solver requires gamma1 > 0")
    if gamma2 <= 0.0:
        raise ValueError("the polar solver requires gamma2 > 0")

    coefficients = _case_coefficients(config)
    eta0 = disk_area_average(field, config.domain.radius)
    delta_eta = eta - eta0
    gradient_norm = np.linalg.norm(gradient, axis=-1)
    max_diffusivity = config.kinetics.D_T0
    max_wave_speed = config.kinetics.v0
    max_relaxation_rate = gamma1
    for row in coefficients:
        D_T_eta, D_R_eta, v_eta, lambda_eta, D_T_zeta, _, v_zeta, _, _ = row
        local_diffusivity = config.kinetics.D_T0 + D_T_eta * delta_eta
        local_diffusivity += np.abs(D_T_zeta) * gradient_norm
        max_diffusivity = max(max_diffusivity, float(np.max(local_diffusivity[mask])))
        local_speed = np.abs(config.kinetics.v0 + v_eta * delta_eta)
        local_speed += abs(v_zeta) * gradient_norm
        max_wave_speed = max(max_wave_speed, float(np.max(local_speed[mask])))
        local_relaxation = config.kinetics.D_R0 + D_R_eta * delta_eta
        local_relaxation += (1.0 - turning.P_turn) * (
            config.kinetics.lambda0 + lambda_eta * delta_eta
        )
        max_relaxation_rate = max(
            max_relaxation_rate, float(np.max(local_relaxation[mask]))
        )
    min_diffusivity = config.kinetics.D_T0
    for row in coefficients:
        D_T_eta = row[0]
        local = config.kinetics.D_T0 + D_T_eta * delta_eta
        min_diffusivity = min(min_diffusivity, float(np.min(local[mask])))
    if min_diffusivity <= 0.0:
        raise ValueError("the moment solver requires a positive local D_T")
    cell_peclet = max_wave_speed * spacing / min_diffusivity
    diffusion_dt = spacing**2 / (4.0 * max_diffusivity)
    wave_dt = np.inf if max_wave_speed == 0.0 else spacing / (2.0 * max_wave_speed)
    relaxation_dt = 1.0 / max_relaxation_rate
    # Forward Euler sees all three damping mechanisms in the same update.
    # Their rates therefore add; taking only the smallest individual bound is
    # insufficient when physical diffusion and Rusanov wave dissipation are
    # comparable.  The harmonic combination is the conservative joint bound.
    inverse_joint_dt = 1.0 / diffusion_dt + 1.0 / relaxation_dt
    if np.isfinite(wave_dt):
        inverse_joint_dt += 1.0 / wave_dt
    safe_dt = config.macro.cfl_safety / inverse_joint_dt
    requested_dt = config.simulation.dt
    if isinstance(requested_dt, str):
        target_dt = safe_dt
    else:
        if requested_dt > safe_dt * (1.0 + 1.0e-12):
            raise ValueError(
                f"the requested moment dt={requested_dt:.6g} exceeds the "
                f"safe limit {safe_dt:.6g}"
            )
        target_dt = requested_dt
    steps = max(1, ceil(config.simulation.duration / target_dt))
    dt = config.simulation.duration / steps
    save_count = min(config.density.time_slices, steps + 1)
    save_steps = np.unique(np.linspace(0, steps, save_count, dtype=np.int64))
    if save_steps[-1] != steps:
        save_steps = np.append(save_steps, steps)
    plan = MomentPlan(
        grid_size=grid_size,
        spacing=spacing,
        gamma1=gamma1,
        gamma2=gamma2,
        diffusion_dt=diffusion_dt,
        wave_dt=wave_dt,
        relaxation_dt=relaxation_dt,
        safe_dt=safe_dt,
        dt=dt,
        steps=steps,
        save_steps=save_steps,
        max_diffusivity=max_diffusivity,
        max_wave_speed=max_wave_speed,
        max_relaxation_rate=max_relaxation_rate,
        min_diffusivity=min_diffusivity,
        cell_peclet=cell_peclet,
    )
    neighbors = _compact_neighbors(active_ids)
    edge_left, edge_right, edge_axis = _compact_edges(active_ids)
    return (
        plan,
        coordinates,
        grid_points,
        mask,
        neighbors,
        edge_left,
        edge_right,
        edge_axis,
        eta,
        gradient,
    )


def moment_timestep_report(plan: MomentPlan, config: ProjectConfig) -> str:
    """Describe the retained fields, closure, and explicit timestep."""

    return (
        "\nDensity--polarity moment evolution (polar closure)\n"
        "  evolve rho and p; set Q = 0 as in Eq. (polar-closure)\n"
        "  partial_t rho = -div(J_rho),  partial_t p = -div(J_p) + angular terms\n"
        "  n.J_rho = 0 and n.J_p = 0 on cut-cell disk faces\n"
        f"  grid={plan.grid_size}x{plan.grid_size}, Delta(x)={plan.spacing:.6g}\n"
        f"  gamma1={plan.gamma1:.6g}, gamma2={plan.gamma2:.6g}\n"
        f"  cell Peclet a_max Delta(x)/D_T_min={plan.cell_peclet:.4g}"
        f" -> added numerical diffusivity {plan.artificial_diffusivity:.4g}"
        f" (Rusanov would add {0.5 * plan.max_wave_speed * plan.spacing:.4g})\n"
        "\n"
        "  Process                 Candidate formula                         dt\n"
        "  ----------------------  ---------------------------------------  -----------\n"
        f"  parabolic               Delta(x)^2 / (4 D_max)                {plan.diffusion_dt:.6g}\n"
        f"  moment waves            Delta(x) / (2 a_max)                  {plan.wave_dt:.6g}\n"
        f"  angular relaxation      1 / gamma_max                         {plan.relaxation_dt:.6g}\n"
        f"  Selected safe limit     {config.macro.cfl_safety:.3g} / sum(inverse candidates)          {plan.safe_dt:.6g}\n"
        f"  Simulation dt           t_final / ceil(t_final/safe limit)      {plan.dt:.6g}\n"
        f"  Steps                    {plan.steps:,} to reach t_final={config.simulation.duration:.6g}"
    )


def _smooth_moment_fields(
    raw_density: NDArray[np.float64],
    raw_px: NDArray[np.float64],
    raw_py: NDArray[np.float64],
    settings,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    density, coordinates, normalization = _normalized_smoothed(raw_density, settings)
    px = smooth_weighted_stack(raw_px, settings, normalization)
    py = smooth_weighted_stack(raw_py, settings, normalization)
    return density, px, py, coordinates


@dataclass(frozen=True)
class MomentSolution:
    """Raw finite-volume ``(rho, p)`` history on the active disk cells.

    Fields are indexed ``[case, saved time, cell]`` and carry no kernel-density
    smoothing, so global moments taken from them are the solver's own.
    """

    plan: MomentPlan
    times: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    points: NDArray[np.float64]
    mask: NDArray[np.bool_]
    spacing: float
    density: NDArray[np.float64]
    polarity_x: NDArray[np.float64]
    polarity_y: NDArray[np.float64]
    initial_grid: NDArray[np.float64]
    initial_heading: tuple[float, float]


def integrate_moment_fields(
    config: ProjectConfig,
    field: ChemicalField | None = None,
    *,
    show_progress: bool = False,
) -> MomentSolution:
    """Integrate the polar-closed system and return the unsmoothed fields."""

    field = create_field(config.field) if field is None else field
    (
        plan,
        coordinates,
        grid_points,
        mask,
        _neighbors,
        edge_left,
        edge_right,
        edge_axis,
        eta,
        gradient,
    ) = build_moment_plan(config, field)
    initial_grid = prescribed_initial_density_grid(
        config, field, grid_points, mask, plan.spacing
    )
    initial_density = initial_grid[mask]
    if config.density.initial_heading_distribution == "isotropic":
        heading_x = 0.0
        heading_y = 0.0
    else:
        heading_x = float(np.cos(config.initial.heading))
        heading_y = float(np.sin(config.initial.heading))
    initial_px = initial_density * heading_x
    initial_py = initial_density * heading_y
    compact_gradient = gradient[mask]

    from .accelerated import configure_threads

    configure_threads(config, len(PDF_CASES))
    bar = progress_bar(
        total=len(PDF_CASES),
        description="Cases 0-9 density-polarity PDE",
        unit="case",
        enabled=show_progress,
    )
    compact_rho, compact_px, compact_py = _integrate_moment_system(
        initial_density,
        initial_px,
        initial_py,
        edge_left,
        edge_right,
        edge_axis,
        (eta - disk_area_average(field, config.domain.radius))[mask],
        compact_gradient[:, 0],
        compact_gradient[:, 1],
        _case_coefficients(config),
        config.kinetics.v0,
        config.kinetics.D_T0,
        config.kinetics.D_R0,
        config.kinetics.lambda0,
        create_turning_kernel(config.turning).P_turn,
        plan.spacing,
        plan.dt,
        plan.save_steps,
    )
    bar.update(len(PDF_CASES))
    bar.close()
    if not (
        np.all(np.isfinite(compact_rho))
        and np.all(np.isfinite(compact_px))
        and np.all(np.isfinite(compact_py))
    ):
        raise RuntimeError(
            "the density--polarity moment solver produced non-finite fields"
        )
    minimum = float(np.min(compact_rho))
    if minimum < -2.0e-5:
        raise RuntimeError(
            f"the density--polarity moment solver produced rho_min={minimum:.6g}"
        )
    times = plan.save_steps.astype(float) * plan.dt
    times[-1] = config.simulation.duration
    return MomentSolution(
        plan=plan,
        times=times,
        coordinates=coordinates,
        points=grid_points[mask],
        mask=mask,
        spacing=plan.spacing,
        density=compact_rho,
        polarity_x=compact_px,
        polarity_y=compact_py,
        initial_grid=initial_grid,
        initial_heading=(heading_x, heading_y),
    )


def compute_moment_pdf(
    config: ProjectConfig,
    *,
    field: ChemicalField | None = None,
    show_progress: bool = False,
) -> PositionPDFResult:
    """Solve the density--polarity PDE with the manuscript's ``Q = 0`` closure."""

    field = create_field(config.field) if field is None else field
    solution = integrate_moment_fields(config, field, show_progress=show_progress)
    plan = solution.plan
    coordinates = solution.coordinates
    mask = solution.mask
    initial_grid = solution.initial_grid
    heading_x, heading_y = solution.initial_heading
    compact_rho = solution.density
    compact_px = solution.polarity_x
    compact_py = solution.polarity_y

    shape = (len(PDF_CASES), len(plan.save_steps), plan.grid_size, plan.grid_size)
    raw_density = np.zeros(shape, dtype=np.float64)
    raw_px = np.zeros(shape, dtype=np.float64)
    raw_py = np.zeros(shape, dtype=np.float64)
    raw_density[:, :, mask] = compact_rho
    raw_px[:, :, mask] = compact_px
    raw_py[:, :, mask] = compact_py
    masses = np.sum(raw_density, axis=(-2, -1), dtype=np.float64) * plan.spacing**2
    if not np.allclose(masses, 1.0, rtol=0.0, atol=3.0e-6):
        raise RuntimeError("the conservative density--polarity solver lost mass")

    settings = replace(config.density, grid_size=plan.grid_size)
    density, polarity_x, polarity_y, output_coordinates = _smooth_moment_fields(
        raw_density, raw_px, raw_py, settings
    )
    if not np.array_equal(coordinates, output_coordinates):
        raise RuntimeError("moment smoothing produced an inconsistent grid")
    difference_bandwidth = resolve_difference_bandwidth(
        settings, inferred_characteristic_length(config, field)
    )
    if np.isclose(difference_bandwidth, settings.bandwidth_fraction, atol=1.0e-15):
        difference_density = density
        difference_px = polarity_x
        difference_py = polarity_y
    else:
        difference_settings = replace(settings, bandwidth_fraction=difference_bandwidth)
        difference_density, difference_px, difference_py, difference_coordinates = (
            _smooth_moment_fields(raw_density, raw_px, raw_py, difference_settings)
        )
        if not np.array_equal(coordinates, difference_coordinates):
            raise RuntimeError(
                "moment difference smoothing produced an inconsistent grid"
            )

    # Preserve the literal common initial state after observation smoothing.
    initial_density_observed, _, initial_normalization = _normalized_smoothed(
        initial_grid[None, None], settings
    )
    density[:, 0] = initial_density_observed[0, 0]
    polarity_x[:, 0] = density[:, 0] * heading_x
    polarity_y[:, 0] = density[:, 0] * heading_y
    del initial_normalization
    if difference_density is density:
        initial_difference = initial_density_observed
    else:
        initial_difference, _, _ = _normalized_smoothed(
            initial_grid[None, None], difference_settings
        )
        difference_density[:, 0] = initial_difference[0, 0]
    difference_px[:, 0] = difference_density[:, 0] * heading_x
    difference_py[:, 0] = difference_density[:, 0] * heading_y

    provenance = build_provenance(str(config.source_path))
    return PositionPDFResult(
        times=solution.times,
        coordinates=coordinates,
        density=density,
        case_numbers=PDF_CASES,
        ensemble_size=0,
        bandwidth_fraction=settings.bandwidth_fraction,
        difference_density=difference_density,
        difference_bandwidth_fraction=difference_bandwidth,
        config_source=str(config.source_path),
        seed=config.simulation.seed,
        dt=plan.dt,
        method="moment",
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
