"""Macroscopic density--polarity system with the polar closure ``Q=0``.

The public ``macro`` strategy follows ``manuscript/main.tex`` and evolves density
and both components of polarity.  The older scalar Scharfetter--Gummel helpers
remain below as explicitly named investigations of the additional adiabatic
reduction, but they are not used by :func:`compute_macroscopic_pdf`.
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
    prescribed_initial_density_grid,
    resolve_difference_bandwidth,
    smooth_density_stack,
)
from .fields import ChemicalField, create_field
from .kinetics import macroscopic_coefficients
from .numerics import inferred_characteristic_length
from .progress import progress_bar
from .provenance import build_provenance
from .turning import create_turning_kernel


@dataclass(frozen=True)
class MacroscopicPlan:
    """Numerical plan for the finite-volume density equation."""

    grid_size: int
    spacing: float
    effective_diffusivity: float
    mobilities: tuple[float, ...]
    diffusion_dt: float
    drift_dt: float
    positivity_dt: float
    safe_dt: float
    dt: float
    steps: int
    save_steps: NDArray[np.int64]
    max_outgoing_rate: float


def _bernoulli(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stable Bernoulli function ``B(z)=z/(exp(z)-1)``."""

    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    small = np.abs(values) < 1.0e-5
    positive_large = values > 50.0
    negative_large = values < -50.0
    regular = ~(small | positive_large | negative_large)
    z = values[small]
    result[small] = 1.0 - z / 2.0 + z**2 / 12.0 - z**4 / 720.0
    z = values[positive_large]
    result[positive_large] = z * np.exp(-z)
    result[negative_large] = -values[negative_large]
    z = values[regular]
    result[regular] = z / np.expm1(z)
    return result


def _macro_grid_size(config: ProjectConfig, field: ChemicalField) -> int:
    """Resolve the chemical field and the narrowest reported KDE."""

    field_length = inferred_characteristic_length(config, field)
    target_spacing = min(
        field_length / config.numerics.scan_points_per_length,
        config.density.bandwidth_fraction / 2.0,
    )
    grid_size = max(config.density.grid_size, ceil(2.0 / target_spacing))
    if grid_size > config.numerics.max_scan_grid:
        raise ValueError(
            f"the macroscopic density requires a {grid_size}x{grid_size} grid "
            f"to reach Delta(x)<={target_spacing:.4g}; increase "
            "numerics.max_scan_grid or use a smoother field/initial density"
        )
    return grid_size


def _disk_grid(
    grid_size: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.int64],
]:
    spacing = 2.0 / grid_size
    coordinates = -1.0 + (np.arange(grid_size) + 0.5) * spacing
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    mask = x_grid * x_grid + y_grid * y_grid <= 1.0
    active_ids = np.full(mask.shape, -1, dtype=np.int64)
    active_ids[mask] = np.arange(np.count_nonzero(mask), dtype=np.int64)
    return coordinates, np.stack((x_grid, y_grid), axis=-1), mask, active_ids


def _closed_coefficients(
    config: ProjectConfig,
) -> tuple[float, tuple[float, ...]]:
    turning_kernel = create_turning_kernel(config.turning)
    gamma1 = config.kinetics.D_R0 + config.kinetics.lambda0 * (
        1.0 - turning_kernel.P_turn
    )
    if gamma1 <= 0.0:
        raise ValueError(
            "--macro requires gamma1 = D_R0 + lambda0(1-P_turn) > 0 because the "
            "closed long-time density equation in Theory.pdf assumes angular "
            "relaxation. These zero values remain valid with --micro."
        )
    gamma2 = 4.0 * config.kinetics.D_R0 + config.kinetics.lambda0 * (
        1.0 - turning_kernel.P_2_turn
    )
    if gamma2 <= 0.0:
        raise ValueError(
            "--macro requires gamma2 = 4 D_R0 + lambda0(1-P_2_turn) > 0 "
            "because the manuscript closure also eliminates the nematic mode. "
            "These parameters remain valid with --micro."
        )
    coefficients = [
        macroscopic_coefficients(
            config.kinetics,
            config.responses.only_case(case_number),
            turning_kernel.P_turn,
        )
        for case_number in PDF_CASES
    ]
    effective_diffusivities = np.asarray(
        [coefficient[1] for coefficient in coefficients],
        dtype=float,
    )
    if not np.allclose(
        effective_diffusivities,
        effective_diffusivities[0],
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise RuntimeError("the closed cases produced inconsistent D_eff values")
    effective_diffusivity = float(effective_diffusivities[0])
    mobilities = tuple(float(coefficient[2]) for coefficient in coefficients)
    if not np.isfinite(effective_diffusivity) or effective_diffusivity <= 0.0:
        raise ValueError("the macroscopic effective diffusivity must be positive")
    if not np.all(np.isfinite(mobilities)):
        raise ValueError("the macroscopic mobilities must be finite")
    return effective_diffusivity, mobilities


def _transport_rates(
    eta: NDArray[np.float64],
    mask: NDArray[np.bool_],
    active_ids: NDArray[np.int64],
    spacing: float,
    effective_diffusivity: float,
    mobilities: tuple[float, ...],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Build neighbor, incoming-rate, and outgoing-rate arrays."""

    case_count = len(mobilities)
    active_count = int(np.count_nonzero(mask))
    neighbors = np.full((active_count, 4), -1, dtype=np.int64)
    incoming = np.zeros((case_count, active_count, 4), dtype=float)
    outgoing = np.zeros((case_count, active_count), dtype=float)
    mobility_array = np.asarray(mobilities, dtype=float)[:, None, None]

    horizontal = mask[:, :-1] & mask[:, 1:]
    delta_x = eta[:, 1:] - eta[:, :-1]
    pe_x = mobility_array * delta_x[None] / effective_diffusivity
    left_to_right = effective_diffusivity / spacing**2 * _bernoulli(-pe_x)
    right_to_left = effective_diffusivity / spacing**2 * _bernoulli(pe_x)
    rows, columns = np.nonzero(horizontal)
    left_ids = active_ids[rows, columns]
    right_ids = active_ids[rows, columns + 1]
    neighbors[left_ids, 1] = right_ids
    neighbors[right_ids, 0] = left_ids
    for case_index in range(case_count):
        lr = left_to_right[case_index, rows, columns]
        rl = right_to_left[case_index, rows, columns]
        outgoing[case_index, left_ids] += lr
        incoming[case_index, right_ids, 0] = lr
        outgoing[case_index, right_ids] += rl
        incoming[case_index, left_ids, 1] = rl

    vertical = mask[:-1, :] & mask[1:, :]
    delta_y = eta[1:, :] - eta[:-1, :]
    pe_y = mobility_array * delta_y[None] / effective_diffusivity
    bottom_to_top = effective_diffusivity / spacing**2 * _bernoulli(-pe_y)
    top_to_bottom = effective_diffusivity / spacing**2 * _bernoulli(pe_y)
    rows, columns = np.nonzero(vertical)
    bottom_ids = active_ids[rows, columns]
    top_ids = active_ids[rows + 1, columns]
    neighbors[bottom_ids, 3] = top_ids
    neighbors[top_ids, 2] = bottom_ids
    for case_index in range(case_count):
        bt = bottom_to_top[case_index, rows, columns]
        tb = top_to_bottom[case_index, rows, columns]
        outgoing[case_index, bottom_ids] += bt
        incoming[case_index, top_ids, 2] = bt
        outgoing[case_index, top_ids] += tb
        incoming[case_index, bottom_ids, 3] = tb

    return neighbors, incoming, outgoing


def _initial_density(
    config: ProjectConfig,
    field: ChemicalField,
    grid_points: NDArray[np.float64],
    mask: NDArray[np.bool_],
    spacing: float,
) -> NDArray[np.float64]:
    """Represent the configured initial spatial ensemble on finite volumes."""

    return prescribed_initial_density_grid(
        config,
        field,
        grid_points,
        mask,
        spacing,
    )


@njit(cache=True, parallel=True)
def _integrate_compact_density(
    initial: NDArray[np.float64],
    neighbors: NDArray[np.int64],
    incoming: NDArray[np.float64],
    outgoing: NDArray[np.float64],
    dt: float,
    save_steps: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Advance all cases in parallel with a positivity-preserving Euler step."""

    case_count, active_count = outgoing.shape
    saved = np.empty(
        (case_count, len(save_steps), active_count),
        dtype=np.float64,
    )
    for case_index in prange(case_count):
        current = initial.copy()
        following = np.empty_like(current)
        for cell in range(active_count):
            saved[case_index, 0, cell] = current[cell]
        save_index = 1
        final_step = save_steps[-1]
        for step in range(1, final_step + 1):
            for cell in range(active_count):
                value = (1.0 - dt * outgoing[case_index, cell]) * current[cell]
                for direction in range(4):
                    neighbor = neighbors[cell, direction]
                    if neighbor >= 0:
                        value += (
                            dt
                            * incoming[case_index, cell, direction]
                            * current[neighbor]
                        )
                following[cell] = value
            swap = current
            current = following
            following = swap
            if save_index < len(save_steps) and step == save_steps[save_index]:
                for cell in range(active_count):
                    saved[case_index, save_index, cell] = current[cell]
                save_index += 1
    return saved


def build_equilibrium_plan(
    config: ProjectConfig,
    field: ChemicalField | None = None,
) -> tuple[
    MacroscopicPlan,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Build the grid, transport generator, and optimal stable timestep."""

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

    effective_diffusivity, mobilities = _closed_coefficients(config)
    neighbors, incoming, outgoing = _transport_rates(
        eta,
        mask,
        active_ids,
        spacing,
        effective_diffusivity,
        mobilities,
    )
    max_outgoing_rate = float(np.max(outgoing))
    positivity_dt = 1.0 / max_outgoing_rate
    diffusion_dt = spacing**2 / (4.0 * effective_diffusivity)
    drift_speed = np.max(
        np.abs(np.asarray(mobilities))[:, None]
        * np.sum(np.abs(gradient[mask]), axis=-1)[None],
    )
    drift_dt = np.inf if drift_speed == 0.0 else spacing / float(drift_speed)
    safe_dt = config.macro.cfl_safety * positivity_dt

    requested_dt = config.simulation.dt
    if isinstance(requested_dt, str):
        target_dt = safe_dt
    else:
        if requested_dt > safe_dt * (1.0 + 1.0e-12):
            raise ValueError(
                f"the requested macro dt={requested_dt:.6g} exceeds the "
                f"finite-volume safe limit {safe_dt:.6g}"
            )
        target_dt = requested_dt
    steps = max(1, ceil(config.simulation.duration / target_dt))
    dt = config.simulation.duration / steps
    save_count = min(config.density.time_slices, steps + 1)
    save_steps = np.unique(
        np.linspace(0, steps, save_count, dtype=np.int64),
    )
    if save_steps[-1] != steps:
        save_steps = np.append(save_steps, steps)
    plan = MacroscopicPlan(
        grid_size=grid_size,
        spacing=spacing,
        effective_diffusivity=effective_diffusivity,
        mobilities=mobilities,
        diffusion_dt=diffusion_dt,
        drift_dt=drift_dt,
        positivity_dt=positivity_dt,
        safe_dt=safe_dt,
        dt=dt,
        steps=steps,
        save_steps=save_steps,
        max_outgoing_rate=max_outgoing_rate,
    )
    return plan, coordinates, grid_points, mask, neighbors, incoming, outgoing


def equilibrium_timestep_report(plan: MacroscopicPlan, config: ProjectConfig) -> str:
    """Human-readable formulas and values used to select the macro timestep."""

    drift = "infinity" if np.isinf(plan.drift_dt) else f"{plan.drift_dt:.6g}"
    mobility_values = ", ".join(
        f"C{case}={mobility:.6g}"
        for case, mobility in zip(PDF_CASES, plan.mobilities, strict=True)
    )
    return (
        "\nClosed density evolution (Theory.pdf, equations 16-20)\n"
        "  partial_t rho = div(D_eff grad(rho) - kappa rho grad(eta)),  n.J_rho = 0\n"
        f"  grid={plan.grid_size}x{plan.grid_size}, Delta(x)={plan.spacing:.6g}\n"
        "  The grid satisfies both the narrowest-field resolution target "
        "and at least two cells per reported KDE width.\n"
        f"  D_eff={plan.effective_diffusivity:.6g}\n"
        f"  kappa: {mobility_values}\n"
        "\n"
        "  Process                 Candidate formula                         dt\n"
        "  ----------------------  ---------------------------------------  -----------\n"
        f"  diffusion               Delta(x)^2 / (4 D_eff)                 "
        f"{plan.diffusion_dt:.6g}\n"
        "  nutrient drift          Delta(x) / max(|kappa| |grad(eta)|_1)   "
        f"{drift}\n"
        "  combined FV positivity  1 / max_cell(sum outgoing face rates)  "
        f"{plan.positivity_dt:.6g}\n"
        f"  Selected safe limit     {config.macro.cfl_safety:.3g} x "
        f"combined FV positivity            {plan.safe_dt:.6g}\n"
        f"  Simulation dt           t_final / ceil(t_final/safe limit)      "
        f"{plan.dt:.6g}\n"
        f"  Steps                    {plan.steps:,} to reach "
        f"t_final={config.simulation.duration:.6g}"
    )


def compute_equilibrium_pdf(
    config: ProjectConfig,
    *,
    field: ChemicalField | None = None,
    show_progress: bool = False,
) -> PositionPDFResult:
    """Solve the closed dimensionless density equation for Cases 0--9."""

    field = create_field(config.field) if field is None else field
    (
        plan,
        coordinates,
        grid_points,
        mask,
        neighbors,
        incoming,
        outgoing,
    ) = build_equilibrium_plan(config, field)

    initial_grid = _initial_density(config, field, grid_points, mask, plan.spacing)
    initial = initial_grid[mask]
    from .accelerated import configure_threads

    configure_threads(config, len(PDF_CASES))
    bar = progress_bar(
        total=len(PDF_CASES),
        description="Cases 0-9 closed density equation",
        unit="case",
        enabled=show_progress,
    )
    compact = _integrate_compact_density(
        initial,
        neighbors,
        incoming,
        outgoing,
        plan.dt,
        plan.save_steps,
    )
    bar.update(len(PDF_CASES))
    bar.close()

    minimum = float(np.min(compact))
    if minimum < -2.0e-6:
        raise RuntimeError(
            f"the positivity-preserving macro solver produced p_min={minimum:.6g}"
        )
    raw_density = np.zeros(
        (
            len(PDF_CASES),
            len(plan.save_steps),
            plan.grid_size,
            plan.grid_size,
        ),
        dtype=np.float64,
    )
    raw_density[:, :, mask] = compact
    masses = np.sum(raw_density, axis=(-2, -1), dtype=np.float64) * plan.spacing**2
    if not np.allclose(masses, 1.0, rtol=0.0, atol=2.0e-6):
        raise RuntimeError(
            "the conservative macroscopic solver failed its unit-mass check"
        )
    if not np.all(np.isfinite(raw_density)):
        raise RuntimeError("the macroscopic probability density is not finite")

    density_settings = replace(
        config.density,
        grid_size=plan.grid_size,
    )
    density, smoothed_coordinates = smooth_density_stack(
        raw_density,
        density_settings,
    )
    if not np.array_equal(coordinates, smoothed_coordinates):
        raise RuntimeError("macroscopic KDE produced an inconsistent grid")
    initial_density, initial_coordinates = smooth_density_stack(
        initial_grid[None, None],
        density_settings,
    )
    if not np.array_equal(coordinates, initial_coordinates):
        raise RuntimeError(
            "macroscopic initial-density estimator produced an inconsistent grid"
        )
    density[:, 0] = initial_density[0, 0]
    difference_bandwidth = resolve_difference_bandwidth(
        density_settings,
        inferred_characteristic_length(config, field),
    )
    if np.isclose(
        difference_bandwidth,
        density_settings.bandwidth_fraction,
        rtol=0.0,
        atol=1.0e-15,
    ):
        difference_density = density
    else:
        difference_settings = replace(
            density_settings,
            bandwidth_fraction=difference_bandwidth,
        )
        difference_density, difference_coordinates = smooth_density_stack(
            raw_density,
            difference_settings,
        )
        if not np.array_equal(coordinates, difference_coordinates):
            raise RuntimeError(
                "macroscopic difference KDE produced an inconsistent grid"
            )
        initial_difference_density, initial_difference_coordinates = (
            smooth_density_stack(
                initial_grid[None, None],
                difference_settings,
            )
        )
        if not np.array_equal(coordinates, initial_difference_coordinates):
            raise RuntimeError(
                "macroscopic difference initial-density estimator produced an "
                "inconsistent grid"
            )
        difference_density[:, 0] = initial_difference_density[0, 0]

    times = plan.save_steps.astype(float) * plan.dt
    times[-1] = config.simulation.duration
    provenance = build_provenance(str(config.source_path))
    return PositionPDFResult(
        times=times,
        coordinates=coordinates,
        density=density,
        case_numbers=PDF_CASES,
        ensemble_size=0,
        bandwidth_fraction=density_settings.bandwidth_fraction,
        difference_density=difference_density,
        difference_bandwidth_fraction=difference_bandwidth,
        config_source=str(config.source_path),
        seed=config.simulation.seed,
        dt=plan.dt,
        method="macro",
        schema_version=provenance.schema_version,
        software_version=provenance.software_version,
        config_text=provenance.config_text,
        config_sha256=provenance.config_sha256,
        runtime_versions=provenance.runtime_versions,
    )


# ---------------------------------------------------------------------------
# Current manuscript public API
# ---------------------------------------------------------------------------
def build_macroscopic_plan(
    config: ProjectConfig,
    field: ChemicalField | None = None,
):
    """Build the finite-volume plan for the dynamic ``(rho,p)`` system."""

    # Imported lazily because moment.py reuses the disk-grid helpers above.
    from .moment import build_moment_plan

    return build_moment_plan(config, field)


def macroscopic_timestep_report(plan, config: ProjectConfig) -> str:
    """Describe the macroscopic polar-closed timestep and retained fields."""

    from .moment import moment_timestep_report

    return moment_timestep_report(plan, config).replace(
        "Density--polarity moment evolution",
        "Macroscopic density--polarity evolution",
        1,
    )


def compute_macroscopic_pdf(
    config: ProjectConfig,
    *,
    field: ChemicalField | None = None,
    show_progress: bool = False,
) -> PositionPDFResult:
    """Solve the manuscript macroscopic equations with dynamic ``p`` and ``Q=0``."""

    from .moment import compute_moment_pdf

    result = compute_moment_pdf(config, field=field, show_progress=show_progress)
    return replace(result, method="macro")
