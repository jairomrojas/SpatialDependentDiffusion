"""Nondimensional scales and parameter-aware resolution selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, inf

import numpy as np

from .config import ProjectConfig
from .fields import ChemicalField
from .fields.analytic import GaussianField, LinearField, MultiGaussianField


@dataclass(frozen=True)
class NondimensionalScales:
    """Input units used to define dimensionless ``x=r/R`` and ``t=t_phys/t_D``."""

    length: float
    time: float

    def position(self, value):
        return np.asarray(value, dtype=float) / self.length

    def time_coordinate(self, value):
        return np.asarray(value, dtype=float) / self.time


@dataclass(frozen=True)
class ResolutionPlan:
    dt: float
    dt_limit: float
    limiting_constraint: str
    scan_grid: int
    field_grid: int
    characteristic_length: float
    target_spatial_step: float
    ranges_by_case: dict[int, dict[str, tuple[float, float]]]
    constraints: dict[str, float]
    scales: NondimensionalScales

    @property
    def dt_star(self) -> float:
        return self.dt

    @property
    def dimensional_dt(self) -> float:
        """Input-unit timestep used only to interface with dimensional settings."""

        return self.dt * self.scales.time

    @property
    def characteristic_length_star(self) -> float:
        return self.characteristic_length / self.scales.length

    @property
    def grid_spacing_star(self) -> float:
        return 2.0 / (self.field_grid - 1)


def nondimensional_scales(config: ProjectConfig) -> NondimensionalScales:
    radius = config.domain.radius
    return NondimensionalScales(
        length=radius,
        time=radius * radius / config.kinetics.D_T0,
    )


def inferred_characteristic_length(
    config: ProjectConfig,
    field: ChemicalField,
) -> float:
    requested = config.numerics.characteristic_length
    if requested > 0:
        return min(config.domain.radius, requested)
    if isinstance(field, GaussianField):
        return min(config.domain.radius, *field.sigma)
    if isinstance(field, MultiGaussianField):
        widths = [width for source in field.sources for width in source.sigma]
        return min(config.domain.radius, *widths)
    if isinstance(field, LinearField):
        return config.domain.radius
    raise ValueError(
        "an automatic accuracy check cannot infer the smallest length of a "
        "custom field; set numerics.characteristic_length"
    )


def _required_grid(config: ProjectConfig, characteristic_length: float) -> int:
    settings = config.numerics
    intervals = ceil(
        2.0
        * config.domain.radius
        * settings.scan_points_per_length
        / characteristic_length
    )
    grid = max(settings.min_scan_grid, intervals + 1)
    if grid % 2 == 0:
        grid += 1
    if grid > settings.max_scan_grid:
        required_spacing = 2.0 * config.domain.radius / (grid - 1)
        raise ValueError(
            f"the inputs require a {grid}x{grid} disk scan "
            f"(spacing {required_spacing:.4g}) to resolve the smallest length "
            f"{characteristic_length:.4g}; increase numerics.max_scan_grid or "
            "provide a smoother field"
        )
    return grid


def _sample_field_in_disk(
    field: ChemicalField,
    radius: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.linspace(-radius, radius, grid_size)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    mask = x_grid * x_grid + y_grid * y_grid <= radius * radius
    positions = np.stack((x_grid[mask], y_grid[mask]), axis=-1)
    concentration, gradient = field.value_and_gradient(positions)
    return (
        np.asarray(concentration, dtype=float).reshape(-1),
        np.asarray(gradient, dtype=float).reshape(-1, 2),
    )


def _signal_bounds(
    config: ProjectConfig,
    field: ChemicalField,
    concentration: np.ndarray,
    gradient: np.ndarray,
) -> tuple[float, float, float]:
    """Conservative ``delta_c`` and ``|grad(c)|`` bounds when analytic."""

    if isinstance(field, GaussianField):
        delta_min = min(0.0, field.amplitude)
        delta_max = max(0.0, field.amplitude)
        gradient_max = abs(field.amplitude) * np.exp(-0.5) / min(field.sigma)
        return delta_min, delta_max, float(gradient_max)

    if isinstance(field, MultiGaussianField):
        delta_min = sum(min(0.0, source.amplitude) for source in field.sources)
        delta_max = sum(max(0.0, source.amplitude) for source in field.sources)
        gradient_max = sum(
            abs(source.amplitude) * np.exp(-0.5) / min(source.sigma)
            for source in field.sources
        )
        return float(delta_min), float(delta_max), float(gradient_max)

    if isinstance(field, LinearField):
        gradient_vector = np.asarray(field.gradient_vector, dtype=float)
        gradient_max = float(np.linalg.norm(gradient_vector))
        center_shift = -float(
            np.dot(np.asarray(field.center, dtype=float), gradient_vector)
        )
        return (
            center_shift - config.domain.radius * gradient_max,
            center_shift + config.domain.radius * gradient_max,
            gradient_max,
        )

    delta = concentration - field.reference_value
    # A declared characteristic length plus N points across it controls the
    # custom-field scan. The margin acknowledges extrema between grid points.
    margin = 1.0 / config.numerics.scan_points_per_length
    delta_span = float(np.max(delta) - np.min(delta))
    return (
        float(np.min(delta) - margin * delta_span),
        float(np.max(delta) + margin * delta_span),
        float((1.0 + margin) * np.max(np.linalg.norm(gradient, axis=1))),
    )


def scan_case_from_samples(
    config: ProjectConfig,
    field: ChemicalField,
    case_number: int,
    concentration: np.ndarray,
    gradient: np.ndarray,
    signal_bounds: tuple[float, float, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Bound every kinetic coefficient over sampled positions and all headings."""

    response = config.responses.only_case(case_number)
    if signal_bounds is None:
        signal_bounds = _signal_bounds(config, field, concentration, gradient)
    delta_min, delta_max, gradient_max = signal_bounds
    base = config.kinetics
    coefficients = {
        "v": (base.v0, response.v_c, response.v_S),
        "D_T": (base.D_T0, response.D_T_c, response.D_T_S),
        "D_R": (base.D_R0, response.D_R_c, response.D_R_S),
        "lambda": (base.lambda0, response.lambda_c, response.lambda_S),
    }
    result: dict[str, tuple[float, float]] = {}
    for name, (
        background,
        concentration_response,
        alignment_response,
    ) in coefficients.items():
        concentration_values = (
            background + concentration_response * delta_min,
            background + concentration_response * delta_max,
        )
        half_width = abs(alignment_response) * gradient_max
        result[name] = (
            float(min(concentration_values) - half_width),
            float(max(concentration_values) + half_width),
        )
    return result


def build_resolution_plan(
    config: ProjectConfig,
    field: ChemicalField,
    case_numbers: tuple[int, ...] | list[int],
) -> ResolutionPlan:
    """Scan requested cases and select a timestep satisfying every target."""

    cases = tuple(dict.fromkeys(int(case) for case in case_numbers))
    if not cases:
        raise ValueError("at least one case is required for resolution selection")

    length = inferred_characteristic_length(config, field)
    scan_grid = _required_grid(config, length)
    concentration, gradient = _sample_field_in_disk(
        field, config.domain.radius, scan_grid
    )
    signal_bounds = _signal_bounds(config, field, concentration, gradient)
    ranges_by_case = {
        case: scan_case_from_samples(
            config, field, case, concentration, gradient, signal_bounds
        )
        for case in cases
    }

    for case, ranges in ranges_by_case.items():
        for name, bounds in ranges.items():
            if bounds[0] < 0:
                raise ValueError(
                    f"case {case} produces negative {name}={bounds[0]:.6g}; "
                    "the linear kinetic law is not physical for these inputs"
                )

    scales = nondimensional_scales(config)
    v_max = (
        max(bounds["v"][1] for bounds in ranges_by_case.values())
        * scales.length
        / config.kinetics.D_T0
    )
    D_T_max = (
        max(bounds["D_T"][1] for bounds in ranges_by_case.values())
        / config.kinetics.D_T0
    )
    D_R_max = max(bounds["D_R"][1] for bounds in ranges_by_case.values()) * scales.time
    lambda_max = (
        max(bounds["lambda"][1] for bounds in ranges_by_case.values()) * scales.time
    )
    target_step = config.numerics.max_spatial_step_fraction * length / scales.length

    constraints = {
        "advection": target_step / v_max if v_max > 0 else inf,
        # sqrt(E|Delta r_T|^2) = sqrt(4 D_T dt) in two dimensions.
        "translational diffusion": (
            target_step * target_step / (4.0 * D_T_max) if D_T_max > 0 else inf
        ),
        "rotational diffusion": (
            config.numerics.max_rotation_rms**2 / (2.0 * D_R_max)
            if D_R_max > 0
            else inf
        ),
        "tumbling": (
            config.simulation.max_lambda_dt / lambda_max if lambda_max > 0 else inf
        ),
    }
    limiting_constraint = min(constraints, key=constraints.get)
    dt_limit = constraints[limiting_constraint]

    configured_dt = config.simulation.dt
    automatic_dt = isinstance(configured_dt, str)
    duration = config.simulation.duration / scales.time
    if automatic_dt:
        step_count = max(1, ceil(duration / dt_limit))
        dt = duration / step_count
    else:
        dt = float(configured_dt) / scales.time
        if dt > dt_limit * (1.0 + 1.0e-12):
            raise ValueError(
                f"dimensionless simulation.dt={dt:.6g} is too large for the requested "
                f"accuracy; {limiting_constraint} requires dt <= "
                f"{dt_limit:.6g}. Use dt='auto' or reduce dt."
            )
        step_count = round(duration / dt)
        if not np.isclose(
            step_count * dt,
            duration,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "simulation.duration must be an integer multiple of a manual "
                "simulation.dt; dt='auto' adjusts it automatically"
            )

    required_field_grid = scan_grid
    configured_field_grid = config.video.field_grid
    if isinstance(configured_field_grid, str):
        field_grid = required_field_grid
    else:
        field_grid = int(configured_field_grid)
        if field_grid < required_field_grid:
            raise ValueError(
                f"video.field_grid={field_grid} under-resolves the smallest "
                f"length {length:.6g}; use field_grid='auto' or at least "
                f"{required_field_grid}"
            )

    return ResolutionPlan(
        dt=dt,
        dt_limit=dt_limit,
        limiting_constraint=limiting_constraint,
        scan_grid=scan_grid,
        field_grid=field_grid,
        characteristic_length=length,
        target_spatial_step=target_step,
        ranges_by_case=ranges_by_case,
        constraints=constraints,
        scales=scales,
    )


def resolve_numerics(
    config: ProjectConfig,
    field: ChemicalField,
    case_numbers: tuple[int, ...] | list[int],
) -> tuple[ProjectConfig, ResolutionPlan | None]:
    """Return a numeric config; repeated calls on a resolved config are free."""

    if config.numerics.resolved:
        return config, None
    plan = build_resolution_plan(config, field, case_numbers)
    resolved = replace(
        config,
        simulation=replace(
            config.simulation,
            dt=plan.dimensional_dt,
        ),
        video=replace(config.video, field_grid=plan.field_grid),
        numerics=replace(config.numerics, resolved=True),
    )
    resolved.validate()
    return resolved, plan
