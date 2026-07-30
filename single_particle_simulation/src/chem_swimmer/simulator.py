"""Euler-Maruyama simulation of the complete jump-diffusion in equation (6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .angles import heading_vector, sample_tumble_heading, wrap_angle
from .boundary import advance_in_disk
from .cases import get_case
from .config import ProjectConfig
from .fields import ChemicalField, create_field
from .kinetics import KineticLaw, KineticState
from .noise import Innovations
from .numerics import resolve_numerics
from .progress import progress_bar


@dataclass(frozen=True)
class Trajectory:
    case_number: int
    case_title: str
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    headings: NDArray[np.float64]
    concentration: NDArray[np.float64]
    alignment: NDArray[np.float64]
    speed: NDArray[np.float64]
    translational_diffusivity: NDArray[np.float64]
    rotational_diffusivity: NDArray[np.float64]
    tumble_rate: NDArray[np.float64]
    tumble_counts: NDArray[np.int64]
    wall_hit_counts: NDArray[np.int64]

    @property
    def duration(self) -> float:
        return float(self.times[-1])


def _save_indices(steps: int, save_every: int) -> NDArray[np.int64]:
    indices = np.arange(0, steps + 1, save_every, dtype=np.int64)
    if indices[-1] != steps:
        indices = np.append(indices, steps)
    return indices


def _evaluate(
    field: ChemicalField,
    law: KineticLaw,
    position: NDArray[np.float64],
    heading: float,
) -> KineticState:
    concentration, gradient = field.value_and_gradient(position)
    return law.evaluate(
        float(concentration),
        np.asarray(gradient, dtype=float),
        heading,
        position,
    )


def _evaluate_dimensionless(
    field: ChemicalField,
    law: KineticLaw,
    position_star: NDArray[np.float64],
    heading: float,
    radius: float,
    time_scale: float,
    D_T0: float,
) -> KineticState:
    """Evaluate the input law and return coefficients in computational units.

    Public configurations are already dimensionless and enforce ``R=D_T0=1``.
    The explicit factors are retained here only to keep the nondimensional
    transformation auditable and to avoid hiding a future change of input API.
    """

    state = _evaluate(field, law, radius * position_star, heading)
    return KineticState(
        concentration=state.concentration,
        delta_c=state.delta_c,
        gradient=radius * state.gradient,
        alignment=radius * state.alignment,
        v=state.v * radius / D_T0,
        D_T=state.D_T / D_T0,
        D_R=state.D_R * time_scale,
        tumble_rate=state.tumble_rate * time_scale,
    )


def _simulate_python(
    config: ProjectConfig,
    case_number: int,
    *,
    field: ChemicalField | None = None,
    innovations: Innovations | None = None,
    show_progress: bool = False,
) -> Trajectory:
    """Simulate one response case.

    All response coefficients except the selected canonical one are set to
    zero. Case 0 is the chemically insensitive baseline.
    """

    case = get_case(case_number)
    field = create_field(config.field) if field is None else field
    responses = config.responses.only_case(case_number)
    law = KineticLaw(config.kinetics, responses, field.reference_value)

    settings = config.simulation
    if isinstance(settings.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    steps = round(settings.duration / settings.dt)
    if not np.isclose(steps * settings.dt, settings.duration, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "simulation.duration must be an integer multiple of simulation.dt"
        )
    innovations = (
        Innovations.generate(steps, settings.seed)
        if innovations is None
        else innovations
    )
    if len(innovations.rotation) != steps:
        raise ValueError("the supplied innovation stream has the wrong number of steps")

    save_steps = _save_indices(steps, settings.save_every)
    number_saved = len(save_steps)
    radius = config.domain.radius
    time_scale = radius * radius / config.kinetics.D_T0
    dt_star = settings.dt / time_scale
    times = save_steps.astype(float) * dt_star
    times[-1] = settings.duration / time_scale
    positions = np.empty((number_saved, 2))
    headings = np.empty(number_saved)
    concentration = np.empty(number_saved)
    alignment = np.empty(number_saved)
    speed = np.empty(number_saved)
    D_T = np.empty(number_saved)
    D_R = np.empty(number_saved)
    tumble_rate = np.empty(number_saved)
    tumble_counts = np.zeros(number_saved, dtype=np.int64)
    wall_hit_counts = np.zeros(number_saved, dtype=np.int64)

    position_star = np.asarray(config.initial.position, dtype=float) / radius
    heading = wrap_angle(config.initial.heading)
    save_cursor = 0
    tumbles_since_save = 0
    wall_hits_since_save = 0

    def save(index: int, state: KineticState) -> None:
        positions[index] = position_star
        headings[index] = heading
        concentration[index] = state.concentration
        alignment[index] = state.alignment
        speed[index] = state.v
        D_T[index] = state.D_T
        D_R[index] = state.D_R
        tumble_rate[index] = state.tumble_rate

    state = _evaluate_dimensionless(
        field,
        law,
        position_star,
        heading,
        radius,
        time_scale,
        config.kinetics.D_T0,
    )
    save(0, state)

    step_iterator = progress_bar(
        range(steps),
        total=steps,
        description=f"Python case {case_number}",
        unit="step",
        enabled=show_progress,
    )
    for step in step_iterator:
        # Left-endpoint evaluation is the Itô convention used in the manuscript.
        state = _evaluate_dimensionless(
            field,
            law,
            position_star,
            heading,
            radius,
            time_scale,
            config.kinetics.D_T0,
        )
        lambda_dt = state.tumble_rate * dt_star
        if lambda_dt > settings.max_lambda_dt:
            raise ValueError(
                f"lambda*dt reached {lambda_dt:.4g}, above max_lambda_dt="
                f"{settings.max_lambda_dt:.4g}; reduce simulation.dt"
            )

        displacement = (
            state.v * heading_vector(heading) * dt_star
            + np.sqrt(2.0 * state.D_T * dt_star) * innovations.translation[step]
        )
        diffused_heading = wrap_angle(
            heading + np.sqrt(2.0 * state.D_R * dt_star) * innovations.rotation[step]
        )

        boundary_result = advance_in_disk(
            position_star,
            displacement,
            diffused_heading,
            1.0,
            config.domain.boundary,
            innovations.wall_uniform[step],
        )
        position_star = boundary_result.position
        heading = boundary_result.heading
        wall_hits_since_save += boundary_result.hits

        # Frozen-rate event probability is exact over dt for a constant rate
        # and first-order accurate for the state-dependent process.
        event_probability = -np.expm1(-lambda_dt)
        if innovations.tumble_event[step] < event_probability:
            heading = sample_tumble_heading(
                heading,
                config.kinetics.a1,
                innovations.tumble_normal[step],
                innovations.tumble_uniform[step],
            )
            tumbles_since_save += 1

        completed_step = step + 1
        if (
            save_cursor + 1 < number_saved
            and completed_step == save_steps[save_cursor + 1]
        ):
            save_cursor += 1
            state = _evaluate_dimensionless(
                field,
                law,
                position_star,
                heading,
                radius,
                time_scale,
                config.kinetics.D_T0,
            )
            save(save_cursor, state)
            tumble_counts[save_cursor] = tumbles_since_save
            wall_hit_counts[save_cursor] = wall_hits_since_save
            tumbles_since_save = 0
            wall_hits_since_save = 0

    return Trajectory(
        case_number=case_number,
        case_title=case.title,
        times=times,
        positions=positions,
        headings=headings,
        concentration=concentration,
        alignment=alignment,
        speed=speed,
        translational_diffusivity=D_T,
        rotational_diffusivity=D_R,
        tumble_rate=tumble_rate,
        tumble_counts=tumble_counts,
        wall_hit_counts=wall_hit_counts,
    )


def simulate(
    config: ProjectConfig,
    case_number: int,
    *,
    field: ChemicalField | None = None,
    innovations: Innovations | None = None,
    show_progress: bool = False,
) -> Trajectory:
    """Dispatch to the native Numba kernel or the readable Python reference."""

    field = create_field(config.field) if field is None else field
    config, _ = resolve_numerics(config, field, (case_number,))
    if isinstance(config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    steps = round(config.simulation.duration / config.simulation.dt)
    innovations = (
        Innovations.generate(steps, config.simulation.seed)
        if innovations is None
        else innovations
    )

    if config.performance.engine != "python":
        from .accelerated import numba_available_for, simulate_numba

        if numba_available_for(field):
            bar = progress_bar(
                total=1,
                description=f"Numba case {case_number}",
                unit="trajectory",
                enabled=show_progress,
            )
            try:
                result = simulate_numba(config, case_number, field, innovations)
            finally:
                bar.update(1)
                bar.close()
            return result
        if config.performance.engine == "numba":
            raise ValueError(
                "performance.engine='numba' requires a gaussian, linear, or "
                "multi_gaussian field; use 'auto' for a custom-field fallback"
            )

    return _simulate_python(
        config,
        case_number,
        field=field,
        innovations=innovations,
        show_progress=show_progress,
    )


def simulate_all_cases(
    config: ProjectConfig,
    field: ChemicalField | None = None,
    *,
    show_progress: bool = False,
) -> list[Trajectory]:
    """Simulate cases 1-8 with common random innovations."""

    field = create_field(config.field) if field is None else field
    config, _ = resolve_numerics(config, field, tuple(range(1, 9)))
    if isinstance(config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    steps = round(config.simulation.duration / config.simulation.dt)
    shared = Innovations.generate(steps, config.simulation.seed)
    trajectories = []
    cases = progress_bar(
        range(1, 9),
        total=8,
        description="Canonical trajectories",
        unit="case",
        enabled=show_progress,
    )
    for case_number in cases:
        trajectories.append(
            simulate(
                config,
                case_number,
                field=field,
                innovations=shared,
                show_progress=False,
            )
        )
    return trajectories
