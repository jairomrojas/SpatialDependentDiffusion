"""Numba/LLVM kernels for Apple-Silicon CPU execution.

The readable implementation in ``simulator.py`` remains the reference and
fallback. These kernels contain the same Euler-Maruyama splitting but keep all
per-step scalar work inside native machine code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numba import njit, prange, set_num_threads
from numpy.typing import NDArray

from .cases import get_case
from .config import ProjectConfig, Responses
from .fields import ChemicalField
from .fields.analytic import GaussianField, LinearField, MultiGaussianField
from .noise import Innovations

TAU = 2.0 * np.pi
FIELD_GAUSSIAN = 0
FIELD_LINEAR = 1
BOUNDARY_MODES = {"sliding": 0, "specular": 1, "diffuse": 2}


@dataclass(frozen=True)
class PackedField:
    kind: int
    baseline: float
    linear_parameters: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    center_x: NDArray[np.float64]
    center_y: NDArray[np.float64]
    sigma_x: NDArray[np.float64]
    sigma_y: NDArray[np.float64]
    cosine: NDArray[np.float64]
    sine: NDArray[np.float64]


def numba_available_for(field: ChemicalField) -> bool:
    return isinstance(field, (GaussianField, LinearField, MultiGaussianField))


def _source_arrays(
    sources: tuple[GaussianField, ...],
) -> tuple[NDArray[np.float64], ...]:
    return (
        np.asarray([source.amplitude for source in sources], dtype=np.float64),
        np.asarray([source.center[0] for source in sources], dtype=np.float64),
        np.asarray([source.center[1] for source in sources], dtype=np.float64),
        np.asarray([source.sigma[0] for source in sources], dtype=np.float64),
        np.asarray([source.sigma[1] for source in sources], dtype=np.float64),
        np.asarray([np.cos(source.angle) for source in sources], dtype=np.float64),
        np.asarray([np.sin(source.angle) for source in sources], dtype=np.float64),
    )


def pack_field(field: ChemicalField) -> PackedField:
    empty = np.empty(0, dtype=np.float64)
    if isinstance(field, LinearField):
        return PackedField(
            kind=FIELD_LINEAR,
            baseline=field.baseline,
            linear_parameters=np.asarray(
                [
                    field.gradient_vector[0],
                    field.gradient_vector[1],
                    field.center[0],
                    field.center[1],
                ],
                dtype=np.float64,
            ),
            amplitudes=empty,
            center_x=empty,
            center_y=empty,
            sigma_x=empty,
            sigma_y=empty,
            cosine=empty,
            sine=empty,
        )

    if isinstance(field, GaussianField):
        baseline = field.baseline
        sources = (field,)
    elif isinstance(field, MultiGaussianField):
        baseline = field.baseline
        sources = field.sources
    else:
        raise TypeError("this chemical field requires the readable Python engine")

    arrays = _source_arrays(sources)
    return PackedField(
        kind=FIELD_GAUSSIAN,
        baseline=baseline,
        linear_parameters=np.zeros(4, dtype=np.float64),
        amplitudes=arrays[0],
        center_x=arrays[1],
        center_y=arrays[2],
        sigma_x=arrays[3],
        sigma_y=arrays[4],
        cosine=arrays[5],
        sine=arrays[6],
    )


def response_array(responses: Responses) -> NDArray[np.float64]:
    return np.asarray(
        [
            responses.D_T_c,
            responses.D_R_c,
            responses.v_c,
            responses.lambda_c,
            responses.D_T_S,
            responses.D_R_S,
            responses.v_S,
            responses.lambda_S,
        ],
        dtype=np.float64,
    )


def configure_threads(config: ProjectConfig, task_count: int) -> None:
    """Apply the configured native-thread limit to a bounded parallel task."""

    available = os.cpu_count() or 1
    if config.performance.threads > 0:
        desired = config.performance.threads
    else:
        desired = available
    set_num_threads(max(1, min(desired, available, task_count)))


@njit(cache=True, inline="always")
def _field_value_gradient(
    x,
    y,
    kind,
    baseline,
    linear_parameters,
    amplitudes,
    center_x,
    center_y,
    sigma_x,
    sigma_y,
    cosine,
    sine,
):
    if kind == FIELD_LINEAR:
        gradient_x = linear_parameters[0]
        gradient_y = linear_parameters[1]
        concentration = (
            baseline
            + (x - linear_parameters[2]) * gradient_x
            + (y - linear_parameters[3]) * gradient_y
        )
        return concentration, gradient_x, gradient_y

    concentration = baseline
    gradient_x = 0.0
    gradient_y = 0.0
    for source_index in range(amplitudes.size):
        dx = x - center_x[source_index]
        dy = y - center_y[source_index]
        cos_angle = cosine[source_index]
        sin_angle = sine[source_index]
        local_x = dx * cos_angle + dy * sin_angle
        local_y = -dx * sin_angle + dy * cos_angle
        sx = sigma_x[source_index]
        sy = sigma_y[source_index]
        gaussian = amplitudes[source_index] * np.exp(
            -0.5 * ((local_x / sx) ** 2 + (local_y / sy) ** 2)
        )
        local_gradient_x = -gaussian * local_x / (sx * sx)
        local_gradient_y = -gaussian * local_y / (sy * sy)
        concentration += gaussian
        gradient_x += local_gradient_x * cos_angle - local_gradient_y * sin_angle
        gradient_y += local_gradient_x * sin_angle + local_gradient_y * cos_angle
    return concentration, gradient_x, gradient_y


@njit(cache=True, inline="always")
def _kinetic_values(
    concentration,
    gradient_x,
    gradient_y,
    heading,
    baseline,
    v0,
    D_T0,
    D_R0,
    lambda0,
    responses,
):
    heading_x = np.cos(heading)
    heading_y = np.sin(heading)
    delta_c = concentration - baseline
    alignment = heading_x * gradient_x + heading_y * gradient_y
    D_T = D_T0 + responses[0] * delta_c + responses[4] * alignment
    D_R = D_R0 + responses[1] * delta_c + responses[5] * alignment
    speed = v0 + responses[2] * delta_c + responses[6] * alignment
    tumble_rate = lambda0 + responses[3] * delta_c + responses[7] * alignment
    return alignment, speed, D_T, D_R, tumble_rate


@njit(cache=True, inline="always")
def _advance_in_disk(
    x,
    y,
    displacement_x,
    displacement_y,
    heading,
    radius,
    boundary_mode,
    wall_uniforms,
):
    tolerance = 2.0e-13 * max(radius, 1.0)
    current_x = x
    current_y = y
    remaining_x = displacement_x
    remaining_y = displacement_y
    angle = heading
    hits = 0

    for reflection_index in range(wall_uniforms.size):
        trial_x = current_x + remaining_x
        trial_y = current_y + remaining_y
        norm_squared = trial_x * trial_x + trial_y * trial_y
        if norm_squared <= radius * radius * (1.0 + 1.0e-13):
            norm = np.sqrt(norm_squared)
            if norm > radius:
                scale = (radius - tolerance) / norm
                trial_x *= scale
                trial_y *= scale
            return trial_x, trial_y, angle % TAU, hits, 0

        a = remaining_x * remaining_x + remaining_y * remaining_y
        b = 2.0 * (current_x * remaining_x + current_y * remaining_y)
        c = current_x * current_x + current_y * current_y - radius * radius
        discriminant = max(0.0, b * b - 4.0 * a * c)
        fraction = (-b + np.sqrt(discriminant)) / (2.0 * a)
        fraction = min(1.0, max(0.0, fraction))

        hit_x = current_x + fraction * remaining_x
        hit_y = current_y + fraction * remaining_y
        hit_norm = np.sqrt(hit_x * hit_x + hit_y * hit_y)
        normal_x = hit_x / hit_norm
        normal_y = hit_y / hit_norm

        remaining_x *= 1.0 - fraction
        remaining_y *= 1.0 - fraction
        normal_component = remaining_x * normal_x + remaining_y * normal_y
        remaining_x -= 2.0 * normal_component * normal_x
        remaining_y -= 2.0 * normal_component * normal_y
        hits += 1

        if boundary_mode == 1:
            direction_x = np.cos(angle)
            direction_y = np.sin(angle)
            heading_normal = direction_x * normal_x + direction_y * normal_y
            direction_x -= 2.0 * heading_normal * normal_x
            direction_y -= 2.0 * heading_normal * normal_y
            angle = np.arctan2(direction_y, direction_x)
        elif boundary_mode == 2:
            inward_angle = np.arctan2(-normal_y, -normal_x)
            angle = inward_angle + np.pi * (wall_uniforms[reflection_index] - 0.5)

        current_x = hit_x - tolerance * normal_x
        current_y = hit_y - tolerance * normal_y

    final_x = current_x + remaining_x
    final_y = current_y + remaining_y
    final_norm = np.sqrt(final_x * final_x + final_y * final_y)
    if final_norm < radius:
        return final_x, final_y, angle % TAU, hits, 0
    scale = (radius - tolerance) / final_norm
    return final_x * scale, final_y * scale, angle % TAU, hits, 3


@njit(cache=True, inline="always")
def _tumble_heading(old_heading, a1, normal_innovation, uniform_innovation):
    if a1 == 0.0:
        turn_angle = TAU * uniform_innovation
    elif abs(a1) == 1.0:
        turn_angle = 0.0 if a1 > 0.0 else np.pi
    else:
        center = 0.0 if a1 > 0.0 else np.pi
        standard_deviation = np.sqrt(-2.0 * np.log(abs(a1)))
        turn_angle = center + standard_deviation * normal_innovation
    return (old_heading + turn_angle) % TAU


@njit(cache=True)
def _simulate_kernel(
    dt,
    save_steps,
    initial_x,
    initial_y,
    initial_heading,
    radius,
    boundary_mode,
    v0,
    D_T0,
    D_R0,
    lambda0,
    a1,
    max_lambda_dt,
    responses,
    field_kind,
    field_baseline,
    linear_parameters,
    amplitudes,
    center_x,
    center_y,
    sigma_x,
    sigma_y,
    cosine,
    sine,
    translation,
    rotation,
    tumble_event,
    tumble_normal,
    tumble_uniform,
    wall_uniform,
):
    steps = rotation.size
    number_saved = save_steps.size
    positions = np.empty((number_saved, 2), dtype=np.float64)
    headings = np.empty(number_saved, dtype=np.float64)
    concentration_values = np.empty(number_saved, dtype=np.float64)
    alignment_values = np.empty(number_saved, dtype=np.float64)
    speed_values = np.empty(number_saved, dtype=np.float64)
    D_T_values = np.empty(number_saved, dtype=np.float64)
    D_R_values = np.empty(number_saved, dtype=np.float64)
    rate_values = np.empty(number_saved, dtype=np.float64)
    tumble_counts = np.zeros(number_saved, dtype=np.int64)
    wall_counts = np.zeros(number_saved, dtype=np.int64)

    time_scale = radius * radius / D_T0
    dt_star = dt / time_scale
    x = initial_x / radius
    y = initial_y / radius
    heading = initial_heading % TAU
    save_cursor = 0
    tumbles_since_save = 0
    walls_since_save = 0
    status = 0

    concentration, gradient_x, gradient_y = _field_value_gradient(
        x * radius,
        y * radius,
        field_kind,
        field_baseline,
        linear_parameters,
        amplitudes,
        center_x,
        center_y,
        sigma_x,
        sigma_y,
        cosine,
        sine,
    )
    alignment, speed, D_T, D_R, tumble_rate = _kinetic_values(
        concentration,
        gradient_x,
        gradient_y,
        heading,
        field_baseline,
        v0,
        D_T0,
        D_R0,
        lambda0,
        responses,
    )
    alignment *= radius
    speed *= radius / D_T0
    D_T /= D_T0
    D_R *= time_scale
    tumble_rate *= time_scale
    positions[0, 0] = x
    positions[0, 1] = y
    headings[0] = heading
    concentration_values[0] = concentration
    alignment_values[0] = alignment
    speed_values[0] = speed
    D_T_values[0] = D_T
    D_R_values[0] = D_R
    rate_values[0] = tumble_rate

    for step in range(steps):
        concentration, gradient_x, gradient_y = _field_value_gradient(
            x * radius,
            y * radius,
            field_kind,
            field_baseline,
            linear_parameters,
            amplitudes,
            center_x,
            center_y,
            sigma_x,
            sigma_y,
            cosine,
            sine,
        )
        alignment, speed, D_T, D_R, tumble_rate = _kinetic_values(
            concentration,
            gradient_x,
            gradient_y,
            heading,
            field_baseline,
            v0,
            D_T0,
            D_R0,
            lambda0,
            responses,
        )
        alignment *= radius
        speed *= radius / D_T0
        D_T /= D_T0
        D_R *= time_scale
        tumble_rate *= time_scale
        if min(speed, D_T, D_R, tumble_rate) < 0.0:
            status = 1
            break
        lambda_dt = tumble_rate * dt_star
        if lambda_dt > max_lambda_dt:
            status = 2
            break

        displacement_scale = np.sqrt(2.0 * D_T * dt_star)
        displacement_x = (
            speed * np.cos(heading) * dt_star
            + displacement_scale * translation[step, 0]
        )
        displacement_y = (
            speed * np.sin(heading) * dt_star
            + displacement_scale * translation[step, 1]
        )
        diffused_heading = (
            heading + np.sqrt(2.0 * D_R * dt_star) * rotation[step]
        ) % TAU

        x, y, heading, wall_hits, boundary_status = _advance_in_disk(
            x,
            y,
            displacement_x,
            displacement_y,
            diffused_heading,
            1.0,
            boundary_mode,
            wall_uniform[step],
        )
        walls_since_save += wall_hits
        if boundary_status != 0:
            status = boundary_status
            break

        if tumble_event[step] < -np.expm1(-lambda_dt):
            heading = _tumble_heading(
                heading,
                a1,
                tumble_normal[step],
                tumble_uniform[step],
            )
            tumbles_since_save += 1

        completed_step = step + 1
        if (
            save_cursor + 1 < number_saved
            and completed_step == save_steps[save_cursor + 1]
        ):
            save_cursor += 1
            concentration, gradient_x, gradient_y = _field_value_gradient(
                x * radius,
                y * radius,
                field_kind,
                field_baseline,
                linear_parameters,
                amplitudes,
                center_x,
                center_y,
                sigma_x,
                sigma_y,
                cosine,
                sine,
            )
            alignment, speed, D_T, D_R, tumble_rate = _kinetic_values(
                concentration,
                gradient_x,
                gradient_y,
                heading,
                field_baseline,
                v0,
                D_T0,
                D_R0,
                lambda0,
                responses,
            )
            alignment *= radius
            speed *= radius / D_T0
            D_T /= D_T0
            D_R *= time_scale
            tumble_rate *= time_scale
            positions[save_cursor, 0] = x
            positions[save_cursor, 1] = y
            headings[save_cursor] = heading
            concentration_values[save_cursor] = concentration
            alignment_values[save_cursor] = alignment
            speed_values[save_cursor] = speed
            D_T_values[save_cursor] = D_T
            D_R_values[save_cursor] = D_R
            rate_values[save_cursor] = tumble_rate
            tumble_counts[save_cursor] = tumbles_since_save
            wall_counts[save_cursor] = walls_since_save
            tumbles_since_save = 0
            walls_since_save = 0

    return (
        positions,
        headings,
        concentration_values,
        alignment_values,
        speed_values,
        D_T_values,
        D_R_values,
        rate_values,
        tumble_counts,
        wall_counts,
        status,
    )


@njit(cache=True, parallel=True)
def _batch_msd_kernel(
    dt,
    save_steps,
    initial_x,
    initial_y,
    initial_heading,
    radius,
    boundary_mode,
    v0,
    D_T0,
    D_R0,
    lambda0,
    a1,
    max_lambda_dt,
    response_strengths,
    field_kind,
    field_baseline,
    linear_parameters,
    amplitudes,
    center_x,
    center_y,
    sigma_x,
    sigma_y,
    cosine,
    sine,
    translation,
    rotation,
    tumble_event,
    tumble_normal,
    tumble_uniform,
    wall_uniform,
):
    replicate_count = rotation.shape[0]
    result = np.empty((replicate_count, 8, save_steps.size), dtype=np.float64)
    statuses = np.zeros((replicate_count, 8), dtype=np.int64)

    for task in prange(replicate_count * 8):
        replicate_index = task // 8
        case_index = task % 8
        responses = np.zeros(8, dtype=np.float64)
        responses[case_index] = response_strengths[case_index]
        outputs = _simulate_kernel(
            dt,
            save_steps,
            initial_x,
            initial_y,
            initial_heading,
            radius,
            boundary_mode,
            v0,
            D_T0,
            D_R0,
            lambda0,
            a1,
            max_lambda_dt,
            responses,
            field_kind,
            field_baseline,
            linear_parameters,
            amplitudes,
            center_x,
            center_y,
            sigma_x,
            sigma_y,
            cosine,
            sine,
            translation[replicate_index],
            rotation[replicate_index],
            tumble_event[replicate_index],
            tumble_normal[replicate_index],
            tumble_uniform[replicate_index],
            wall_uniform[replicate_index],
        )
        positions = outputs[0]
        for save_index in range(save_steps.size):
            dx = positions[save_index, 0] - initial_x / radius
            dy = positions[save_index, 1] - initial_y / radius
            result[replicate_index, case_index, save_index] = dx * dx + dy * dy
        statuses[replicate_index, case_index] = outputs[10]
    return result, statuses


@njit(cache=True, parallel=True)
def _seeded_position_kernel(
    dt,
    save_steps,
    initial_x,
    initial_y,
    initial_heading,
    isotropic_initial_heading,
    uniform_initial_position,
    radius,
    boundary_mode,
    v0,
    D_T0,
    D_R0,
    lambda0,
    a1,
    max_lambda_dt,
    response_matrix,
    field_kind,
    field_baseline,
    linear_parameters,
    amplitudes,
    center_x,
    center_y,
    sigma_x,
    sigma_y,
    cosine,
    sine,
    seeds,
):
    """Simulate all nine paired cases without materializing random streams."""

    replicate_count = seeds.size
    case_count = response_matrix.shape[0]
    number_saved = save_steps.size
    steps = save_steps[-1]
    result = np.empty(
        (replicate_count, case_count, number_saved, 2),
        dtype=np.float64,
    )
    heading_result = np.empty(
        (replicate_count, case_count, number_saved),
        dtype=np.float64,
    )
    tumble_result = np.zeros(
        (replicate_count, case_count, number_saved),
        dtype=np.uint32,
    )
    statuses = np.zeros((replicate_count, case_count), dtype=np.int64)

    time_scale = radius * radius / D_T0
    dt_star = dt / time_scale
    initial_x_star = initial_x / radius
    initial_y_star = initial_y / radius

    # Each parallel task owns its RNG state and all nine case states. Random
    # increments are generated once per particle-step and shared across cases,
    # preserving the paired comparison without allocating O(M * steps) noise.
    for replicate_index in prange(replicate_count):
        np.random.seed(seeds[replicate_index])
        position_x = initial_x_star
        position_y = initial_y_star
        if uniform_initial_position:
            radial_coordinate = np.sqrt(np.random.random())
            polar_angle = TAU * np.random.random()
            position_x = radial_coordinate * np.cos(polar_angle)
            position_y = radial_coordinate * np.sin(polar_angle)
        x = np.full(case_count, position_x, dtype=np.float64)
        y = np.full(case_count, position_y, dtype=np.float64)
        heading_value = initial_heading % TAU
        if isotropic_initial_heading:
            heading_value = TAU * np.random.random()
        heading = np.full(
            case_count,
            heading_value,
            dtype=np.float64,
        )
        cumulative_tumbles = np.zeros(case_count, dtype=np.uint32)
        # A grazing contact can require several tiny geometric corrections even
        # though the physical displacement is short. Keep ample headroom here;
        # sliding/specular walls do not generate or store these values.
        wall_uniforms = np.empty(64, dtype=np.float64)
        save_cursor = 0
        for case_number in range(case_count):
            result[replicate_index, case_number, 0, 0] = position_x
            result[replicate_index, case_number, 0, 1] = position_y
            heading_result[replicate_index, case_number, 0] = heading_value

        for step in range(steps):
            translation_x = np.random.standard_normal()
            translation_y = np.random.standard_normal()
            rotation = np.random.standard_normal()
            tumble_event = np.random.random()
            tumble_normal = np.random.standard_normal()
            tumble_uniform = np.random.random()
            if boundary_mode == 2:
                for reflection_index in range(wall_uniforms.size):
                    wall_uniforms[reflection_index] = np.random.random()

            for case_number in range(case_count):
                if statuses[replicate_index, case_number] != 0:
                    continue

                concentration, gradient_x, gradient_y = _field_value_gradient(
                    x[case_number] * radius,
                    y[case_number] * radius,
                    field_kind,
                    field_baseline,
                    linear_parameters,
                    amplitudes,
                    center_x,
                    center_y,
                    sigma_x,
                    sigma_y,
                    cosine,
                    sine,
                )
                alignment, speed, D_T, D_R, tumble_rate = _kinetic_values(
                    concentration,
                    gradient_x,
                    gradient_y,
                    heading[case_number],
                    field_baseline,
                    v0,
                    D_T0,
                    D_R0,
                    lambda0,
                    response_matrix[case_number],
                )
                alignment *= radius
                speed *= radius / D_T0
                D_T /= D_T0
                D_R *= time_scale
                tumble_rate *= time_scale
                if min(speed, D_T, D_R, tumble_rate) < 0.0:
                    statuses[replicate_index, case_number] = 1
                    continue

                lambda_dt = tumble_rate * dt_star
                if lambda_dt > max_lambda_dt:
                    statuses[replicate_index, case_number] = 2
                    continue

                angle = heading[case_number]
                displacement_scale = np.sqrt(2.0 * D_T * dt_star)
                displacement_x = (
                    speed * np.cos(angle) * dt_star + displacement_scale * translation_x
                )
                displacement_y = (
                    speed * np.sin(angle) * dt_star + displacement_scale * translation_y
                )
                diffused_heading = (
                    angle + np.sqrt(2.0 * D_R * dt_star) * rotation
                ) % TAU

                (
                    next_x,
                    next_y,
                    next_heading,
                    _,
                    boundary_status,
                ) = _advance_in_disk(
                    x[case_number],
                    y[case_number],
                    displacement_x,
                    displacement_y,
                    diffused_heading,
                    1.0,
                    boundary_mode,
                    wall_uniforms,
                )
                x[case_number] = next_x
                y[case_number] = next_y
                heading[case_number] = next_heading
                if boundary_status != 0:
                    statuses[replicate_index, case_number] = boundary_status
                    continue

                if tumble_event < -np.expm1(-lambda_dt):
                    heading[case_number] = _tumble_heading(
                        heading[case_number],
                        a1,
                        tumble_normal,
                        tumble_uniform,
                    )
                    cumulative_tumbles[case_number] += 1

            completed_step = step + 1
            if (
                save_cursor + 1 < number_saved
                and completed_step == save_steps[save_cursor + 1]
            ):
                save_cursor += 1
                for case_number in range(case_count):
                    result[replicate_index, case_number, save_cursor, 0] = x[
                        case_number
                    ]
                    result[replicate_index, case_number, save_cursor, 1] = y[
                        case_number
                    ]
                    heading_result[
                        replicate_index, case_number, save_cursor
                    ] = heading[case_number]
                    tumble_result[
                        replicate_index, case_number, save_cursor
                    ] = cumulative_tumbles[case_number]

    return result, heading_result, tumble_result, statuses


def _save_steps(config: ProjectConfig) -> NDArray[np.int64]:
    if isinstance(config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    steps = round(config.simulation.duration / config.simulation.dt)
    result = np.arange(
        0,
        steps + 1,
        config.simulation.save_every,
        dtype=np.int64,
    )
    if result[-1] != steps:
        result = np.append(result, steps)
    return result


def _kernel_arguments(
    config: ProjectConfig,
    packed: PackedField,
    save_steps: NDArray[np.int64],
) -> tuple:
    base = config.kinetics
    if isinstance(config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    return (
        config.simulation.dt,
        save_steps,
        config.initial.position[0],
        config.initial.position[1],
        config.initial.heading,
        config.domain.radius,
        BOUNDARY_MODES[config.domain.boundary],
        base.v0,
        base.D_T0,
        base.D_R0,
        base.lambda0,
        base.a1,
        config.simulation.max_lambda_dt,
        packed.kind,
        packed.baseline,
        packed.linear_parameters,
        packed.amplitudes,
        packed.center_x,
        packed.center_y,
        packed.sigma_x,
        packed.sigma_y,
        packed.cosine,
        packed.sine,
    )


def simulate_numba(
    config: ProjectConfig,
    case_number: int,
    field: ChemicalField,
    innovations: Innovations,
):
    """Return a ``Trajectory`` generated by the native scalar kernel."""

    from .simulator import Trajectory

    packed = pack_field(field)
    save_steps = _save_steps(config)
    common = _kernel_arguments(config, packed, save_steps)
    responses = response_array(config.responses.only_case(case_number))
    outputs = _simulate_kernel(
        *common[:13],
        responses,
        *common[13:],
        innovations.translation,
        innovations.rotation,
        innovations.tumble_event,
        innovations.tumble_normal,
        innovations.tumble_uniform,
        innovations.wall_uniform,
    )
    status = int(outputs[10])
    if status:
        messages = {
            1: "the linear response produced a negative kinetic parameter",
            2: "lambda*dt exceeded simulation.max_lambda_dt",
            3: "one step exceeded the wall-reflection limit",
        }
        raise ValueError(
            f"Numba simulation stopped: {messages.get(status, 'unknown error')}"
        )

    time_scale = config.domain.radius * config.domain.radius / config.kinetics.D_T0
    times = save_steps.astype(float) * config.simulation.dt / time_scale
    times[-1] = config.simulation.duration / time_scale
    case = get_case(case_number)
    return Trajectory(
        case_number=case_number,
        case_title=case.title,
        times=times,
        positions=outputs[0],
        headings=outputs[1],
        concentration=outputs[2],
        alignment=outputs[3],
        speed=outputs[4],
        translational_diffusivity=outputs[5],
        rotational_diffusivity=outputs[6],
        tumble_rate=outputs[7],
        tumble_counts=outputs[8],
        wall_hit_counts=outputs[9],
    )


def _noise_batch(seeds: list[int], steps: int) -> tuple[NDArray[np.float64], ...]:
    count = len(seeds)
    translation = np.empty((count, steps, 2), dtype=np.float64)
    rotation = np.empty((count, steps), dtype=np.float64)
    tumble_event = np.empty((count, steps), dtype=np.float64)
    tumble_normal = np.empty((count, steps), dtype=np.float64)
    tumble_uniform = np.empty((count, steps), dtype=np.float64)
    wall_uniform = np.empty((count, steps, 16), dtype=np.float64)
    for index, seed in enumerate(seeds):
        innovations = Innovations.generate(steps, seed)
        translation[index] = innovations.translation
        rotation[index] = innovations.rotation
        tumble_event[index] = innovations.tumble_event
        tumble_normal[index] = innovations.tumble_normal
        tumble_uniform[index] = innovations.tumble_uniform
        wall_uniform[index] = innovations.wall_uniform
    return (
        translation,
        rotation,
        tumble_event,
        tumble_normal,
        tumble_uniform,
        wall_uniform,
    )


def msd_batch_numba(
    config: ProjectConfig,
    field: ChemicalField,
    seeds: list[int],
) -> NDArray[np.float64]:
    if isinstance(config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    packed = pack_field(field)
    save_steps = _save_steps(config)
    steps = round(config.simulation.duration / config.simulation.dt)
    noise = _noise_batch(seeds, steps)
    configure_threads(config, len(seeds) * 8)
    common = _kernel_arguments(config, packed, save_steps)
    response_strengths = response_array(config.responses)
    result, statuses = _batch_msd_kernel(
        *common[:13],
        response_strengths,
        *common[13:],
        *noise,
    )
    if np.any(statuses):
        status = int(statuses[np.nonzero(statuses)][0])
        raise ValueError(f"Numba MSD kernel stopped with status {status}")
    return result


def state_batch_numba(
    config: ProjectConfig,
    field: ChemicalField,
    seeds: list[int],
    *,
    isotropic_headings: bool = False,
    uniform_positions: bool = False,
    case_numbers: tuple[int, ...] = tuple(range(9)),
    response_scale: float = 1.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.uint32],
]:
    """Return saved states for paired selected cases.

    ``response_scale`` multiplies the one active sensitivity in every
    nonzero selected case. It is primarily intended for perturbative scaling
    and sign-reversal studies. Case 0 remains the unresponsive baseline.
    """

    if isinstance(config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    if not case_numbers:
        raise ValueError("state_batch_numba requires at least one case")
    if any(case_number not in range(9) for case_number in case_numbers):
        raise ValueError("case_numbers must contain only integers from 0 through 8")
    if len(set(case_numbers)) != len(case_numbers):
        raise ValueError("case_numbers must not contain duplicates")
    if not np.isfinite(response_scale):
        raise ValueError("response_scale must be finite")
    packed = pack_field(field)
    save_steps = _save_steps(config)
    configure_threads(config, len(seeds))
    common = _kernel_arguments(config, packed, save_steps)
    response_strengths = response_array(config.responses)
    response_matrix = np.zeros((len(case_numbers), 8), dtype=np.float64)
    for row, case_number in enumerate(case_numbers):
        if case_number > 0:
            response_matrix[row, case_number - 1] = (
                response_scale * response_strengths[case_number - 1]
            )
    unique_responses, inverse_cases = np.unique(
        response_matrix,
        axis=0,
        return_inverse=True,
    )
    native_seeds = np.fromiter(
        (int(seed) & 0xFFFFFFFF for seed in seeds),
        dtype=np.uint32,
        count=len(seeds),
    )
    if np.unique(native_seeds).size != native_seeds.size:
        raise ValueError(
            "particle seeds are not unique after conversion to the native "
            "32-bit RNG; use distinct uint32 seeds or a different root seed"
        )
    unique_result, unique_headings, unique_tumbles, statuses = (
        _seeded_position_kernel(
            *common[:5],
            isotropic_headings,
            uniform_positions,
            *common[5:13],
            unique_responses,
            *common[13:],
            native_seeds,
        )
    )
    if np.any(statuses):
        status = int(statuses[np.nonzero(statuses)][0])
        messages = {
            1: "the linear response produced a negative kinetic parameter",
            2: "lambda*dt exceeded simulation.max_lambda_dt",
            3: "one step exceeded the wall-reflection limit",
        }
        raise ValueError(
            f"Numba position kernel stopped: {messages.get(status, 'unknown error')}"
        )
    return (
        np.take(unique_result, inverse_cases, axis=1),
        np.take(unique_headings, inverse_cases, axis=1),
        np.take(unique_tumbles, inverse_cases, axis=1),
    )


def position_batch_numba(
    config: ProjectConfig,
    field: ChemicalField,
    seeds: list[int],
    *,
    isotropic_headings: bool = False,
    uniform_positions: bool = False,
) -> NDArray[np.float64]:
    """Return ``r/R`` with shape ``(M, 9, times, 2)``."""

    positions, _, _ = state_batch_numba(
        config,
        field,
        seeds,
        isotropic_headings=isotropic_headings,
        uniform_positions=uniform_positions,
    )
    return positions
