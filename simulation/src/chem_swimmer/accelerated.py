"""Portable Numba/LLVM CPU kernels for simulation and reduction.

The readable implementation in ``simulator.py`` remains the reference and
fallback. These kernels contain the same Euler-Maruyama splitting but keep all
per-step scalar work inside native machine code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import config as numba_config
from numba import njit, prange, set_num_threads
from numpy.typing import NDArray

from .cases import get_case
from .config import ProjectConfig, Responses
from .fields import ChemicalField, disk_area_average
from .fields.builtin import (
    GaussianField,
    LinearField,
    MultiGaussianField,
    PeriodicField,
)
from .noise import Innovations
from .turning import TurningKernel, create_turning_kernel

TAU = 2.0 * np.pi
FIELD_GAUSSIAN = 0
FIELD_LINEAR = 1
# The periodic field reuses ``linear_parameters`` for its four numbers:
# amplitude, q_x, q_y, phase.
FIELD_PERIODIC = 2


@dataclass(frozen=True)
class PackedField:
    kind: int
    baseline: float
    eta0: float
    linear_parameters: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    center_x: NDArray[np.float64]
    center_y: NDArray[np.float64]
    sigma_x: NDArray[np.float64]
    sigma_y: NDArray[np.float64]
    cosine: NDArray[np.float64]
    sine: NDArray[np.float64]


def numba_available_for(
    field: ChemicalField,
    turning_kernel: TurningKernel | None = None,
) -> bool:
    """Whether both model components have an exact native implementation."""

    field_supported = isinstance(
        field, (GaussianField, LinearField, MultiGaussianField, PeriodicField)
    )
    turning_supported = turning_kernel is None or turning_kernel.supports_numba
    return field_supported and turning_supported


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


def pack_field(field: ChemicalField, radius: float) -> PackedField:
    eta0 = disk_area_average(field, radius)
    empty = np.empty(0, dtype=np.float64)
    if isinstance(field, LinearField):
        return PackedField(
            kind=FIELD_LINEAR,
            baseline=field.baseline,
            eta0=eta0,
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

    if isinstance(field, PeriodicField):
        return PackedField(
            kind=FIELD_PERIODIC,
            baseline=field.baseline,
            eta0=eta0,
            linear_parameters=np.asarray(
                [
                    field.amplitude,
                    field.wavevector[0],
                    field.wavevector[1],
                    field.phase,
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
        eta0=eta0,
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
            responses.D_T_eta,
            responses.D_R_eta,
            responses.v_eta,
            responses.lambda_eta,
            responses.D_T_zeta,
            responses.D_R_zeta,
            responses.v_zeta,
            responses.lambda_zeta,
            responses.chi_theta,
        ],
        dtype=np.float64,
    )


def _performance_core_count() -> int:
    """Threads to use by default, given a possibly hybrid CPU.

    A restricted affinity mask wins outright, so ``taskset -c 0-7 python ...``
    gets exactly eight worker threads. Under a hypervisor this bounds guest
    scheduling but does not prove that those virtual CPUs stay on physical
    P-cores. Otherwise, native Linux hybrid parts expose their
    performance cores as ``/sys/devices/cpu_core/cpus``; when that is readable
    we use it directly. WSL generally hides the mapping, so we fall back to
    three quarters of the available virtual CPUs. Either way the point is the
    same: with static scheduling the slowest worker sets the makespan, so a
    measured cap can beat oversubscribing a heterogeneous host.
    """

    # Honour an explicit guest affinity mask without claiming that a WSL
    # virtual-CPU number identifies a specific physical core type.
    try:
        allowed = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = os.cpu_count() or 1
    logical = os.cpu_count() or 1
    if allowed < logical:
        return max(1, allowed)

    try:
        listing = Path("/sys/devices/cpu_core/cpus").read_text().strip()
    except OSError:
        listing = ""
    count = 0
    for span in filter(None, listing.split(",")):
        first, _, last = span.partition("-")
        try:
            count += int(last) - int(first) + 1 if last else 1
        except ValueError:
            count = 0
            break
    if count:
        return min(count, allowed)
    return max(1, (allowed * 3) // 4)


def configure_threads(config: ProjectConfig, task_count: int) -> None:
    """Apply the configured native-thread limit to a bounded parallel task.

    The ceiling is Numba's own ``NUMBA_NUM_THREADS``, not ``os.cpu_count()``:
    asking for more than Numba was built to launch raises, which used to make
    the package crash whenever a user exported ``NUMBA_NUM_THREADS``.

    With ``performance.threads = 0`` the default is deliberately *not* every
    logical core.  On hybrid CPUs the efficiency cores finish their static
    share late and set the makespan, so the whole batch waits on them; a cap
    near the performance-core count measures faster than using all cores.
    """

    ceiling = int(numba_config.NUMBA_NUM_THREADS)
    if config.performance.threads > 0:
        desired = config.performance.threads
    else:
        desired = min(ceiling, _performance_core_count())
    set_num_threads(max(1, min(desired, ceiling, task_count)))


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
    if kind == FIELD_PERIODIC:
        amplitude = linear_parameters[0]
        wave_x = linear_parameters[1]
        wave_y = linear_parameters[2]
        angle = wave_x * x + wave_y * y + linear_parameters[3]
        slope = -amplitude * np.sin(angle)
        return (
            baseline + amplitude * np.cos(angle),
            slope * wave_x,
            slope * wave_y,
        )

    if kind == FIELD_LINEAR:
        gradient_x = linear_parameters[0]
        gradient_y = linear_parameters[1]
        eta = (
            baseline
            + (x - linear_parameters[2]) * gradient_x
            + (y - linear_parameters[3]) * gradient_y
        )
        return eta, gradient_x, gradient_y

    eta = baseline
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
        eta += gaussian
        gradient_x += local_gradient_x * cos_angle - local_gradient_y * sin_angle
        gradient_y += local_gradient_x * sin_angle + local_gradient_y * cos_angle
    return eta, gradient_x, gradient_y


@njit(cache=True, inline="always")
def _kinetic_values(
    eta,
    gradient_x,
    gradient_y,
    heading,
    eta0,
    v0,
    D_T0,
    D_R0,
    lambda0,
    responses,
):
    heading_x = np.cos(heading)
    heading_y = np.sin(heading)
    delta_eta = eta - eta0
    zeta = heading_x * gradient_x + heading_y * gradient_y
    D_T = D_T0 + responses[0] * delta_eta + responses[4] * zeta
    D_R = D_R0 + responses[1] * delta_eta + responses[5] * zeta
    speed = v0 + responses[2] * delta_eta + responses[6] * zeta
    tumble_rate = lambda0 + responses[3] * delta_eta + responses[7] * zeta
    xi_perp = -heading_y * gradient_x + heading_x * gradient_y
    angular_drift = responses[8] * xi_perp
    return zeta, speed, D_T, D_R, tumble_rate, angular_drift


@njit(cache=True, inline="always")
def _advance_in_disk(
    x,
    y,
    displacement_x,
    displacement_y,
    heading,
    radius,
):
    """Native twin of ``boundary.advance_in_disk``: the sliding wall.

    An increment that would leave the disk is advanced to the wall; the
    outward normal component of what remains is absorbed by impermeability
    and the tangential component is travelled as arc length.  The heading is
    an independent internal state and is never altered by wall contact.
    """

    tolerance = 2.0e-13 * max(radius, 1.0)
    angle = heading % TAU

    trial_x = x + displacement_x
    trial_y = y + displacement_y
    norm_squared = trial_x * trial_x + trial_y * trial_y
    if norm_squared <= radius * radius * (1.0 + 1.0e-13):
        norm = np.sqrt(norm_squared)
        if norm > radius:
            scale = (radius - tolerance) / norm
            trial_x *= scale
            trial_y *= scale
        return trial_x, trial_y, angle, 0, 0

    a = displacement_x * displacement_x + displacement_y * displacement_y
    if a == 0.0:
        return x, y, angle, 0, 3
    b = 2.0 * (x * displacement_x + y * displacement_y)
    c = x * x + y * y - radius * radius
    discriminant = max(0.0, b * b - 4.0 * a * c)
    fraction = (-b + np.sqrt(discriminant)) / (2.0 * a)
    fraction = min(1.0, max(0.0, fraction))

    contact_x = x + fraction * displacement_x
    contact_y = y + fraction * displacement_y
    contact_norm = np.sqrt(contact_x * contact_x + contact_y * contact_y)
    if contact_norm == 0.0:
        return x, y, angle, 0, 3
    normal_x = contact_x / contact_norm
    normal_y = contact_y / contact_norm
    tangent_x = -normal_y
    tangent_y = normal_x

    remaining_x = (1.0 - fraction) * displacement_x
    remaining_y = (1.0 - fraction) * displacement_y
    arc = remaining_x * tangent_x + remaining_y * tangent_y
    rotation = arc / radius
    cosine = np.cos(rotation)
    sine = np.sin(rotation)
    slid_x = normal_x * cosine - normal_y * sine
    slid_y = normal_x * sine + normal_y * cosine
    scale = radius - tolerance
    return slid_x * scale, slid_y * scale, angle, 1, 0


@njit(cache=True, parallel=True)
def _observable_fields_kernel(
    positions,
    headings,
    radius,
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
    eta,
    zeta,
    nematic_zeta,
):
    """Evaluate eta, zeta and the nematic gradient projection at saved states.

    The NumPy path rebuilt the whole chemical field over every particle, case
    and reported slice, which dominated the density computation.  This walks
    the same points once, in parallel over cases, reusing the packed field the
    trajectory kernel already uses.
    """

    _, case_count, time_count = headings.shape
    batch_size = headings.shape[0]
    epsilon = 2.220446049250313e-16
    for case_index in prange(case_count):
        for particle in range(batch_size):
            for time_index in range(time_count):
                x = positions[particle, case_index, time_index, 0] * radius
                y = positions[particle, case_index, time_index, 1] * radius
                value, gradient_x, gradient_y = _field_value_gradient(
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
                )
                gradient_x *= radius
                gradient_y *= radius
                angle = headings[particle, case_index, time_index]
                projection = np.cos(angle) * gradient_x + np.sin(angle) * gradient_y
                norm = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
                eta[particle, case_index, time_index] = value
                zeta[particle, case_index, time_index] = projection
                if norm > epsilon:
                    nematic_zeta[particle, case_index, time_index] = (
                        2.0 * projection * projection / norm - norm
                    )
                else:
                    nematic_zeta[particle, case_index, time_index] = 0.0


def observable_fields_numba(field, radius, positions, headings):
    """Host wrapper: allocate outputs and dispatch the packed field."""

    packed = pack_field(field, radius)
    shape = headings.shape
    eta = np.empty(shape, dtype=np.float64)
    zeta = np.empty(shape, dtype=np.float64)
    nematic_zeta = np.empty(shape, dtype=np.float64)
    _observable_fields_kernel(
        positions,
        headings,
        radius,
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
        eta,
        zeta,
        nematic_zeta,
    )
    return eta, zeta, nematic_zeta


@njit(cache=True, inline="always")
def _tumble_heading(old_heading, P_turn, normal_innovation, uniform_innovation):
    """Native sampler for the wrapped-normal kernel with first moment P_turn."""

    if P_turn == 0.0:
        turn_angle = TAU * (uniform_innovation - 0.5)
    elif abs(P_turn) == 1.0:
        turn_angle = 0.0 if P_turn > 0.0 else np.pi
    else:
        center = 0.0 if P_turn > 0.0 else np.pi
        standard_deviation = np.sqrt(-2.0 * np.log(abs(P_turn)))
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
    v0,
    D_T0,
    D_R0,
    lambda0,
    P_turn,
    max_lambda_dt,
    responses,
    field_kind,
    field_baseline,
    field_eta0,
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
):
    steps = rotation.size
    number_saved = save_steps.size
    positions = np.empty((number_saved, 2), dtype=np.float64)
    headings = np.empty(number_saved, dtype=np.float64)
    eta_values = np.empty(number_saved, dtype=np.float64)
    zeta_values = np.empty(number_saved, dtype=np.float64)
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

    eta, gradient_x, gradient_y = _field_value_gradient(
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
    zeta, speed, D_T, D_R, tumble_rate, _ = _kinetic_values(
        eta,
        gradient_x,
        gradient_y,
        heading,
        field_eta0,
        v0,
        D_T0,
        D_R0,
        lambda0,
        responses,
    )
    zeta *= radius
    speed *= radius / D_T0
    D_T /= D_T0
    D_R *= time_scale
    tumble_rate *= time_scale
    positions[0, 0] = x
    positions[0, 1] = y
    headings[0] = heading
    eta_values[0] = eta
    zeta_values[0] = zeta
    speed_values[0] = speed
    D_T_values[0] = D_T
    D_R_values[0] = D_R
    rate_values[0] = tumble_rate

    for step in range(steps):
        eta, gradient_x, gradient_y = _field_value_gradient(
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
        zeta, speed, D_T, D_R, tumble_rate, angular_drift = _kinetic_values(
            eta,
            gradient_x,
            gradient_y,
            heading,
            field_eta0,
            v0,
            D_T0,
            D_R0,
            lambda0,
            responses,
        )
        zeta *= radius
        speed *= radius / D_T0
        D_T /= D_T0
        D_R *= time_scale
        tumble_rate *= time_scale
        angular_drift *= time_scale
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
        advanced_heading = (
            heading
            + angular_drift * dt_star
            + np.sqrt(2.0 * D_R * dt_star) * rotation[step]
        ) % TAU

        x, y, heading, wall_hits, boundary_status = _advance_in_disk(
            x,
            y,
            displacement_x,
            displacement_y,
            advanced_heading,
            1.0,
        )
        walls_since_save += wall_hits
        if boundary_status != 0:
            status = boundary_status
            break

        if tumble_event[step] < -np.expm1(-lambda_dt):
            heading = _tumble_heading(
                heading,
                P_turn,
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
            eta, gradient_x, gradient_y = _field_value_gradient(
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
            zeta, speed, D_T, D_R, tumble_rate, _ = _kinetic_values(
                eta,
                gradient_x,
                gradient_y,
                heading,
                field_eta0,
                v0,
                D_T0,
                D_R0,
                lambda0,
                responses,
            )
            zeta *= radius
            speed *= radius / D_T0
            D_T /= D_T0
            D_R *= time_scale
            tumble_rate *= time_scale
            positions[save_cursor, 0] = x
            positions[save_cursor, 1] = y
            headings[save_cursor] = heading
            eta_values[save_cursor] = eta
            zeta_values[save_cursor] = zeta
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
        eta_values,
        zeta_values,
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
    v0,
    D_T0,
    D_R0,
    lambda0,
    P_turn,
    max_lambda_dt,
    response_strengths,
    field_kind,
    field_baseline,
    field_eta0,
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
):
    replicate_count = rotation.shape[0]
    response_count = response_strengths.size
    result = np.empty(
        (replicate_count, response_count, save_steps.size), dtype=np.float64
    )
    statuses = np.zeros((replicate_count, response_count), dtype=np.int64)

    for task in prange(replicate_count * response_count):
        replicate_index = task // response_count
        case_index = task % response_count
        responses = np.zeros(response_count, dtype=np.float64)
        responses[case_index] = response_strengths[case_index]
        outputs = _simulate_kernel(
            dt,
            save_steps,
            initial_x,
            initial_y,
            initial_heading,
            radius,
            v0,
            D_T0,
            D_R0,
            lambda0,
            P_turn,
            max_lambda_dt,
            responses,
            field_kind,
            field_baseline,
            field_eta0,
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
    gaussian_initial_position,
    gaussian_sigma,
    radius,
    v0,
    D_T0,
    D_R0,
    lambda0,
    P_turn,
    max_lambda_dt,
    response_matrix,
    field_kind,
    field_baseline,
    field_eta0,
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
    """Simulate all paired cases without materializing random streams."""

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
        elif gaussian_initial_position:
            # Rejection sampling: draw from the untruncated 2D Gaussian and
            # discard anything that lands outside the disk, so the accepted
            # sample matches gaussian_projection's disk-renormalized target
            # exactly (not an approximation of it).
            while True:
                candidate_x = (
                    initial_x_star + gaussian_sigma * np.random.standard_normal()
                )
                candidate_y = (
                    initial_y_star + gaussian_sigma * np.random.standard_normal()
                )
                if candidate_x * candidate_x + candidate_y * candidate_y <= 1.0:
                    position_x = candidate_x
                    position_y = candidate_y
                    break
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
            for case_number in range(case_count):
                if statuses[replicate_index, case_number] != 0:
                    continue

                eta, gradient_x, gradient_y = _field_value_gradient(
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
                zeta, speed, D_T, D_R, tumble_rate, angular_drift = _kinetic_values(
                    eta,
                    gradient_x,
                    gradient_y,
                    heading[case_number],
                    field_eta0,
                    v0,
                    D_T0,
                    D_R0,
                    lambda0,
                    response_matrix[case_number],
                )
                zeta *= radius
                speed *= radius / D_T0
                D_T /= D_T0
                D_R *= time_scale
                tumble_rate *= time_scale
                angular_drift *= time_scale
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
                advanced_heading = (
                    angle
                    + angular_drift * dt_star
                    + np.sqrt(2.0 * D_R * dt_star) * rotation
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
                    advanced_heading,
                    1.0,
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
                        P_turn,
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
                    heading_result[replicate_index, case_number, save_cursor] = heading[
                        case_number
                    ]
                    tumble_result[replicate_index, case_number, save_cursor] = (
                        cumulative_tumbles[case_number]
                    )

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
    turning_kernel = create_turning_kernel(config.turning)
    if not turning_kernel.supports_numba:
        raise TypeError("this turning kernel requires the readable Python engine")
    return (
        config.simulation.dt,
        save_steps,
        config.initial.position[0],
        config.initial.position[1],
        config.initial.heading,
        config.domain.radius,
        base.v0,
        base.D_T0,
        base.D_R0,
        base.lambda0,
        turning_kernel.P_turn,
        config.simulation.max_lambda_dt,
        packed.kind,
        packed.baseline,
        packed.eta0,
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

    packed = pack_field(field, config.domain.radius)
    save_steps = _save_steps(config)
    common = _kernel_arguments(config, packed, save_steps)
    responses = response_array(config.responses.only_case(case_number))
    outputs = _simulate_kernel(
        *common[:12],
        responses,
        *common[12:],
        innovations.translation,
        innovations.rotation,
        innovations.tumble_event,
        innovations.tumble_normal,
        innovations.tumble_uniform,
    )
    status = int(outputs[10])
    if status:
        messages = {
            1: "the linear response produced a negative kinetic parameter",
            2: "lambda*dt exceeded simulation.max_lambda_dt",
            3: "a degenerate wall contact was encountered; reduce simulation.dt",
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
        eta=outputs[2],
        zeta=outputs[3],
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
    for index, seed in enumerate(seeds):
        innovations = Innovations.generate(steps, seed)
        translation[index] = innovations.translation
        rotation[index] = innovations.rotation
        tumble_event[index] = innovations.tumble_event
        tumble_normal[index] = innovations.tumble_normal
        tumble_uniform[index] = innovations.tumble_uniform
    return (
        translation,
        rotation,
        tumble_event,
        tumble_normal,
        tumble_uniform,
    )


def msd_batch_numba(
    config: ProjectConfig,
    field: ChemicalField,
    seeds: list[int],
) -> NDArray[np.float64]:
    if isinstance(config.simulation.dt, str):
        raise TypeError("internal error: automatic dt was not resolved")
    packed = pack_field(field, config.domain.radius)
    save_steps = _save_steps(config)
    steps = round(config.simulation.duration / config.simulation.dt)
    noise = _noise_batch(seeds, steps)
    configure_threads(config, len(seeds) * 8)
    common = _kernel_arguments(config, packed, save_steps)
    response_strengths = response_array(config.responses)
    result, statuses = _batch_msd_kernel(
        *common[:12],
        response_strengths,
        *common[12:],
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
    gaussian_positions: bool = False,
    gaussian_sigma: float = 0.0,
    case_numbers: tuple[int, ...] = tuple(range(10)),
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
    if any(case_number not in range(10) for case_number in case_numbers):
        raise ValueError("case_numbers must contain only integers from 0 through 9")
    if len(set(case_numbers)) != len(case_numbers):
        raise ValueError("case_numbers must not contain duplicates")
    if not np.isfinite(response_scale):
        raise ValueError("response_scale must be finite")
    packed = pack_field(field, config.domain.radius)
    save_steps = _save_steps(config)
    configure_threads(config, len(seeds))
    common = _kernel_arguments(config, packed, save_steps)
    response_strengths = response_array(config.responses)
    response_matrix = np.zeros((len(case_numbers), 9), dtype=np.float64)
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
    # The canonical comparison has ten distinct response rows.  In that common
    # case, preserving the original order avoids three full ``np.take`` copies
    # of the batch outputs (positions, headings, and tumble counts).  Retain
    # deduplication for zero-valued or intentionally repeated responses.
    all_responses_unique = len(unique_responses) == len(response_matrix)
    kernel_responses = response_matrix if all_responses_unique else unique_responses
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
    unique_result, unique_headings, unique_tumbles, statuses = _seeded_position_kernel(
        *common[:5],
        isotropic_headings,
        uniform_positions,
        gaussian_positions,
        gaussian_sigma,
        *common[5:12],
        kernel_responses,
        *common[12:],
        native_seeds,
    )
    if np.any(statuses):
        status = int(statuses[np.nonzero(statuses)][0])
        messages = {
            1: "the linear response produced a negative kinetic parameter",
            2: "lambda*dt exceeded simulation.max_lambda_dt",
            3: "a degenerate wall contact was encountered; reduce simulation.dt",
        }
        raise ValueError(
            f"Numba position kernel stopped: {messages.get(status, 'unknown error')}"
        )
    if all_responses_unique:
        return unique_result, unique_headings, unique_tumbles
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
    """Return ``r/R`` with shape ``(M, 10, times, 2)``."""

    positions, _, _ = state_batch_numba(
        config,
        field,
        seeds,
        isotropic_headings=isotropic_headings,
        uniform_positions=uniform_positions,
    )
    return positions
