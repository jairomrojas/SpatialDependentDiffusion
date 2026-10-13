"""Pre-generated random innovations for speed and reproducible case comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray


@dataclass(frozen=True)
class Innovations:
    translation: NDArray[np.float64]
    rotation: NDArray[np.float64]
    tumble_event: NDArray[np.float64]
    tumble_normal: NDArray[np.float64]
    tumble_uniform: NDArray[np.float64]

    @classmethod
    def generate(cls, steps: int, seed: int) -> Innovations:
        rng = np.random.default_rng(seed)
        return cls(
            translation=rng.standard_normal((steps, 2)),
            rotation=rng.standard_normal(steps),
            tumble_event=rng.random(steps),
            tumble_normal=rng.standard_normal(steps),
            tumble_uniform=rng.random(steps),
        )


@dataclass(frozen=True)
class DensityParticleDraws:
    """Initial state and step innovations used by the fused density kernel."""

    initial_position: tuple[float, float]
    initial_heading: float
    innovations: Innovations


@njit(cache=True)
def _density_particle_draws_kernel(
    steps: int,
    seed: int,
    initial_x: float,
    initial_y: float,
    initial_heading: float,
    uniform_initial_position: bool,
    gaussian_initial_position: bool,
    gaussian_sigma: float,
    isotropic_initial_heading: bool,
):
    """Mirror the fused Numba kernel's scalar draw order exactly."""

    np.random.seed(seed)
    position_x = initial_x
    position_y = initial_y
    if uniform_initial_position:
        radial_coordinate = np.sqrt(np.random.random())
        polar_angle = 2.0 * np.pi * np.random.random()
        position_x = radial_coordinate * np.cos(polar_angle)
        position_y = radial_coordinate * np.sin(polar_angle)
    elif gaussian_initial_position:
        while True:
            candidate_x = initial_x + gaussian_sigma * np.random.standard_normal()
            candidate_y = initial_y + gaussian_sigma * np.random.standard_normal()
            if candidate_x * candidate_x + candidate_y * candidate_y <= 1.0:
                position_x = candidate_x
                position_y = candidate_y
                break
    heading = initial_heading % (2.0 * np.pi)
    if isotropic_initial_heading:
        heading = 2.0 * np.pi * np.random.random()

    translation = np.empty((steps, 2), dtype=np.float64)
    rotation = np.empty(steps, dtype=np.float64)
    tumble_event = np.empty(steps, dtype=np.float64)
    tumble_normal = np.empty(steps, dtype=np.float64)
    tumble_uniform = np.empty(steps, dtype=np.float64)
    for step in range(steps):
        translation[step, 0] = np.random.standard_normal()
        translation[step, 1] = np.random.standard_normal()
        rotation[step] = np.random.standard_normal()
        tumble_event[step] = np.random.random()
        tumble_normal[step] = np.random.standard_normal()
        tumble_uniform[step] = np.random.random()
    return (
        position_x,
        position_y,
        heading,
        translation,
        rotation,
        tumble_event,
        tumble_normal,
        tumble_uniform,
    )


def generate_density_particle_draws(
    steps: int,
    seed: int,
    *,
    initial_position: tuple[float, float],
    initial_heading: float,
    uniform_initial_position: bool,
    isotropic_initial_heading: bool,
    gaussian_initial_position: bool = False,
    gaussian_sigma: float = 0.0,
) -> DensityParticleDraws:
    """Generate a density particle identically in Python and fused engines."""

    outputs = _density_particle_draws_kernel(
        steps,
        int(seed) & 0xFFFFFFFF,
        float(initial_position[0]),
        float(initial_position[1]),
        float(initial_heading),
        uniform_initial_position,
        gaussian_initial_position,
        float(gaussian_sigma),
        isotropic_initial_heading,
    )
    return DensityParticleDraws(
        initial_position=(float(outputs[0]), float(outputs[1])),
        initial_heading=float(outputs[2]),
        innovations=Innovations(
            translation=outputs[3],
            rotation=outputs[4],
            tumble_event=outputs[5],
            tumble_normal=outputs[6],
            tumble_uniform=outputs[7],
        ),
    )
