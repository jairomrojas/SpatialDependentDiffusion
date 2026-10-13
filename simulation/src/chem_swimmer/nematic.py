"""Microscopic nematic-field diagnostics for auditing the ``Q=0`` closure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ProjectConfig
from .density import PositionPDFResult
from .fields import ChemicalField, disk_area_average


@dataclass(frozen=True)
class DensityFluxDecomposition:
    """Exact density current reconstructed from measured angular moments."""

    total_x: NDArray[np.float64]
    total_y: NDArray[np.float64]
    q_x: NDArray[np.float64]
    q_y: NDArray[np.float64]
    q_sampling_error_x: NDArray[np.float64]
    q_sampling_error_y: NDArray[np.float64]
    isotropic_directional_x: NDArray[np.float64]
    isotropic_directional_y: NDArray[np.float64]


def nematic_frobenius(
    q_xx: NDArray[np.float64],
    q_xy: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return ``|Q|_F`` for ``Q=[[Qxx,Qxy],[Qxy,-Qxx]]``."""

    return np.sqrt(2.0 * (q_xx * q_xx + q_xy * q_xy))


def _masked_gradient(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    spacing: float,
    axis: int,
) -> NDArray[np.float64]:
    """Differentiate without using the zero-filled cells outside the disk."""

    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (1, 1)
    padded_values = np.pad(values, pad_width, mode="edge")
    padded_mask = np.pad(mask, pad_width, mode="constant", constant_values=False)

    def shifted(array: NDArray, offset: int) -> NDArray:
        indexer: list[slice] = [slice(None)] * array.ndim
        indexer[axis] = slice(1 + offset, 1 + offset + values.shape[axis])
        return array[tuple(indexer)]

    previous = shifted(padded_values, -1)
    following = shifted(padded_values, 1)
    previous_valid = shifted(padded_mask, -1)
    following_valid = shifted(padded_mask, 1)
    central = (following - previous) / (2.0 * spacing)
    forward = (following - values) / spacing
    backward = (values - previous) / spacing
    derivative = np.where(
        previous_valid & following_valid,
        central,
        np.where(following_valid, forward, np.where(previous_valid, backward, 0.0)),
    )
    return np.where(mask, derivative, 0.0)


def _require_microscopic_fields(
    result: PositionPDFResult,
) -> tuple[NDArray[np.float64], ...]:
    fields = (
        result.polarity_x,
        result.polarity_y,
        result.nematic_xx,
        result.nematic_xy,
    )
    if any(field is None for field in fields):
        raise ValueError(
            "the Q=0 audit requires microscopic polarity, Qxx, and Qxy fields"
        )
    return fields  # type: ignore[return-value]


def reconstruct_density_flux(
    result: PositionPDFResult,
    config: ProjectConfig,
    field: ChemicalField,
    case_index: int,
    time_index: int,
) -> DensityFluxDecomposition:
    """Reconstruct Eq. (exact-flux), including ``v_zeta Q grad(eta)``.

    All moment fields are the identically smoothed microscopic estimates, so
    subtraction and norm comparisons use one observation operator. The
    derivative is taken only between cells inside the disk.
    """

    polarity_x, polarity_y, nematic_xx, nematic_xy = _require_microscopic_fields(result)
    rho = result.density[case_index, time_index].astype(float)
    px = polarity_x[case_index, time_index].astype(float)
    py = polarity_y[case_index, time_index].astype(float)
    qxx = nematic_xx[case_index, time_index].astype(float)
    qxy = nematic_xy[case_index, time_index].astype(float)
    if (
        result.nematic_split_half_error_xx is None
        or result.nematic_split_half_error_xy is None
    ):
        q_error_xx = np.zeros_like(qxx)
        q_error_xy = np.zeros_like(qxy)
    else:
        q_error_xx = result.nematic_split_half_error_xx[case_index, time_index].astype(
            float
        )
        q_error_xy = result.nematic_split_half_error_xy[case_index, time_index].astype(
            float
        )

    coordinates = result.coordinates
    spacing = float(coordinates[1] - coordinates[0])
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    mask = x_grid * x_grid + y_grid * y_grid <= config.domain.radius**2
    points = np.stack((x_grid, y_grid), axis=-1)
    eta, gradient = field.value_and_gradient(points)
    eta = np.asarray(eta, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    delta_eta = eta - disk_area_average(field, config.domain.radius)
    gx = gradient[..., 0]
    gy = gradient[..., 1]

    response = config.responses.only_case(int(result.case_numbers[case_index]))
    translational_factor = config.kinetics.D_T0 + response.D_T_eta * delta_eta
    diffusion_potential = translational_factor * rho
    diffusion_potential += response.D_T_zeta * (px * gx + py * gy)
    potential_x = _masked_gradient(diffusion_potential, mask, spacing, axis=-1)
    potential_y = _masked_gradient(diffusion_potential, mask, spacing, axis=-2)

    q_gradient_x = qxx * gx + qxy * gy
    q_gradient_y = qxy * gx - qxx * gy
    q_flux_x = response.v_zeta * q_gradient_x
    q_flux_y = response.v_zeta * q_gradient_y
    q_error_gradient_x = q_error_xx * gx + q_error_xy * gy
    q_error_gradient_y = q_error_xy * gx - q_error_xx * gy
    q_flux_error_x = response.v_zeta * q_error_gradient_x
    q_flux_error_y = response.v_zeta * q_error_gradient_y
    isotropic_x = 0.5 * response.v_zeta * rho * gx
    isotropic_y = 0.5 * response.v_zeta * rho * gy
    local_speed = config.kinetics.v0 + response.v_eta * delta_eta
    total_x = local_speed * px + isotropic_x + q_flux_x - potential_x
    total_y = local_speed * py + isotropic_y + q_flux_y - potential_y
    zero = np.zeros_like(total_x)
    return DensityFluxDecomposition(
        total_x=np.where(mask, total_x, zero),
        total_y=np.where(mask, total_y, zero),
        q_x=np.where(mask, q_flux_x, zero),
        q_y=np.where(mask, q_flux_y, zero),
        q_sampling_error_x=np.where(mask, q_flux_error_x, zero),
        q_sampling_error_y=np.where(mask, q_flux_error_y, zero),
        isotropic_directional_x=np.where(mask, isotropic_x, zero),
        isotropic_directional_y=np.where(mask, isotropic_y, zero),
    )


def _field_l2(
    x: NDArray[np.float64],
    y: NDArray[np.float64] | None,
    mask: NDArray[np.bool_],
    cell_area: float,
) -> float:
    squared = x * x if y is None else x * x + y * y
    return float(np.sqrt(np.sum(squared[mask], dtype=np.float64) * cell_area))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 1.0e-30 else float("nan")


def _weighted_percentile(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    percentile: float,
) -> float:
    if values.size == 0 or float(np.sum(weights)) <= 0.0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    target = percentile / 100.0 * cumulative[-1]
    return float(
        sorted_values[min(np.searchsorted(cumulative, target), len(order) - 1)]
    )


def nematic_closure_rows(
    result: PositionPDFResult,
    config: ProjectConfig,
    field: ChemicalField,
    *,
    population_fraction: float = 0.01,
    polarity_fraction: float = 0.05,
) -> list[dict[str, float | int]]:
    """Return stable local/global measures of the discarded nematic field.

    Pointwise ``|Q|/rho`` is restricted to cells with at least
    ``population_fraction`` of the case/time peak density. Pointwise
    ``|Q|/|p|`` additionally requires ``|p|`` to exceed
    ``polarity_fraction`` of its 99th percentile on that population mask.
    Global L2 ratios use the full populated disk and remain the primary
    threshold-independent measures.
    """

    if not 0.0 < population_fraction < 1.0:
        raise ValueError("population_fraction must lie in (0, 1)")
    if not 0.0 < polarity_fraction < 1.0:
        raise ValueError("polarity_fraction must lie in (0, 1)")
    polarity_x, polarity_y, nematic_xx, nematic_xy = _require_microscopic_fields(result)
    coordinates = result.coordinates
    spacing = float(coordinates[1] - coordinates[0])
    cell_area = spacing**2
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    disk = x_grid * x_grid + y_grid * y_grid <= config.domain.radius**2
    baseline_fluxes = [
        reconstruct_density_flux(result, config, field, 0, time_index)
        for time_index in range(len(result.times))
    ]
    rows: list[dict[str, float | int]] = []

    for case_index, case_number in enumerate(result.case_numbers):
        for time_index, time in enumerate(result.times):
            rho = result.density[case_index, time_index].astype(float)
            px = polarity_x[case_index, time_index].astype(float)
            py = polarity_y[case_index, time_index].astype(float)
            qxx = nematic_xx[case_index, time_index].astype(float)
            qxy = nematic_xy[case_index, time_index].astype(float)
            q_norm = nematic_frobenius(qxx, qxy)
            if (
                result.nematic_split_half_error_xx is None
                or result.nematic_split_half_error_xy is None
            ):
                q_error_norm = np.zeros_like(q_norm)
            else:
                q_error_norm = nematic_frobenius(
                    result.nematic_split_half_error_xx[case_index, time_index].astype(
                        float
                    ),
                    result.nematic_split_half_error_xy[case_index, time_index].astype(
                        float
                    ),
                )
            p_norm = np.hypot(px, py)
            populated = disk & (rho >= population_fraction * float(np.max(rho[disk])))
            q_over_rho = q_norm[populated] / np.maximum(rho[populated], 1.0e-30)
            rho_weights = rho[populated]
            q_l2 = _field_l2(q_norm, None, populated, cell_area)
            q_sampling_error_l2 = _field_l2(q_error_norm, None, populated, cell_area)
            q_debiased_l2 = float(
                np.sqrt(max(q_l2 * q_l2 - q_sampling_error_l2**2, 0.0))
            )
            rho_l2 = _field_l2(rho, None, populated, cell_area)
            p_l2 = _field_l2(px, py, populated, cell_area)

            p99 = float(np.percentile(p_norm[populated], 99.0))
            polar = populated & (p_norm >= polarity_fraction * p99) & (p_norm > 0.0)
            q_over_p = q_norm[polar] / p_norm[polar]
            p_weights = p_norm[polar]

            flux = reconstruct_density_flux(
                result, config, field, case_index, time_index
            )
            baseline = baseline_fluxes[time_index]
            response_x = flux.total_x - baseline.total_x
            response_y = flux.total_y - baseline.total_y
            retained_response_x = response_x - flux.q_x
            retained_response_y = response_y - flux.q_y
            q0_total_x = flux.total_x - flux.q_x
            q0_total_y = flux.total_y - flux.q_y
            q_flux_l2 = _field_l2(flux.q_x, flux.q_y, populated, cell_area)
            q_flux_sampling_error_l2 = _field_l2(
                flux.q_sampling_error_x,
                flux.q_sampling_error_y,
                populated,
                cell_area,
            )
            q_flux_debiased_l2 = float(
                np.sqrt(max(q_flux_l2**2 - q_flux_sampling_error_l2**2, 0.0))
            )
            total_flux_l2 = _field_l2(flux.total_x, flux.total_y, populated, cell_area)
            q0_total_flux_l2 = _field_l2(q0_total_x, q0_total_y, populated, cell_area)
            response_flux_l2 = _field_l2(response_x, response_y, populated, cell_area)
            retained_flux_l2 = _field_l2(
                retained_response_x, retained_response_y, populated, cell_area
            )
            isotropic_flux_l2 = _field_l2(
                flux.isotropic_directional_x,
                flux.isotropic_directional_y,
                populated,
                cell_area,
            )
            rows.append(
                {
                    "case": int(case_number),
                    "time": float(time),
                    "q_frobenius_l2": q_l2,
                    "q_split_half_sampling_error_l2": q_sampling_error_l2,
                    "q_noise_debiased_l2": q_debiased_l2,
                    "rho_l2": rho_l2,
                    "polarity_l2": p_l2,
                    "q_over_rho_l2": _safe_ratio(q_l2, rho_l2),
                    "q_sampling_error_over_rho_l2": _safe_ratio(
                        q_sampling_error_l2, rho_l2
                    ),
                    "q_noise_debiased_over_rho_l2": _safe_ratio(q_debiased_l2, rho_l2),
                    "q_signal_to_sampling_error_l2": _safe_ratio(
                        q_l2, q_sampling_error_l2
                    ),
                    "q_over_p_l2": _safe_ratio(q_l2, p_l2),
                    "q_over_rho_weighted_mean": _safe_ratio(
                        float(np.sum(rho_weights * q_over_rho)),
                        float(np.sum(rho_weights)),
                    ),
                    "q_over_rho_weighted_p95": _weighted_percentile(
                        q_over_rho, rho_weights, 95.0
                    ),
                    "q_over_p_weighted_mean": _safe_ratio(
                        float(np.sum(p_weights * q_over_p)),
                        float(np.sum(p_weights)),
                    ),
                    "q_over_p_weighted_p95": _weighted_percentile(
                        q_over_p, p_weights, 95.0
                    ),
                    "populated_cell_count": int(np.count_nonzero(populated)),
                    "polar_cell_count": int(np.count_nonzero(polar)),
                    "q_density_flux_l2": q_flux_l2,
                    "q_density_flux_sampling_error_l2": q_flux_sampling_error_l2,
                    "q_density_flux_noise_debiased_l2": q_flux_debiased_l2,
                    "q_density_flux_signal_to_sampling_error": _safe_ratio(
                        q_flux_l2, q_flux_sampling_error_l2
                    ),
                    "q_density_flux_noise_debiased_over_exact_flux": _safe_ratio(
                        q_flux_debiased_l2, total_flux_l2
                    ),
                    "q_density_flux_noise_debiased_over_response_flux": _safe_ratio(
                        q_flux_debiased_l2, response_flux_l2
                    ),
                    "exact_density_flux_l2": total_flux_l2,
                    "q0_exact_density_flux_l2": q0_total_flux_l2,
                    "response_density_flux_l2": response_flux_l2,
                    "q0_response_density_flux_l2": retained_flux_l2,
                    "isotropic_directional_flux_l2": isotropic_flux_l2,
                    "q_flux_over_response_flux": _safe_ratio(
                        q_flux_l2, response_flux_l2
                    ),
                    "q_flux_over_exact_flux": _safe_ratio(q_flux_l2, total_flux_l2),
                    "q_flux_over_q0_exact_flux": _safe_ratio(
                        q_flux_l2, q0_total_flux_l2
                    ),
                    "q_flux_over_q0_response_flux": _safe_ratio(
                        q_flux_l2, retained_flux_l2
                    ),
                    "q_flux_over_isotropic_directional_flux": _safe_ratio(
                        q_flux_l2, isotropic_flux_l2
                    ),
                }
            )
    return rows
