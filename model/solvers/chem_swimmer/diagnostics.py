"""Concise mathematical and numerical checks before a long render."""

from __future__ import annotations

import os
import platform
from math import isfinite

from .cases import get_case
from .config import ProjectConfig
from .fields import ChemicalField
from .kinetics import macroscopic_coefficients
from .numerics import ResolutionPlan, build_resolution_plan
from .turning import create_turning_kernel


def scan_case(
    config: ProjectConfig,
    field: ChemicalField,
    case_number: int,
    grid_size: int = 51,
    heading_count: int = 32,
) -> dict[str, tuple[float, float]]:
    # ``grid_size`` and ``heading_count`` remain in the signature for callers
    # of the original API. The resolution-controlled scan derives its grid from the
    # field length and bounds zeta analytically over every heading.
    del grid_size, heading_count
    return build_resolution_plan(config, field, (case_number,)).ranges_by_case[
        case_number
    ]


def timestep_report(config: ProjectConfig, plan: ResolutionPlan) -> str:
    """Format every candidate timestep and the final automatic selection."""

    duration = config.simulation.duration / plan.scales.time
    step_count = round(duration / plan.dt)
    process_width = max(len(name) for name in plan.constraints)
    formulas = {
        "advection": "dx_target / v_max",
        "translational diffusion": "dx_target^2 / (4 D_T,max)",
        "rotational diffusion": "theta_rms,max^2 / (2 D_R,max)",
        "angular steering": "theta_max / |omega_theta|_max",
        "tumbling": "(lambda dt)_max / lambda_max",
    }
    formula_width = max(len(formula) for formula in formulas.values())
    rows = []
    for name, value in plan.constraints.items():
        candidate = f"{value:.6g}" if isfinite(value) else "unbounded"
        if name == plan.limiting_constraint:
            status = "<-- limiting process"
        elif not isfinite(value):
            status = "process inactive"
        else:
            status = ""
        rows.append(
            f"  {name:<{process_width}}  "
            f"{formulas[name]:<{formula_width}}  "
            f"{candidate:>14}  {status}"
        )
    return "\n".join(
        [
            "Spatial displacement target",
            (
                "  dx_target = max_spatial_step_fraction * ell_min"
                f" = {config.numerics.max_spatial_step_fraction:.6g}"
                f" * {plan.characteristic_length_star:.6g}"
                f" = {plan.target_spatial_step:.6g}"
            ),
            (
                "  ell_min is the smallest resolved dimensionless field length "
                "(capped by the unit-disk radius)."
            ),
            (
                "  The advective increment and translational RMS increment target "
                "v_max*dt <= dx_target and sqrt(4*D_T,max*dt) <= dx_target."
            ),
            (
                "  Gaussian increments are unbounded, so the RMS condition is an "
                "accuracy target, not a pathwise maximum."
            ),
            "",
            "Automatic timestep selection (all values are dimensionless)",
            (
                f"  {'Process':<{process_width}}  "
                f"{'Formula for dt':<{formula_width}}  "
                f"{'Candidate dt':>14}  Status"
            ),
            (f"  {'-' * process_width}  {'-' * formula_width}  {'-' * 14}  {'-' * 20}"),
            *rows,
            (
                f"  Selected safe limit : {plan.dt_limit:.6g} "
                f"({plan.limiting_constraint})"
            ),
            (
                f"  Simulation dt       : {plan.dt:.6g} "
                f"({step_count:,} steps to reach t_final={duration:.6g})"
            ),
        ]
    )


def theory_regime_report(
    config: ProjectConfig,
    plan: ResolutionPlan,
    case_numbers: list[int],
) -> str:
    """Report the small-response and scale-separation parameters of the closure."""

    base = config.kinetics
    turning_kernel = create_turning_kernel(config.turning)
    gamma1 = base.D_R0 + base.lambda0 * (1.0 - turning_kernel.P_turn)
    backgrounds = {
        "v": base.v0,
        "D_T": base.D_T0,
        "D_R": base.D_R0,
        "lambda": base.lambda0,
        # The steering drift has zero reference value.  Its natural smallness
        # scale is the angular relaxation rate, not division by zero.
        "angular_drift": gamma1,
    }
    case_modulations: dict[int, float] = {}
    for case_number in case_numbers:
        largest = 0.0
        for name, bounds in plan.ranges_by_case[case_number].items():
            background = backgrounds[name]
            if name == "angular_drift":
                absolute_change = max(abs(bounds[0]), abs(bounds[1]))
            else:
                absolute_change = max(
                    abs(bounds[0] - background),
                    abs(bounds[1] - background),
                )
            if background > 0.0:
                relative_change = absolute_change / background
            elif absolute_change == 0.0:
                relative_change = 0.0
            else:
                relative_change = float("inf")
            largest = max(largest, relative_change)
        case_modulations[case_number] = largest

    maximum_modulation = max(case_modulations.values())
    gamma2 = 4.0 * base.D_R0 + base.lambda0 * (1.0 - turning_kernel.P_2_turn)
    gamma_star = gamma1 * plan.scales.time
    speed_star = base.v0 * plan.scales.length / base.D_T0
    duration_star = config.simulation.duration / plan.scales.time
    relaxation_count = gamma_star * duration_star
    if gamma_star > 0.0:
        persistence_length = speed_star / gamma_star
        scale_separation = persistence_length / plan.characteristic_length_star
        persistence_text = (
            f"ell_p=v0/gamma1={persistence_length:.6g}, "
            f"delta=ell_p/ell_min={scale_separation:.6g}"
        )
    elif speed_star == 0.0:
        persistence_text = (
            "ell_p and delta=N/A (v0=gamma1=0; no active orientation closure)"
        )
    else:
        persistence_text = "ell_p and delta=unbounded (gamma1=0 with v0>0)"

    gamma2_star = gamma2 * plan.scales.time
    ell2 = float("inf") if gamma2_star <= 0.0 else speed_star / gamma2_star
    delta2 = ell2 / plan.characteristic_length_star
    if base.D_R0 > 0.0:
        higher_harmonic_text = (
            "all higher reference angular harmonics have positive relaxation "
            "rates because D_R0>0"
        )
    else:
        higher_harmonic_text = (
            "D_R0=0: the manuscript additionally requires every higher rate "
            "lambda0(1-P_k_turn) to be positive and fast relative to spatial evolution"
        )

    modulation_text = ", ".join(
        (
            f"C{case_number}={value:.3g}"
            if isfinite(value)
            else f"C{case_number}=unbounded"
        )
        for case_number, value in case_modulations.items()
    )
    maximum_text = (
        f"{maximum_modulation:.6g}" if isfinite(maximum_modulation) else "unbounded"
    )
    return "\n".join(
        [
            "Linear-theory regime diagnostics",
            (
                "  epsilon_X=max|X-X0|/X0="
                f"{maximum_text} (smaller than 1 is required; epsilon_X << 1 is desired)"
            ),
            f"  per-case epsilon_X: {modulation_text}",
            f"  {persistence_text} (the closure requires delta << 1)",
            (
                f"  P_turn={turning_kernel.P_turn:.6g}, "
                f"P_2_turn={turning_kernel.P_2_turn:.6g}, "
                f"gamma2={gamma2_star:.6g}, ell_2/ell_min={delta2:.6g}"
            ),
            f"  {higher_harmonic_text}",
            (
                f"  gamma1*t_final={relaxation_count:.6g} "
                "(adiabatic elimination requires this >> 1)"
            ),
            (
                "  Practical conservative targets: epsilon_X <= 0.15, "
                "delta <= 0.2, gamma1*t_final >= 10."
            ),
        ]
    )


def diagnostic_report(
    config: ProjectConfig, field: ChemicalField, case_numbers: list[int]
) -> str:
    plan = build_resolution_plan(config, field, case_numbers)
    turning_kernel = create_turning_kernel(config.turning)
    selected_engine = "python"
    if config.performance.engine != "python":
        from .accelerated import numba_available_for

        if numba_available_for(field, turning_kernel):
            selected_engine = "numba"
    thread_count = (
        config.performance.threads
        if config.performance.threads > 0
        else (os.cpu_count() or 1)
    )
    lines = [
        (
            f"Runtime: {platform.machine()}, engine={selected_engine}, "
            f"native_threads={thread_count}, progress={config.performance.progress}"
        ),
        (
            "Configured variables are already dimensionless: "
            f"R={plan.scales.length:.6g}, D_T0={config.kinetics.D_T0:.6g}, "
            f"and t_final={config.simulation.duration:.6g}. "
            "Physical measurements must be converted before editing the TOML."
        ),
        (
            f"Resolution: ell_min={plan.characteristic_length_star:.4g}, "
            f"scan={plan.scan_grid}x{plan.scan_grid}, "
            f"field_grid={plan.field_grid}x{plan.field_grid}, "
            f"Delta(x)={plan.grid_spacing_star:.4g}"
        ),
        "",
        theory_regime_report(config, plan, case_numbers),
        "",
        timestep_report(config, plan),
        "",
    ]
    for case_number in case_numbers:
        case = get_case(case_number)
        responses = config.responses.only_case(case_number)
        gamma1, effective_diffusivity, mobility = macroscopic_coefficients(
            config.kinetics, responses, turning_kernel.P_turn
        )
        ranges = plan.ranges_by_case[case_number]
        scale_factors = {
            "v": plan.scales.length / config.kinetics.D_T0,
            "D_T": 1.0 / config.kinetics.D_T0,
            "D_R": plan.scales.time,
            "lambda": plan.scales.time,
            "angular_drift": plan.scales.time,
        }
        if gamma1 == 0.0:
            theory_line = (
                "  gamma1=0; D_eff=N/A, kappa=N/A "
                "(long-time isotropic theory requires orientational relaxation)"
            )
        else:
            theory_line = (
                f"  gamma1={gamma1 * plan.scales.time:.6g}, "
                f"D_eff={effective_diffusivity / config.kinetics.D_T0:.6g}, "
                f"kappa={mobility / config.kinetics.D_T0:+.6g}"
            )
        lines.extend(
            [
                f"Case {case.number}: {case.title} ({case.math_label})",
                theory_line,
                (
                    "  dimensionless safe ranges: "
                    + ", ".join(
                        f"{name}=["
                        f"{bounds[0] * scale_factors[name]:.5g}, "
                        f"{bounds[1] * scale_factors[name]:.5g}]"
                        for name, bounds in ranges.items()
                    )
                ),
                (
                    f"  max(lambda dt)~"
                    f"{ranges['lambda'][1] * plan.scales.time * plan.dt:.4g}"
                ),
            ]
        )
    return "\n".join(lines)
