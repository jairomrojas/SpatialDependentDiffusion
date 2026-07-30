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


def scan_case(
    config: ProjectConfig,
    field: ChemicalField,
    case_number: int,
    grid_size: int = 51,
    heading_count: int = 32,
) -> dict[str, tuple[float, float]]:
    # ``grid_size`` and ``heading_count`` remain in the signature for callers
    # of the original API. The resolution-controlled scan derives its grid from the
    # field length and bounds alignment analytically over every heading.
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
    backgrounds = {
        "v": base.v0,
        "D_T": base.D_T0,
        "D_R": base.D_R0,
        "lambda": base.lambda0,
    }
    case_modulations: dict[int, float] = {}
    for case_number in case_numbers:
        largest = 0.0
        for name, bounds in plan.ranges_by_case[case_number].items():
            background = backgrounds[name]
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
    gamma0 = base.D_R0 + base.lambda0 * (1.0 - base.a1)
    gamma_star = gamma0 * plan.scales.time
    speed_star = base.v0 * plan.scales.length / base.D_T0
    duration_star = config.simulation.duration / plan.scales.time
    relaxation_count = gamma_star * duration_star
    if gamma_star > 0.0:
        persistence_length = speed_star / gamma_star
        scale_separation = persistence_length / plan.characteristic_length_star
        persistence_text = (
            f"ell_p=v0/gamma0={persistence_length:.6g}, "
            f"delta=ell_p/ell_min={scale_separation:.6g}"
        )
    elif speed_star == 0.0:
        persistence_text = (
            "ell_p and delta=N/A (v0=gamma0=0; no active orientation closure)"
        )
    else:
        persistence_text = "ell_p and delta=unbounded (gamma0=0 with v0>0)"

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
                "  eta=max|X-X0|/X0="
                f"{maximum_text} (smaller than 1 is required; eta << 1 is desired)"
            ),
            f"  per-case eta: {modulation_text}",
            f"  {persistence_text} (the closure requires delta << 1)",
            (
                f"  gamma0*t_final={relaxation_count:.6g} "
                "(adiabatic elimination requires this >> 1)"
            ),
            (
                "  Practical conservative targets: eta <= 0.15, "
                "delta <= 0.2, gamma0*t_final >= 10."
            ),
        ]
    )


def diagnostic_report(
    config: ProjectConfig, field: ChemicalField, case_numbers: list[int]
) -> str:
    plan = build_resolution_plan(config, field, case_numbers)
    selected_engine = "python"
    if config.performance.engine != "python":
        from .accelerated import numba_available_for

        if numba_available_for(field):
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
        gamma0, effective_diffusivity, mobility = macroscopic_coefficients(
            config.kinetics, responses
        )
        ranges = plan.ranges_by_case[case_number]
        scale_factors = {
            "v": plan.scales.length / config.kinetics.D_T0,
            "D_T": 1.0 / config.kinetics.D_T0,
            "D_R": plan.scales.time,
            "lambda": plan.scales.time,
        }
        if gamma0 == 0.0:
            theory_line = (
                "  gamma0=0; D_eff=N/A, kappa=N/A "
                "(long-time isotropic theory requires orientational relaxation)"
            )
        else:
            theory_line = (
                f"  gamma0={gamma0 * plan.scales.time:.6g}, "
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
