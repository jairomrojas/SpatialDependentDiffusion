"""The eight canonical response mechanisms from Theory.pdf."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalCase:
    number: int
    response_name: str
    math_label: str
    title: str


CASES: dict[int, CanonicalCase] = {
    0: CanonicalCase(0, "", "baseline", "No chemical response"),
    1: CanonicalCase(1, "D_T_c", r"$D_T^c$", "Concentration-dependent diffusion"),
    2: CanonicalCase(
        2, "D_R_c", r"$D_R^c$", "Concentration-dependent rotational noise"
    ),
    3: CanonicalCase(3, "v_c", r"$v^c$", "Orthokinesis"),
    4: CanonicalCase(4, "lambda_c", r"$\lambda^c$", "Concentration-dependent tumbling"),
    5: CanonicalCase(5, "D_T_S", r"$D_T^S$", "Alignment-gated positional noise"),
    6: CanonicalCase(6, "D_R_S", r"$D_R^S$", "Alignment-gated rotational noise"),
    7: CanonicalCase(7, "v_S", r"$v^S$", "Alignment-gated propulsion"),
    8: CanonicalCase(8, "lambda_S", r"$\lambda^S$", "Local run-and-tumble surrogate"),
}


def get_case(number: int) -> CanonicalCase:
    try:
        return CASES[number]
    except KeyError as exc:
        raise ValueError(f"case must be an integer from 0 to 8, got {number}") from exc
