"""The nine canonical response mechanisms from the manuscript."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalCase:
    number: int
    response_name: str
    math_label: str
    title: str


CASES: dict[int, CanonicalCase] = {
    0: CanonicalCase(0, "", "baseline", "No nutrient response"),
    1: CanonicalCase(1, "D_T_eta", r"$D_T^{(\eta)}$", "Nutrient-dependent diffusion"),
    2: CanonicalCase(
        2, "D_R_eta", r"$D_R^{(\eta)}$", "Nutrient-dependent rotational noise"
    ),
    3: CanonicalCase(3, "v_eta", r"$v^{(\eta)}$", "Orthokinesis"),
    4: CanonicalCase(
        4, "lambda_eta", r"$\lambda^{(\eta)}$", "Nutrient-dependent tumbling"
    ),
    5: CanonicalCase(5, "D_T_zeta", r"$D_T^{(\zeta)}$", "Directional positional noise"),
    6: CanonicalCase(6, "D_R_zeta", r"$D_R^{(\zeta)}$", "Directional rotational noise"),
    7: CanonicalCase(7, "v_zeta", r"$v^{(\zeta)}$", "Directional propulsion"),
    8: CanonicalCase(
        8, "lambda_zeta", r"$\lambda^{(\zeta)}$", "Local run-and-tumble surrogate"
    ),
    9: CanonicalCase(
        9,
        "chi_theta",
        r"$\chi_\theta$",
        "Continuous angular steering",
    ),
}

# Stable publication-facing filenames shared by plotting and CLI code.
STRATEGY_FIGURE_STEMS: dict[int, str] = {
    1: "figure-3-case-1-validation",
    2: "figure-s01-case-2-level-dependent-rotational-diffusion",
    3: "figure-s02-case-3-level-dependent-speed",
    4: "figure-s03-case-4-level-dependent-tumbling",
    5: "figure-s04-case-5-directional-translational-diffusion",
    6: "figure-s05-case-6-directional-rotational-diffusion",
    7: "figure-s06-case-7-directional-speed",
    8: "figure-s07-case-8-directional-tumbling",
    9: "figure-s08-case-9-continuous-steering",
}


def get_case(number: int) -> CanonicalCase:
    try:
        return CASES[number]
    except KeyError as exc:
        raise ValueError(f"case must be an integer from 0 to 9, got {number}") from exc
