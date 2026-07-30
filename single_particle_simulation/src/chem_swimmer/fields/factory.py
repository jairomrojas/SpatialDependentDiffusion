"""Construct a chemical field from the ``[field]`` TOML table."""

from __future__ import annotations

from typing import Any

import numpy as np

from .analytic import GaussianField, LinearField, MultiGaussianField
from .base import ChemicalField
from .user_field import UserField


def _pair(value: Any, name: str) -> tuple[float, float]:
    try:
        pair = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two numbers") from exc
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    result = float(pair[0]), float(pair[1])
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite numbers")
    return result


def _reject_unknown(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unknown {context} setting(s): {joined}")


def _gaussian_from_dict(
    data: dict[str, Any], baseline: float | None = None
) -> GaussianField:
    allowed = {"amplitude", "center", "sigma", "angle"}
    if baseline is None:
        allowed.add("baseline")
    _reject_unknown(data, allowed, "Gaussian field")
    return GaussianField(
        baseline=float(data.get("baseline", 0.0) if baseline is None else baseline),
        amplitude=float(data["amplitude"]),
        center=_pair(data.get("center", [0.0, 0.0]), "field.center"),
        sigma=_pair(data["sigma"], "field.sigma"),
        angle=float(data.get("angle", 0.0)),
    )


def create_field(settings: dict[str, Any]) -> ChemicalField:
    data = dict(settings)
    kind = str(data.pop("kind")).lower()

    if kind == "gaussian":
        return _gaussian_from_dict(data)
    if kind == "linear":
        _reject_unknown(
            data,
            {"baseline", "gradient", "gradient_vector", "center"},
            "linear field",
        )
        if "gradient" in data and "gradient_vector" in data:
            raise ValueError("use only one of field.gradient and field.gradient_vector")
        gradient = data.pop("gradient", data.pop("gradient_vector", None))
        if gradient is None:
            raise ValueError("a linear field requires field.gradient = [gx, gy]")
        return LinearField(
            baseline=float(data["baseline"]),
            gradient_vector=_pair(gradient, "field.gradient"),
            center=_pair(data.get("center", [0.0, 0.0]), "field.center"),
        )
    if kind == "multi_gaussian":
        _reject_unknown(data, {"baseline", "sources"}, "multi-Gaussian field")
        baseline = float(data["baseline"])
        sources = tuple(
            _gaussian_from_dict(dict(source), baseline=0.0)
            for source in data["sources"]
        )
        return MultiGaussianField(baseline=baseline, sources=sources)
    if kind == "user":
        _reject_unknown(data, {"baseline"}, "user field")
        return UserField(baseline=float(data.get("baseline", 1.0)))

    raise ValueError(
        f"unknown field kind {kind!r}; choose gaussian, linear, multi_gaussian, or user"
    )
