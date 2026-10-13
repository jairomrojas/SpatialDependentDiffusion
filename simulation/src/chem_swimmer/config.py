"""Typed configuration loaded from a single human-editable TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from math import isclose, isfinite
from pathlib import Path
from typing import Any

from .cases import get_case


@dataclass(frozen=True)
class SimulationSettings:
    duration: float
    dt: float | str
    save_every: int
    seed: int
    max_lambda_dt: float = 0.05


@dataclass(frozen=True)
class NumericsSettings:
    """Accuracy targets used to resolve an automatic mesh scan and timestep."""

    max_rotation_rms: float = 0.10
    max_spatial_step_fraction: float = 0.05
    scan_points_per_length: int = 10
    min_scan_grid: int = 101
    max_scan_grid: int = 1601
    characteristic_length: float = 0.0
    resolved: bool = False


@dataclass(frozen=True)
class DomainSettings:
    radius: float


@dataclass(frozen=True)
class InitialState:
    position: tuple[float, float]
    heading: float


@dataclass(frozen=True)
class BackgroundKinetics:
    v0: float
    D_T0: float
    D_R0: float
    lambda0: float


@dataclass(frozen=True)
class Responses:
    D_T_eta: float
    D_R_eta: float
    v_eta: float
    lambda_eta: float
    D_T_zeta: float
    D_R_zeta: float
    v_zeta: float
    lambda_zeta: float
    chi_theta: float

    @classmethod
    def zeros(cls) -> Responses:
        return cls(*(0.0 for _ in range(9)))

    def only_case(self, case_number: int) -> Responses:
        case = get_case(case_number)
        if case_number == 0:
            return self.zeros()
        values = {name: 0.0 for name in self.__dataclass_fields__}
        values[case.response_name] = getattr(self, case.response_name)
        return Responses(**values)


@dataclass(frozen=True)
class VideoSettings:
    fps: int
    dpi: int
    trail_time: float
    field_grid: int | str
    codec: str
    bitrate_kbps: int


@dataclass(frozen=True)
class StatisticsSettings:
    ensemble_size: int
    workers: int
    show_standard_error: bool


@dataclass(frozen=True)
class DensitySettings:
    ensemble_size: int | str
    initial_position_distribution: str
    initial_heading_distribution: str
    grid_size: int
    bandwidth_fraction: float
    min_samples_per_kernel: int
    batch_size: int
    time_slices: int
    video_fps: int
    frame_size: int
    max_video_seconds: float
    difference_bandwidth_fraction: float | str = "auto"
    difference_case_percentile: float = 99.5
    initial_gaussian_sigma: float | str = "auto"


@dataclass(frozen=True)
class PerformanceSettings:
    engine: str
    threads: int
    msd_batch_size: int
    progress: bool


@dataclass(frozen=True)
class MacroSettings:
    """Controls for the polar-closed density--polarity finite-volume system."""

    cfl_safety: float = 0.90


@dataclass(frozen=True)
class ProjectConfig:
    simulation: SimulationSettings
    numerics: NumericsSettings
    domain: DomainSettings
    initial: InitialState
    kinetics: BackgroundKinetics
    turning: dict[str, Any]
    responses: Responses
    field: dict[str, Any]
    video: VideoSettings
    statistics: StatisticsSettings
    density: DensitySettings
    performance: PerformanceSettings
    macro: MacroSettings
    source_path: Path

    def with_overrides(
        self,
        *,
        duration: float | None = None,
        dt: float | str | None = None,
        seed: int | None = None,
        fps: int | None = None,
        engine: str | None = None,
        threads: int | None = None,
        progress: bool | None = None,
    ) -> ProjectConfig:
        simulation = replace(
            self.simulation,
            duration=self.simulation.duration if duration is None else duration,
            dt=self.simulation.dt if dt is None else dt,
            seed=self.simulation.seed if seed is None else seed,
        )
        video = replace(
            self.video,
            fps=self.video.fps if fps is None else fps,
        )
        performance = replace(
            self.performance,
            engine=self.performance.engine if engine is None else engine,
            threads=self.performance.threads if threads is None else threads,
            progress=self.performance.progress if progress is None else progress,
        )
        result = replace(
            self,
            simulation=simulation,
            video=video,
            performance=performance,
            numerics=replace(self.numerics, resolved=False),
        )
        result.validate()
        return result

    def validate(self) -> None:
        def require_finite(label: str, *values: float) -> None:
            if not all(isfinite(float(value)) for value in values):
                raise ValueError(f"{label} must contain only finite numbers")

        sim = self.simulation
        require_finite("simulation.duration", sim.duration)
        if sim.duration <= 0:
            raise ValueError("simulation.duration must be positive")
        if isinstance(sim.dt, str):
            if sim.dt.lower() != "auto":
                raise ValueError("simulation.dt must be positive or 'auto'")
        else:
            require_finite("simulation.dt", sim.dt)
            if sim.dt <= 0:
                raise ValueError("simulation.dt must be positive or 'auto'")
        require_finite("simulation.max_lambda_dt", sim.max_lambda_dt)
        if not isinstance(sim.seed, int) or isinstance(sim.seed, bool) or sim.seed < 0:
            raise ValueError("simulation.seed must be a non-negative integer")
        if sim.save_every < 1:
            raise ValueError("simulation.save_every must be at least 1")
        if not 0 < sim.max_lambda_dt <= 0.25:
            raise ValueError("simulation.max_lambda_dt must lie in (0, 0.25]")

        numerics = self.numerics
        require_finite(
            "numerics accuracy settings",
            numerics.max_rotation_rms,
            numerics.max_spatial_step_fraction,
            numerics.characteristic_length,
        )
        if not 0 < numerics.max_rotation_rms <= 0.5:
            raise ValueError("numerics.max_rotation_rms must lie in (0, 0.5]")
        if not 0 < numerics.max_spatial_step_fraction <= 0.25:
            raise ValueError("numerics.max_spatial_step_fraction must lie in (0, 0.25]")
        if numerics.scan_points_per_length < 4:
            raise ValueError("numerics.scan_points_per_length must be at least 4")
        if numerics.min_scan_grid < 33:
            raise ValueError("numerics.min_scan_grid must be at least 33")
        if numerics.max_scan_grid < numerics.min_scan_grid:
            raise ValueError(
                "numerics.max_scan_grid must be at least numerics.min_scan_grid"
            )
        if numerics.characteristic_length < 0:
            raise ValueError("numerics.characteristic_length cannot be negative")

        if len(self.initial.position) != 2:
            raise ValueError("initial.position must contain exactly two numbers")
        require_finite(
            "domain and initial state",
            self.domain.radius,
            *self.initial.position,
            self.initial.heading,
        )
        if self.domain.radius <= 0:
            raise ValueError("domain.radius must be positive")
        if not isclose(self.domain.radius, 1.0):
            raise ValueError(
                "domain.radius must equal 1 in the dimensionless formulation"
            )
        x0, y0 = self.initial.position
        if x0 * x0 + y0 * y0 >= self.domain.radius**2:
            raise ValueError("initial.position must lie strictly inside the disk")

        base = self.kinetics
        require_finite(
            "background kinetics",
            base.v0,
            base.D_T0,
            base.D_R0,
            base.lambda0,
        )
        require_finite(
            "response coefficients",
            *(
                getattr(self.responses, name)
                for name in self.responses.__dataclass_fields__
            ),
        )
        if min(base.v0, base.D_R0, base.lambda0) < 0:
            raise ValueError("all background kinetic parameters must be non-negative")
        if base.D_T0 <= 0:
            raise ValueError(
                "kinetics.D_T0 must be positive because the time scale is R^2/D_T0"
            )
        if not isclose(base.D_T0, 1.0):
            raise ValueError("kinetics.D_T0 must equal 1 in diffusion-time units")
        from .turning import create_turning_kernel

        create_turning_kernel(self.turning)

        require_finite("video.trail_time", self.video.trail_time)
        if self.video.fps < 1 or self.video.dpi < 50:
            raise ValueError(
                "video.fps must be positive and video.dpi must be at least 50"
            )
        if self.video.bitrate_kbps < 1:
            raise ValueError("video.bitrate_kbps must be positive")
        if not self.video.codec:
            raise ValueError("video.codec cannot be empty")
        field_grid = self.video.field_grid
        if isinstance(field_grid, str):
            if field_grid.lower() != "auto":
                raise ValueError("video.field_grid must be at least 32 or 'auto'")
        elif field_grid < 32:
            raise ValueError("video.field_grid must be at least 32 or 'auto'")
        if self.video.trail_time < 0:
            raise ValueError("video trail and field-grid settings are invalid")
        if self.statistics.ensemble_size < 1:
            raise ValueError("statistics.ensemble_size must be at least 1")
        if self.statistics.workers < 0:
            raise ValueError("statistics.workers cannot be negative")
        density = self.density
        if isinstance(density.ensemble_size, str):
            if density.ensemble_size.lower() != "auto":
                raise ValueError("density.ensemble_size must be positive or 'auto'")
        elif density.ensemble_size < 1:
            raise ValueError("density.ensemble_size must be positive or 'auto'")
        if density.initial_position_distribution not in {
            "uniform",
            "fixed",
            "gaussian",
        }:
            raise ValueError(
                "density.initial_position_distribution must be "
                "'uniform', 'fixed', or 'gaussian'"
            )
        if isinstance(density.initial_gaussian_sigma, str):
            if density.initial_gaussian_sigma.lower() != "auto":
                raise ValueError(
                    "density.initial_gaussian_sigma must be positive or 'auto'"
                )
        elif density.initial_gaussian_sigma <= 0:
            raise ValueError(
                "density.initial_gaussian_sigma must be positive or 'auto'"
            )
        if density.initial_heading_distribution not in {"isotropic", "fixed"}:
            raise ValueError(
                "density.initial_heading_distribution must be 'isotropic' or 'fixed'"
            )
        if density.grid_size < 33:
            raise ValueError("density.grid_size must be at least 33")
        if not 0 < density.bandwidth_fraction <= 0.25:
            raise ValueError("density.bandwidth_fraction must lie in (0, 0.25]")
        require_finite(
            "density settings",
            density.bandwidth_fraction,
            density.max_video_seconds,
        )
        difference_bandwidth = density.difference_bandwidth_fraction
        if isinstance(difference_bandwidth, str):
            if difference_bandwidth.lower() != "auto":
                raise ValueError(
                    "density.difference_bandwidth_fraction must be a number or 'auto'"
                )
        else:
            require_finite(
                "density.difference_bandwidth_fraction",
                difference_bandwidth,
            )
            if not density.bandwidth_fraction <= difference_bandwidth <= 0.25:
                raise ValueError(
                    "density.difference_bandwidth_fraction must lie between "
                    "density.bandwidth_fraction and 0.25"
                )
        require_finite(
            "density.difference_case_percentile",
            density.difference_case_percentile,
        )
        if not 90.0 <= density.difference_case_percentile <= 100.0:
            raise ValueError("density.difference_case_percentile must lie in [90, 100]")
        if density.bandwidth_fraction < 2.0 / density.grid_size:
            raise ValueError(
                "density bandwidth must span at least two density-grid cells"
            )
        if density.min_samples_per_kernel < 5:
            raise ValueError("density.min_samples_per_kernel must be at least 5")
        if density.batch_size < 1:
            raise ValueError("density.batch_size must be at least 1")
        if density.time_slices < 2:
            raise ValueError("density.time_slices must be at least 2")
        if density.video_fps < 1:
            raise ValueError("density.video_fps must be positive")
        if density.frame_size < 600 or density.frame_size % 2:
            raise ValueError(
                "density.frame_size must be an even integer of at least 600"
            )
        if not 0 < density.max_video_seconds <= 10:
            raise ValueError("density.max_video_seconds must lie in (0, 10]")
        if self.performance.engine not in {"auto", "numba", "python"}:
            raise ValueError("performance.engine must be auto, numba, or python")
        if self.performance.threads < 0:
            raise ValueError("performance.threads cannot be negative")
        if self.performance.msd_batch_size < 1:
            raise ValueError("performance.msd_batch_size must be at least 1")

        macro = self.macro
        require_finite("macro settings", macro.cfl_safety)
        if not 0 < macro.cfl_safety < 1:
            raise ValueError("macro.cfl_safety must lie in (0, 1)")


def load_config(path: str | Path) -> ProjectConfig:
    source_path = Path(path).expanduser().resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    allowed_sections = {
        "simulation",
        "numerics",
        "domain",
        "initial",
        "kinetics",
        "turning",
        "responses",
        "field",
        "video",
        "statistics",
        "density",
        "performance",
        "macro",
    }
    unknown_sections = sorted(set(raw) - allowed_sections)
    if unknown_sections:
        raise ValueError(
            "unknown top-level configuration section(s): " + ", ".join(unknown_sections)
        )

    if "boundary" in raw.get("domain", {}):
        raise ValueError(
            "domain.boundary has been removed: the manuscript sliding wall is "
            "the only supported boundary condition"
        )
    if "a1" in raw.get("kinetics", {}):
        raise ValueError(
            "kinetics.a1 has been replaced by the [turning] kernel; use "
            "kind='wrapped_normal' with P_turn=<old a1> to reproduce the old law"
        )
    legacy_responses = sorted(
        set(raw.get("responses", {}))
        & {
            "D_T_c",
            "D_R_c",
            "v_c",
            "lambda_c",
            "D_T_S",
            "D_R_S",
            "v_S",
            "lambda_S",
        }
    )
    if legacy_responses:
        raise ValueError(
            "legacy c/S response notation is no longer accepted; rename "
            "coefficients to the manuscript eta/zeta notation: "
            + ", ".join(legacy_responses)
        )
    if "turning" not in raw:
        raise ValueError(
            "a [turning] table is required; see config/default.toml for the "
            "wrapped-normal, discrete, tabulated, and user-kernel interface"
        )

    density_raw = {
        "ensemble_size": "auto",
        "initial_position_distribution": "uniform",
        "initial_heading_distribution": "isotropic",
        "grid_size": 81,
        "bandwidth_fraction": 0.05,
        "min_samples_per_kernel": 40,
        "batch_size": 32,
        "time_slices": 301,
        "video_fps": 10,
        "frame_size": 1200,
        "max_video_seconds": 10.0,
        "difference_bandwidth_fraction": "auto",
        "difference_case_percentile": 99.5,
        "initial_gaussian_sigma": "auto",
    }
    density_raw.update(raw.get("density", {}))
    macro_raw = {
        "cfl_safety": 0.90,
    }
    macro_raw.update(raw.get("macro", {}))
    numerics_raw = dict(raw.get("numerics", {}))
    video_raw = dict(raw["video"])
    if "trail_time" not in video_raw:
        video_raw["trail_time"] = video_raw.pop("trail_seconds")
    else:
        video_raw.pop("trail_seconds", None)

    config = ProjectConfig(
        simulation=SimulationSettings(**raw["simulation"]),
        numerics=NumericsSettings(**numerics_raw),
        domain=DomainSettings(**raw["domain"]),
        initial=InitialState(
            position=tuple(float(x) for x in raw["initial"]["position"]),
            heading=float(raw["initial"]["heading"]),
        ),
        kinetics=BackgroundKinetics(**raw["kinetics"]),
        turning=dict(raw["turning"]),
        responses=Responses(**raw["responses"]),
        field=dict(raw["field"]),
        video=VideoSettings(**video_raw),
        statistics=StatisticsSettings(**raw["statistics"]),
        density=DensitySettings(**density_raw),
        performance=PerformanceSettings(**raw["performance"]),
        macro=MacroSettings(**macro_raw),
        source_path=source_path,
    )
    config.validate()
    return config
