from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chem_swimmer.accelerated import state_batch_numba
from chem_swimmer.angles import sample_tumble_heading
from chem_swimmer.boundary import advance_in_disk
from chem_swimmer.cli import build_parser
from chem_swimmer.config import Responses, load_config
from chem_swimmer.density import (
    PositionPDFResult,
    compute_position_pdf,
    load_position_pdf,
    recommended_ensemble_size,
    resolve_density_ensemble_size,
    save_position_pdf,
)
from chem_swimmer.fields import create_field
from chem_swimmer.fields.analytic import GaussianField
from chem_swimmer.fields.base import positions_array
from chem_swimmer.fields.user_field import UserField
from chem_swimmer.kinetics import macroscopic_coefficients
from chem_swimmer.macroscopic import (
    build_macroscopic_plan,
    compute_macroscopic_pdf,
)
from chem_swimmer.noise import Innovations
from chem_swimmer.numerics import build_resolution_plan, resolve_numerics
from chem_swimmer.simulator import simulate, simulate_all_cases
from chem_swimmer.statistics import compute_ensemble_msd, free_space_baseline_msd
from chem_swimmer.visualization.density_video import (
    _case_specific_linear_limits,
    _normalize_linear_difference,
    synchronized_case_linear_limits,
    synchronized_global_linear_limits,
)
from chem_swimmer.visualization.field_plot import render_chemical_field
from chem_swimmer.visualization.inference_plot import (
    render_microscopic_mechanism_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FieldTests(unittest.TestCase):
    def test_gaussian_gradient_matches_finite_difference(self) -> None:
        field = GaussianField(
            baseline=1.0,
            amplitude=1.7,
            center=(0.4, -0.2),
            sigma=(1.1, 2.3),
            angle=0.7,
        )
        point = np.array([0.8, 0.3])
        analytic = field.gradient(point)
        h = 1.0e-6
        finite_difference = np.array(
            [
                (field.value(point + [h, 0.0]) - field.value(point - [h, 0.0]))
                / (2.0 * h),
                (field.value(point + [0.0, h]) - field.value(point - [0.0, h]))
                / (2.0 * h),
            ]
        )
        np.testing.assert_allclose(
            analytic, finite_difference, rtol=1.0e-8, atol=1.0e-9
        )

    def test_example_user_field_is_smooth_at_the_origin(self) -> None:
        field = UserField()
        origin = np.zeros(2)
        np.testing.assert_allclose(field.gradient(origin), np.zeros(2), atol=1.0e-9)
        directions = np.asarray(
            ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]),
            dtype=float,
        )
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        near_origin = np.asarray(field.value(1.0e-7 * directions))
        np.testing.assert_allclose(
            near_origin,
            np.full(4, field.baseline + 1.0),
            rtol=0.0,
            atol=1.0e-13,
        )

    def test_invalid_field_inputs_fail_early(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        with self.assertRaisesRegex(ValueError, "unknown Gaussian field"):
            create_field({**config.field, "sigm_typo": [1.0, 1.0]})
        with self.assertRaisesRegex(ValueError, "finite"):
            GaussianField(
                baseline=1.0,
                amplitude=np.nan,
                center=(0.0, 0.0),
                sigma=(1.0, 1.0),
            )
        with self.assertRaisesRegex(ValueError, "final dimension 2"):
            positions_array(1.0)

    def test_unknown_top_level_config_section_is_rejected(self) -> None:
        source = PROJECT_ROOT / "configs" / "default.toml"
        with TemporaryDirectory() as directory:
            modified = Path(directory) / "unknown_section.toml"
            modified.write_text(
                source.read_text(encoding="utf-8") + "\n[simulaton_typo]\nvalue = 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown top-level"):
                load_config(modified)


class TumbleTests(unittest.TestCase):
    def test_sampled_mean_cosine_is_a1(self) -> None:
        rng = np.random.default_rng(21)
        count = 80_000
        normal = rng.standard_normal(count)
        uniform = rng.random(count)
        for a1 in (-0.45, 0.0, 0.3, 0.85):
            turns = np.array(
                [
                    sample_tumble_heading(0.0, a1, z, u)
                    for z, u in zip(normal, uniform, strict=True)
                ]
            )
            self.assertAlmostEqual(float(np.mean(np.cos(turns))), a1, delta=0.008)


class BoundaryTests(unittest.TestCase):
    def test_sliding_reflects_position_not_heading(self) -> None:
        result = advance_in_disk(
            np.array([0.9, 0.0]),
            np.array([0.3, 0.0]),
            heading=0.0,
            radius=1.0,
            mode="sliding",
            wall_uniforms=np.full(16, 0.5),
        )
        np.testing.assert_allclose(result.position, [0.8, 0.0], atol=1.0e-11)
        self.assertAlmostEqual(result.heading, 0.0)
        self.assertEqual(result.hits, 1)

    def test_specular_reflects_heading(self) -> None:
        result = advance_in_disk(
            np.array([0.9, 0.0]),
            np.array([0.3, 0.0]),
            heading=0.0,
            radius=1.0,
            mode="specular",
            wall_uniforms=np.full(16, 0.5),
        )
        self.assertAlmostEqual(result.heading, np.pi)


class TheoryCoefficientTests(unittest.TestCase):
    def test_cases_match_table_2(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        base = replace(
            config.kinetics,
            v0=0.1,
            D_R0=0.1,
            lambda0=0.1,
        )
        gamma0 = base.D_R0 + base.lambda0 * (1.0 - base.a1)

        expected = {
            1: -config.responses.D_T_c,
            2: 0.0,
            3: -base.v0 * config.responses.v_c / (2.0 * gamma0),
            4: 0.0,
            5: 0.0,
            6: -base.v0 * config.responses.D_R_S / (2.0 * gamma0),
            7: config.responses.v_S / 2.0,
            8: -base.v0 * (1.0 - base.a1) * config.responses.lambda_S / (2.0 * gamma0),
        }
        for case_number, expected_chi in expected.items():
            _, _, chi = macroscopic_coefficients(
                base, config.responses.only_case(case_number)
            )
            self.assertAlmostEqual(chi, expected_chi)

    def test_zero_orientation_relaxation_is_valid_for_simulation(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        base = replace(
            config.kinetics,
            v0=0.0,
            D_R0=0.0,
            lambda0=0.0,
        )
        gamma0, effective_diffusivity, mobility = macroscopic_coefficients(
            base,
            Responses.zeros(),
        )
        self.assertEqual(gamma0, 0.0)
        self.assertTrue(np.isnan(effective_diffusivity))
        self.assertTrue(np.isnan(mobility))
        passive = replace(
            config.with_overrides(duration=1.0e-4),
            kinetics=base,
            responses=Responses.zeros(),
        )
        passive.validate()
        trajectory = simulate(passive, 0)
        self.assertTrue(np.all(trajectory.headings == passive.initial.heading))
        times = np.array([0.0, 0.25, 1.0])
        np.testing.assert_array_equal(
            free_space_baseline_msd(passive, times),
            4.0 * times,
        )
        slowly_relaxing = replace(
            passive,
            kinetics=replace(base, v0=0.2, D_R0=1.0e-12),
        )
        np.testing.assert_allclose(
            free_space_baseline_msd(slowly_relaxing, times),
            4.0 * times + 0.2**2 * times**2,
            rtol=1.0e-12,
            atol=1.0e-14,
        )


class SimulatorTests(unittest.TestCase):
    def test_short_run_is_reproducible_and_stays_in_disk(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml").with_overrides(
            duration=1.6e-6
        )
        first = simulate(config, 8)
        second = simulate(config, 8)
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.headings, second.headings)
        radii = np.linalg.norm(first.positions, axis=1)
        self.assertTrue(np.all(radii < 1.0))
        self.assertEqual(first.times[-1], 1.6e-6)

    def test_trajectory_outputs_are_dimensionless(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml").with_overrides(
            duration=8.0e-7
        )
        trajectory = simulate(config, 0)
        time_scale = config.domain.radius**2 / config.kinetics.D_T0
        self.assertEqual(
            trajectory.times[-1],
            config.simulation.duration / time_scale,
        )
        self.assertAlmostEqual(
            trajectory.speed[0],
            config.kinetics.v0 * config.domain.radius / config.kinetics.D_T0,
        )
        self.assertAlmostEqual(trajectory.translational_diffusivity[0], 1.0)
        self.assertAlmostEqual(
            trajectory.rotational_diffusivity[0],
            config.kinetics.D_R0 * time_scale,
        )
        self.assertAlmostEqual(
            trajectory.tumble_rate[0],
            config.kinetics.lambda0 * time_scale,
        )

    def test_all_cases_share_center_and_driving_noise(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml").with_overrides(
            duration=8.0e-7
        )
        self.assertEqual(config.initial.position, (0.0, 0.0))
        insensitive = replace(config, responses=Responses.zeros())
        trajectories = simulate_all_cases(insensitive)
        reference = trajectories[0]
        for trajectory in trajectories[1:]:
            np.testing.assert_array_equal(trajectory.positions, reference.positions)
            np.testing.assert_array_equal(trajectory.headings, reference.headings)

    def test_ensemble_msd_starts_at_zero(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml").with_overrides(
            duration=8.0e-7
        )
        result = compute_ensemble_msd(config, ensemble_size=3, workers=1)
        self.assertEqual(result.mean.shape, (8, len(result.times)))
        np.testing.assert_array_equal(result.mean[:, 0], np.zeros(8))
        self.assertTrue(np.all(result.mean >= 0.0))

    def test_numba_matches_readable_reference(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml").with_overrides(
            duration=8.0e-6
        )
        config, _ = resolve_numerics(config, create_field(config.field), (8,))
        python_config = replace(
            config,
            performance=replace(config.performance, engine="python"),
        )
        field = GaussianField(
            baseline=1.0,
            amplitude=1.0,
            center=(1.5, 1.0),
            sigma=(1.6, 2.2),
            angle=0.35,
        )
        steps = int(config.simulation.duration / config.simulation.dt)
        innovations = Innovations.generate(steps, config.simulation.seed)
        native = simulate(config, 8, field=field, innovations=innovations)
        readable = simulate(
            python_config,
            8,
            field=field,
            innovations=innovations,
        )
        np.testing.assert_array_equal(native.positions, readable.positions)
        np.testing.assert_array_equal(native.headings, readable.headings)
        np.testing.assert_array_equal(native.tumble_counts, readable.tumble_counts)

    def test_numba_matches_reference_at_every_wall_law(self) -> None:
        base = load_config(PROJECT_ROOT / "configs" / "default.toml").with_overrides(
            duration=8.0e-6
        )
        base = replace(
            base,
            kinetics=replace(base.kinetics, v0=2000.0),
        )
        field = GaussianField(
            baseline=1.0,
            amplitude=1.0,
            center=(1.5, 1.0),
            sigma=(1.6, 2.2),
            angle=0.35,
        )
        for boundary_mode in ("sliding", "specular", "diffuse"):
            with self.subTest(boundary_mode=boundary_mode):
                config = replace(
                    base,
                    initial=replace(
                        base.initial,
                        position=(0.99, 0.0),
                    ),
                    domain=replace(
                        base.domain,
                        boundary=boundary_mode,
                    ),
                )
                config, _ = resolve_numerics(config, field, (0,))
                steps = int(config.simulation.duration / config.simulation.dt)
                innovations = Innovations.generate(steps, config.simulation.seed)
                python_config = replace(
                    config,
                    performance=replace(config.performance, engine="python"),
                )
                native = simulate(
                    config,
                    0,
                    field=field,
                    innovations=innovations,
                )
                readable = simulate(
                    python_config,
                    0,
                    field=field,
                    innovations=innovations,
                )
                self.assertGreater(int(native.wall_hit_counts.sum()), 0)
                np.testing.assert_allclose(
                    native.positions,
                    readable.positions,
                    rtol=0.0,
                    atol=1.0e-14,
                )
                np.testing.assert_allclose(
                    native.headings,
                    readable.headings,
                    rtol=0.0,
                    atol=1.0e-14,
                )
                np.testing.assert_array_equal(
                    native.wall_hit_counts,
                    readable.wall_hit_counts,
                )


class ResolutionTests(unittest.TestCase):
    def test_nonfinite_configuration_is_rejected(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        invalid = replace(
            config,
            responses=replace(config.responses, v_S=float("nan")),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid.validate()

    def test_default_profile_meets_working_theory_targets(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        plan = build_resolution_plan(
            config,
            create_field(config.field),
            tuple(range(9)),
        )
        backgrounds = {
            "v": config.kinetics.v0,
            "D_T": config.kinetics.D_T0,
            "D_R": config.kinetics.D_R0,
            "lambda": config.kinetics.lambda0,
        }
        eta = max(
            max(abs(bound - backgrounds[name]) for bound in bounds) / backgrounds[name]
            for ranges in plan.ranges_by_case.values()
            for name, bounds in ranges.items()
        )
        gamma0 = config.kinetics.D_R0 + config.kinetics.lambda0 * (
            1.0 - config.kinetics.a1
        )
        persistence_ratio = (
            config.kinetics.v0 / gamma0 / plan.characteristic_length_star
        )
        self.assertLessEqual(eta, 0.15)
        self.assertLessEqual(persistence_ratio, 0.2)
        self.assertGreaterEqual(
            gamma0 * config.simulation.duration,
            10.0,
        )
        self.assertEqual(
            recommended_ensemble_size(config.density),
            16384,
        )

    def test_radius_and_diffusion_define_requested_scales(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        field = create_field(config.field)
        plan = build_resolution_plan(config, field, tuple(range(1, 9)))
        self.assertEqual(plan.scales.length, config.domain.radius)
        self.assertEqual(
            plan.scales.time,
            config.domain.radius**2 / config.kinetics.D_T0,
        )
        self.assertLessEqual(plan.dt, plan.dt_limit)
        self.assertEqual(
            set(plan.constraints),
            {
                "advection",
                "translational diffusion",
                "rotational diffusion",
                "tumbling",
            },
        )
        duration = config.simulation.duration / plan.scales.time
        self.assertAlmostEqual(
            duration / plan.dt,
            round(duration / plan.dt),
        )

    def test_field_grid_resolves_smallest_gaussian_width(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        plan = build_resolution_plan(config, create_field(config.field), (8,))
        spacing = 2.0 * config.domain.radius / (plan.field_grid - 1)
        self.assertLessEqual(
            spacing,
            plan.characteristic_length / config.numerics.scan_points_per_length,
        )

    def test_unsafe_manual_dt_is_rejected(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml").with_overrides(
            dt=0.1
        )
        with self.assertRaisesRegex(ValueError, "too large"):
            build_resolution_plan(config, create_field(config.field), (8,))

    def test_high_velocity_reduces_automatic_dt(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        fast = replace(
            config,
            kinetics=replace(config.kinetics, v0=20000.0),
        )
        plan = build_resolution_plan(fast, create_field(fast.field), (0,))
        self.assertEqual(plan.limiting_constraint, "advection")
        velocity = fast.kinetics.v0 * fast.domain.radius / fast.kinetics.D_T0
        self.assertLessEqual(
            velocity * plan.dt,
            plan.target_spatial_step,
        )

    def test_underresolved_manual_field_grid_is_rejected(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        coarse = replace(
            config,
            video=replace(config.video, field_grid=33),
        )
        with self.assertRaisesRegex(ValueError, "under-resolves"):
            build_resolution_plan(coarse, create_field(coarse.field), (8,))


class ProbabilityDensityTests(unittest.TestCase):
    @staticmethod
    def _small_density_config():
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        return replace(
            config.with_overrides(duration=8.0e-7, progress=False),
            density=replace(
                config.density,
                initial_position_distribution="fixed",
                bandwidth_fraction=0.25,
                difference_bandwidth_fraction=0.25,
                min_samples_per_kernel=5,
                batch_size=8,
                time_slices=5,
                grid_size=33,
                video_fps=5,
                frame_size=600,
            ),
        )

    def test_automatic_M_meets_kernel_sample_target(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        count = recommended_ensemble_size(config.density)
        self.assertEqual(count, 16384)
        self.assertGreaterEqual(
            2.0 * count * config.density.bandwidth_fraction**2,
            config.density.min_samples_per_kernel,
        )
        with self.assertRaisesRegex(ValueError, "below"):
            resolve_density_ensemble_size(config.density, requested=count // 2)

    def test_cases_zero_through_eight_are_normalized(self) -> None:
        config = self._small_density_config()
        result = compute_position_pdf(config)
        self.assertEqual(result.case_numbers, tuple(range(9)))
        self.assertEqual(result.density.shape[:2], (9, len(result.times)))
        time_scale = config.domain.radius**2 / config.kinetics.D_T0
        self.assertEqual(
            result.times[-1],
            config.simulation.duration / time_scale,
        )
        self.assertGreaterEqual(float(result.coordinates[0]), -1.0)
        self.assertLessEqual(float(result.coordinates[-1]), 1.0)
        cell_width = 2.0 / config.density.grid_size
        integrals = np.sum(result.density, axis=(-2, -1)) * cell_width**2
        np.testing.assert_allclose(integrals, 1.0, atol=2.0e-7)
        self.assertGreater(float(np.max(result.density)), 1.0)
        self.assertIsNotNone(result.microscopic_observables)
        observables = result.microscopic_observables
        assert observables is not None
        self.assertEqual(
            observables.values.shape[1:],
            (9, len(result.times)),
        )
        self.assertEqual(
            observables.paired_standard_errors.shape,
            observables.values.shape,
        )
        self.assertTrue(np.all(observables.paired_standard_errors >= 0.0))

    def test_density_python_and_numba_engines_share_reproducible_draws(self) -> None:
        base = self._small_density_config()
        numba_config = replace(
            base,
            performance=replace(base.performance, engine="numba", threads=2),
        )
        serial_numba_config = replace(
            base,
            performance=replace(base.performance, engine="numba", threads=1),
        )
        python_config = replace(
            base,
            performance=replace(base.performance, engine="python"),
        )
        native = compute_position_pdf(numba_config)
        serial_native = compute_position_pdf(serial_numba_config)
        readable = compute_position_pdf(python_config)
        np.testing.assert_array_equal(native.density, readable.density)
        np.testing.assert_array_equal(native.density, serial_native.density)
        np.testing.assert_array_equal(
            native.difference_density,
            readable.difference_density,
        )
        assert native.microscopic_observables is not None
        assert readable.microscopic_observables is not None
        np.testing.assert_allclose(
            native.microscopic_observables.values,
            readable.microscopic_observables.values,
            rtol=0.0,
            atol=5.0e-14,
        )
        np.testing.assert_allclose(
            native.microscopic_observables.paired_standard_errors,
            readable.microscopic_observables.paired_standard_errors,
            rtol=0.0,
            atol=5.0e-14,
        )

    def test_native_density_rejects_duplicate_truncated_seeds(self) -> None:
        config = self._small_density_config()
        field = create_field(config.field)
        config, _ = resolve_numerics(config, field, tuple(range(9)))
        with self.assertRaisesRegex(ValueError, "not unique"):
            state_batch_numba(
                config,
                field,
                [0, 2**32],
                case_numbers=(0,),
            )

    def test_case_zero_uses_only_background_coefficients(self) -> None:
        config = replace(
            self._small_density_config(),
            responses=Responses.zeros(),
        )
        result = compute_position_pdf(config)
        for case_number in range(1, 9):
            np.testing.assert_array_equal(
                result.density[case_number],
                result.density[0],
            )
        observables = result.microscopic_observables
        assert observables is not None
        for case_number in range(1, 9):
            np.testing.assert_array_equal(
                observables.values[:, case_number],
                observables.values[:, 0],
            )
            np.testing.assert_array_equal(
                observables.paired_standard_errors[:, case_number],
                0.0,
            )

    def test_difference_kde_is_directly_computed_and_smoother(self) -> None:
        config = self._small_density_config()
        config = replace(
            config,
            density=replace(
                config.density,
                bandwidth_fraction=0.125,
                difference_bandwidth_fraction=0.25,
            ),
        )
        result = compute_position_pdf(config)
        self.assertIsNotNone(result.difference_density)
        self.assertEqual(result.difference_bandwidth_fraction, 0.25)
        difference_density = result.difference_density
        assert difference_density is not None
        cell_width = 2.0 / config.density.grid_size
        integrals = np.sum(difference_density, axis=(-2, -1)) * cell_width**2
        np.testing.assert_allclose(integrals, 1.0, atol=2.0e-7)
        self.assertLess(
            float(np.max(difference_density[:, 0])),
            float(np.max(result.density[:, 0])),
        )

        with TemporaryDirectory() as directory:
            output = save_position_pdf(
                result,
                Path(directory) / "density.npz",
            )
            with np.load(output) as saved:
                self.assertEqual(int(saved["schema_version"]), 2)
                self.assertEqual(str(saved["software_version"]), "0.2.0")
                self.assertTrue(str(saved["config_text"]).startswith("# Every value"))
                self.assertEqual(len(str(saved["config_sha256"])), 64)
                self.assertIn('"numpy":', str(saved["runtime_versions"]))
                self.assertIn("difference_density", saved.files)
                self.assertIn("config_source", saved.files)
                self.assertIn("microscopic_observable_values", saved.files)
                self.assertIn(
                    "microscopic_observable_paired_standard_errors",
                    saved.files,
                )
                self.assertEqual(
                    float(saved["difference_bandwidth_fraction"]),
                    0.25,
                )
            loaded = load_position_pdf(output)
            np.testing.assert_array_equal(loaded.density, result.density)
            self.assertEqual(loaded.schema_version, 2)
            self.assertEqual(loaded.software_version, "0.2.0")
            self.assertTrue(loaded.config_text.startswith("# Every value"))
            self.assertEqual(len(loaded.config_sha256), 64)
            self.assertIn('"numpy":', loaded.runtime_versions)
            self.assertEqual(
                loaded.microscopic_observables.names,
                result.microscopic_observables.names,
            )

    def test_case_specific_difference_scale_is_linear_and_fixed(self) -> None:
        values = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2])
        mapped = _normalize_linear_difference(values, 0.2)
        np.testing.assert_allclose(
            mapped,
            [-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        differences = np.zeros((2, 2, 3, 3))
        differences[1, 0] = np.arange(9).reshape(3, 3) / 10.0
        differences[1, 1] = -differences[1, 0]
        limits, maxima = _case_specific_linear_limits(
            differences,
            np.asarray([-0.5, 0.0, 0.5]),
            100.0,
        )
        np.testing.assert_allclose(limits, [0.0, 0.8])
        np.testing.assert_allclose(maxima, [0.0, 0.8])

        result_a = PositionPDFResult(
            times=np.asarray([0.0, 1.0]),
            coordinates=np.asarray([-0.5, 0.0, 0.5]),
            density=differences.astype(np.float32),
            difference_density=differences.astype(np.float32),
            case_numbers=(0, 1),
            ensemble_size=1,
            bandwidth_fraction=0.1,
        )
        result_b = replace(
            result_a,
            difference_density=(2.0 * differences).astype(np.float32),
            method="macro",
        )
        shared = synchronized_case_linear_limits([result_a, result_b], 100.0)
        np.testing.assert_allclose(shared, [0.0, 1.6])
        global_shared = synchronized_global_linear_limits(
            [result_a, result_b],
            100.0,
        )
        np.testing.assert_allclose(global_shared, [1.6, 1.6])

    def test_dimensionless_chemical_field_plot_is_written(self) -> None:
        config = self._small_density_config()
        field = create_field(config.field)
        with TemporaryDirectory() as directory:
            output = render_chemical_field(
                field,
                config,
                Path(directory) / "chemical_field.png",
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)

    def test_microscopic_mechanism_report_is_renderable(self) -> None:
        config = self._small_density_config()
        result = compute_position_pdf(config)
        with TemporaryDirectory() as directory:
            outputs = render_microscopic_mechanism_report(
                result,
                config,
                Path(directory) / "mechanisms.png",
                Path(directory) / "summary.csv",
                Path(directory) / "timeseries.csv",
            )
            for output in outputs:
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 100)


class MacroscopicDensityTests(unittest.TestCase):
    @staticmethod
    def _small_macro_config():
        config = load_config(PROJECT_ROOT / "configs" / "default.toml")
        return replace(
            config.with_overrides(duration=0.002, progress=False),
            field={
                "kind": "linear",
                "baseline": 1.0,
                "gradient": [0.2, -0.1],
                "center": [0.0, 0.0],
            },
            density=replace(
                config.density,
                grid_size=33,
                time_slices=5,
            ),
        )

    def test_macro_pdf_is_positive_conservative_and_dimensionless(self) -> None:
        config = self._small_macro_config()
        result = compute_macroscopic_pdf(config)
        spacing = 2.0 / len(result.coordinates)
        masses = np.sum(result.density, axis=(-2, -1)) * spacing**2
        np.testing.assert_allclose(masses, 1.0, rtol=0.0, atol=2.0e-6)
        self.assertGreaterEqual(float(np.min(result.density)), 0.0)
        self.assertEqual(result.method, "macro")
        self.assertEqual(result.ensemble_size, 0)
        self.assertEqual(result.times[-1], config.simulation.duration)
        self.assertGreaterEqual(len(result.coordinates), config.density.grid_size)

    def test_zero_mobility_cases_equal_the_baseline_exactly(self) -> None:
        result = compute_macroscopic_pdf(self._small_macro_config())
        for case_number in (2, 4, 5):
            np.testing.assert_array_equal(
                result.density[case_number],
                result.density[0],
            )

    def test_macro_plan_uses_table_2_coefficients_and_safe_dt(self) -> None:
        config = self._small_macro_config()
        plan, *_ = build_macroscopic_plan(config)
        for case_number, mobility in enumerate(plan.mobilities):
            _, effective_diffusivity, expected_mobility = macroscopic_coefficients(
                config.kinetics,
                config.responses.only_case(case_number),
            )
            self.assertEqual(plan.effective_diffusivity, effective_diffusivity)
            self.assertEqual(mobility, expected_mobility)
        self.assertLessEqual(plan.dt, plan.safe_dt)
        self.assertAlmostEqual(
            plan.steps * plan.dt,
            config.simulation.duration,
        )

    def test_exponential_equilibrium_has_zero_discrete_flux(self) -> None:
        config = self._small_macro_config()
        field = create_field(config.field)
        (
            plan,
            _,
            grid_points,
            mask,
            neighbors,
            incoming,
            outgoing,
        ) = build_macroscopic_plan(config, field)
        concentration = np.asarray(field.value(grid_points), dtype=float)[mask]
        for case_index, mobility in enumerate(plan.mobilities):
            equilibrium = np.exp(mobility * concentration / plan.effective_diffusivity)
            derivative = -outgoing[case_index] * equilibrium
            for direction in range(4):
                neighbor = neighbors[:, direction]
                connected = neighbor >= 0
                derivative[connected] += (
                    incoming[case_index, connected, direction]
                    * equilibrium[neighbor[connected]]
                )
            np.testing.assert_allclose(
                derivative,
                0.0,
                rtol=0.0,
                atol=2.0e-11,
            )

    def test_macro_and_micro_use_the_same_observed_point_initial_state(self) -> None:
        micro_config = ProbabilityDensityTests._small_density_config()
        micro_config = replace(
            micro_config,
            density=replace(micro_config.density, grid_size=41),
        )
        micro = compute_position_pdf(micro_config)
        macro = compute_macroscopic_pdf(micro_config)
        np.testing.assert_array_equal(micro.coordinates, macro.coordinates)
        np.testing.assert_array_equal(micro.density[:, 0], macro.density[:, 0])

    def test_macro_rejects_missing_angular_relaxation(self) -> None:
        config = self._small_macro_config()
        config = replace(
            config,
            kinetics=replace(
                config.kinetics,
                v0=0.0,
                D_R0=0.0,
                lambda0=0.0,
            ),
        )
        with self.assertRaisesRegex(ValueError, "requires gamma0"):
            build_macroscopic_plan(config)

    def test_density_cli_selects_micro_or_macro(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["compare"]).strategy, "both")
        self.assertEqual(parser.parse_args(["compare", "--micro"]).strategy, "micro")
        self.assertEqual(parser.parse_args(["compare", "--macro"]).strategy, "macro")
        with self.assertRaises(SystemExit):
            parser.parse_args(["compare", "--micro", "--macro"])


if __name__ == "__main__":
    unittest.main()
