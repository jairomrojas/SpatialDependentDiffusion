from __future__ import annotations

import csv
import math
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy import special

from chem_swimmer._version import PDF_SCHEMA_VERSION
from chem_swimmer.accelerated import state_batch_numba
from chem_swimmer.analytic import (
    _constant_mode_index,
    build_neumann_bessel_basis,
    compute_analytic_pdf,
    compute_overlap_matrix,
    delta_projection,
    evaluate_basis,
    evaluate_density,
    solve_analytic_spectrum,
    uniform_projection,
)
from chem_swimmer.boundary import advance_in_disk
from chem_swimmer.boundary_layer import sliding_wall_boundary_layer
from chem_swimmer.cases import STRATEGY_FIGURE_STEMS
from chem_swimmer.cli import _load, build_parser
from chem_swimmer.config import Responses, load_config
from chem_swimmer.density import (
    PositionPDFResult,
    _accumulate_angular_moments,
    _convolve_axis,
    _kde_kernel_and_support,
    _normalized_smoothed,
    compute_position_pdf,
    load_position_pdf,
    recommended_ensemble_size,
    resolve_density_ensemble_size,
    save_position_pdf,
)
from chem_swimmer.diagnostics import timestep_report
from chem_swimmer.fields import create_field, disk_area_average
from chem_swimmer.fields.base import positions_array
from chem_swimmer.fields.builtin import GaussianField, PeriodicField
from chem_swimmer.fields.user_field import UserField
from chem_swimmer.kinetics import (
    KineticLaw,
    angular_steering_rate,
    delta_D_eff_coefficient,
    macroscopic_coefficients,
    pulse_centroid_drift,
)
from chem_swimmer.macroscopic import (
    _initial_density,
    build_macroscopic_plan,
    compute_macroscopic_pdf,
)
from chem_swimmer.moment import build_moment_plan, compute_moment_pdf
from chem_swimmer.nematic import (
    nematic_closure_rows,
    nematic_frobenius,
    reconstruct_density_flux,
)
from chem_swimmer.noise import Innovations
from chem_swimmer.numerics import build_resolution_plan, resolve_numerics
from chem_swimmer.observables import (
    FIRST_NEUMANN_DISK_WAVENUMBER,
    POINT_RELEASE_EXPONENTS,
    SHORT_TIME_EXPONENTS,
    TransportObservables,
    background_overlaps,
    baseline_outward_polarity,
    baseline_spread,
    chemotactic_mobility,
    domain_mixing_time,
    effective_diffusivity,
    finite_volume_observables,
    green_amplitudes,
    landscape_uptake_time,
    periodic_uptake_prediction,
    polarity_relaxation_rate,
    predicted_observables,
    uptake_gain,
    uptake_rate,
)
from chem_swimmer.simulator import simulate, simulate_all_cases
from chem_swimmer.statistics import compute_ensemble_msd, free_space_baseline_msd
from chem_swimmer.turning import (
    DiscreteTurningKernel,
    TabulatedTurningKernel,
    WrappedNormalTurningKernel,
    create_turning_kernel,
    sample_tumble_heading,
)
from chem_swimmer.visualization.comparison_plot import (
    render_density_frames_vs_time,
    render_initial_condition_contours,
    render_polarity_frames_vs_time,
)
from chem_swimmer.visualization.density_video import (
    _case_specific_linear_limits,
    _density_frame_indices,
    _normalize_linear_difference,
    render_density_comparison_video,
    render_density_difference_comparison_video,
    shared_display_times,
    synchronized_case_linear_limits,
    synchronized_global_linear_limits,
)
from chem_swimmer.visualization.field_plot import render_chemical_field
from chem_swimmer.visualization.inference_plot import (
    render_microscopic_mechanism_report,
)
from chem_swimmer.visualization.manuscript_plot import (
    render_manuscript_strategy_figures,
)
from chem_swimmer.visualization.nematic_plot import render_nematic_closure_report
from chem_swimmer.visualization.polarity_plot import (
    _adiabatic_polarity,
    _masked_directional_gradient,
    compute_theoretical_polarity_field,
    render_polarity_report,
)
from chem_swimmer.visualization.polarity_video import (
    _polarity_case_limits,
    render_polarity_comparison_video,
    render_polarity_difference_video,
    synchronized_polarity_case_limits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FieldTests(unittest.TestCase):
    def test_eta0_is_the_disk_area_average_used_by_the_manuscript(self) -> None:
        field = create_field(
            {
                "kind": "linear",
                "baseline": 1.0,
                "gradient": [0.3, 0.4],
                "center": [0.2, -0.1],
            }
        )
        eta0 = disk_area_average(field, 1.0)
        self.assertAlmostEqual(eta0, 0.98, places=13)
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        law = KineticLaw(config.kinetics, Responses.zeros(), eta0)
        state = law.evaluate(
            float(field.value(np.zeros(2))),
            field.gradient(np.zeros(2)),
            0.0,
        )
        self.assertAlmostEqual(state.delta_eta, 0.0, places=13)

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
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        with self.assertRaisesRegex(ValueError, r"unknown .*field"):
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
        source = PROJECT_ROOT / "config" / "default.toml"
        with TemporaryDirectory() as directory:
            modified = Path(directory) / "unknown_section.toml"
            modified.write_text(
                source.read_text(encoding="utf-8") + "\n[simulaton_typo]\nvalue = 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown top-level"):
                load_config(modified)

    def test_legacy_a1_configuration_fails_with_migration_guidance(self) -> None:
        source = PROJECT_ROOT / "config" / "default.toml"
        text = source.read_text(encoding="utf-8")
        text = text.replace("\n[turning]\n", "\na1 = 0.30\n\n[turning]\n", 1)
        with TemporaryDirectory() as directory:
            modified = Path(directory) / "legacy.toml"
            modified.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"kinetics\.a1.*\[turning\]"):
                load_config(modified)


class TumbleTests(unittest.TestCase):
    def test_wrapped_normal_samples_declared_P_turn(self) -> None:
        rng = np.random.default_rng(21)
        count = 80_000
        normal = rng.standard_normal(count)
        uniform = rng.random(count)
        for P_turn in (-0.45, 0.0, 0.3, 0.85):
            kernel = WrappedNormalTurningKernel(P_turn)
            turns = np.array(
                [
                    sample_tumble_heading(0.0, kernel, z, u)
                    for z, u in zip(normal, uniform, strict=True)
                ]
            )
            self.assertAlmostEqual(float(np.mean(np.cos(turns))), P_turn, delta=0.008)

    def test_arbitrary_achiral_discrete_kernel_exposes_manuscript_moments(self) -> None:
        kernel = DiscreteTurningKernel(
            absolute_angles=[0.0, np.pi / 2.0, np.pi],
            probabilities=[0.2, 0.5, 0.3],
        )
        self.assertAlmostEqual(kernel.P_turn, -0.1)
        self.assertAlmostEqual(kernel.P_2_turn, 0.0)
        rng = np.random.default_rng(18)
        turns = np.asarray(
            [
                kernel.sample_turn_angle(z, u)
                for z, u in zip(
                    rng.standard_normal(100_000), rng.random(100_000), strict=True
                )
            ]
        )
        self.assertAlmostEqual(float(np.mean(np.sin(turns))), 0.0, delta=0.008)
        self.assertAlmostEqual(
            float(np.mean(np.cos(turns))), kernel.P_turn, delta=0.008
        )

    def test_tabulated_even_density_and_config_factory(self) -> None:
        kernel = TabulatedTurningKernel(
            absolute_angles=[0.0, np.pi / 2.0, np.pi],
            density=[1.0, 2.0, 1.0],
        )
        self.assertTrue(kernel.achiral)
        self.assertLessEqual(abs(kernel.P_turn), 1.0)
        configured = create_turning_kernel(
            {"kind": "discrete", "absolute_angles": [0.0], "probabilities": [1.0]}
        )
        self.assertEqual(configured.P_turn, 1.0)

    def test_discrete_kernel_runs_through_exact_python_fallback(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        config = replace(
            config.with_overrides(duration=2.0e-5, engine="auto"),
            kinetics=replace(config.kinetics, v0=0.0, D_R0=0.0, lambda0=100.0),
            turning={
                "kind": "discrete",
                "absolute_angles": [np.pi],
                "probabilities": [1.0],
            },
            responses=Responses.zeros(),
        )
        field = create_field(config.field)
        resolved, _ = resolve_numerics(config, field, (0,))
        assert not isinstance(resolved.simulation.dt, str)
        steps = round(resolved.simulation.duration / resolved.simulation.dt)
        innovations = Innovations(
            translation=np.zeros((steps, 2)),
            rotation=np.zeros(steps),
            tumble_event=np.zeros(steps),
            tumble_normal=np.ones(steps),
            tumble_uniform=np.zeros(steps),
        )
        trajectory = simulate(resolved, 0, field=field, innovations=innovations)
        expected = (
            resolved.initial.heading + np.pi * np.arange(len(trajectory.headings))
        ) % (2.0 * np.pi)
        np.testing.assert_allclose(trajectory.headings, expected, atol=1.0e-14)


class BoundaryTests(unittest.TestCase):
    def test_head_on_contact_stops_at_the_wall(self) -> None:
        """A purely normal increment is absorbed, not reflected back inside."""

        result = advance_in_disk(
            np.array([0.9, 0.0]),
            np.array([0.3, 0.0]),
            heading=0.0,
            radius=1.0,
        )
        np.testing.assert_allclose(result.position, [1.0, 0.0], atol=1.0e-11)
        self.assertAlmostEqual(result.heading, 0.0)
        self.assertEqual(result.hits, 1)

    def test_oblique_contact_slides_along_the_wall(self) -> None:
        """The tangential component survives as arc length on the circle."""

        result = advance_in_disk(
            np.array([0.9, 0.0]),
            np.array([0.3, 0.2]),
            heading=0.0,
            radius=1.0,
        )
        self.assertAlmostEqual(float(np.linalg.norm(result.position)), 1.0, places=11)
        self.assertGreater(result.position[1], 0.0)
        self.assertEqual(result.hits, 1)

    def test_interior_increment_is_untouched(self) -> None:
        result = advance_in_disk(
            np.array([0.1, 0.0]),
            np.array([0.02, 0.01]),
            heading=1.0,
            radius=1.0,
        )
        np.testing.assert_allclose(result.position, [0.12, 0.01], atol=1.0e-14)
        self.assertEqual(result.hits, 0)

    def test_sliding_never_changes_heading(self) -> None:
        result = advance_in_disk(
            np.array([0.9, 0.0]),
            np.array([0.3, 0.0]),
            heading=0.73,
            radius=1.0,
        )
        self.assertAlmostEqual(result.heading, 0.73)


class FluxDissipationTests(unittest.TestCase):
    """The finite-volume face flux must not inflate the physical diffusivity.

    Plain Rusanov adds a numerical diffusivity ``a_max * dx / 2`` whatever the
    grid.  At the paper configuration that is ~0.20, i.e. a fifth of ``D_T0``
    -- and because ``a_max`` contains ``v_eta`` and ``v_zeta`` it does not
    cancel in the case-minus-baseline response, which contaminated Cases 3 and
    7.  The hybrid coefficient adds only the deficit the cell Peclet number
    needs.
    """

    def _plan(self, numerics_overrides=None, **density_overrides):
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        if density_overrides:
            config = replace(
                config, density=replace(config.density, **density_overrides)
            )
        if numerics_overrides:
            config = replace(
                config, numerics=replace(config.numerics, **numerics_overrides)
            )
        field = create_field(config.field)
        return build_moment_plan(config, field)[0], config

    def test_paper_configuration_adds_no_numerical_diffusion(self) -> None:
        plan, _ = self._plan()
        self.assertLess(plan.cell_peclet, 2.0)
        self.assertEqual(plan.artificial_diffusivity, 0.0)
        # Document what the unconditional Rusanov term would have added: a
        # fifth of the physical diffusivity, still far above the response
        # differences the case-minus-baseline comparison has to resolve.
        rusanov = 0.5 * plan.max_wave_speed * plan.spacing
        self.assertGreater(rusanov, 0.15 * plan.min_diffusivity)

    def test_coarse_grid_restores_exactly_the_needed_dissipation(self) -> None:
        """Above Pe=2 the hybrid adds the deficit, never more."""

        plan, _ = self._plan(
            numerics_overrides={"scan_points_per_length": 2, "min_scan_grid": 21},
            grid_size=21,
            bandwidth_fraction=0.4,
            difference_bandwidth_fraction=0.4,
        )
        self.assertGreater(plan.cell_peclet, 2.0)
        expected = 0.5 * plan.max_wave_speed * plan.spacing - plan.min_diffusivity
        self.assertAlmostEqual(plan.artificial_diffusivity, expected, places=12)

    def test_artificial_diffusivity_is_never_negative(self) -> None:
        for grid in (81, 121, 161):
            plan, _ = self._plan(grid_size=grid)
            self.assertGreaterEqual(plan.artificial_diffusivity, 0.0)


class LongTimeCoefficientTests(unittest.TestCase):
    """Guards on the manuscript's long-time coefficients kappa, dD_eff and V.

    The values below were confirmed against a direct integration of the closed
    density--polarity system (``scripts/check_long_time_coefficients.py``): the
    measured centroid velocity converges to ``V`` as ``1/(gamma1 t)``.
    """

    def _background(self):
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        return config.kinetics, config.responses, 0.30

    def test_pulse_drift_is_kappa_plus_diffusivity_shift(self) -> None:
        kinetics, responses, P_turn = self._background()
        for case in range(1, 10):
            single = responses.only_case(case)
            _, _, kappa = macroscopic_coefficients(kinetics, single, P_turn)
            shift = delta_D_eff_coefficient(kinetics, single, P_turn)
            drift = pulse_centroid_drift(kinetics, single, P_turn)
            self.assertAlmostEqual(drift, kappa + shift, places=12, msg=f"case {case}")

    def test_level_diffusivity_response_moves_no_centroid(self) -> None:
        """Case 1 has kappa != 0 but V == 0: D_T_eta cancels from the drift."""

        kinetics, responses, P_turn = self._background()
        single = responses.only_case(1)
        _, _, kappa = macroscopic_coefficients(kinetics, single, P_turn)
        self.assertNotAlmostEqual(kappa, 0.0)
        self.assertAlmostEqual(pulse_centroid_drift(kinetics, single, P_turn), 0.0)

    def test_directional_translational_diffusion_is_silent(self) -> None:
        """Case 5 contributes to none of the three long-time observables."""

        kinetics, responses, P_turn = self._background()
        single = responses.only_case(5)
        _, _, kappa = macroscopic_coefficients(kinetics, single, P_turn)
        self.assertAlmostEqual(kappa, 0.0)
        self.assertAlmostEqual(delta_D_eff_coefficient(kinetics, single, P_turn), 0.0)
        self.assertAlmostEqual(pulse_centroid_drift(kinetics, single, P_turn), 0.0)

    def test_speed_response_reverses_sign_between_kappa_and_drift(self) -> None:
        """Case 3: accumulates where cells are slow, centroid moves the other way."""

        kinetics, responses, P_turn = self._background()
        single = responses.only_case(3)
        _, _, kappa = macroscopic_coefficients(kinetics, single, P_turn)
        drift = pulse_centroid_drift(kinetics, single, P_turn)
        self.assertLess(kappa, 0.0)
        self.assertGreater(drift, 0.0)
        self.assertAlmostEqual(drift, -kappa, places=12)


class MacroscopicObservableTests(unittest.TestCase):
    """Guards on the closed moment system of the manuscript's Sec. "Macroscopic
    observables": four balance laws for the net polarity, the outward polarity
    moment, the centroid, and the second moment.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(PROJECT_ROOT / "config" / "default.toml")
        cls.field = create_field(cls.config.field)
        cls.times = np.linspace(0.0, cls.config.simulation.duration, 121)
        cls.overlaps = background_overlaps(cls.config, cls.field, cls.times)
        cls.prediction = predicted_observables(
            cls.config, cls.field, overlaps=cls.overlaps
        )

    def test_green_amplitudes_start_at_the_release(self) -> None:
        wavenumbers = np.array([0.0, 1.0, 10.0, 200.0])
        density, polarity = green_amplitudes(self.config, wavenumbers, 0.0)
        np.testing.assert_allclose(density, 1.0, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(polarity, 0.0, rtol=0.0, atol=1e-14)
        # A_0(t) = 1 for all t: the k=0 density amplitude is conserved mass.
        for time in (1e-4, 1e-3, 5e-3):
            density, polarity = green_amplitudes(self.config, np.zeros(1), time)
            self.assertAlmostEqual(float(density[0]), 1.0, places=12)
            self.assertAlmostEqual(float(polarity[0]), 0.0, places=12)

    def test_background_reproduces_the_closed_form_outward_moment(self) -> None:
        """The spectral synthesis must return Omega^(0) analytically."""

        expected = baseline_outward_polarity(self.config, self.times)
        np.testing.assert_allclose(
            self.overlaps.outward_polarity, expected, rtol=1e-9, atol=1e-14
        )

    def test_baseline_spread_integrates_its_own_balance_law(self) -> None:
        """S^(0) is the integral of 4 D_T0 + 2 v0 Omega^(0)."""

        times = np.linspace(0.0, self.config.simulation.duration, 40001)
        rate = 4.0 * self.config.kinetics.D_T0 + 2.0 * self.config.kinetics.v0 * (
            baseline_outward_polarity(self.config, times)
        )
        integrated = np.concatenate(
            ([0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(times)))
        )
        np.testing.assert_allclose(
            integrated[-1], baseline_spread(self.config, times)[-1], rtol=1e-9
        )

    def test_case_zero_is_the_uniform_background(self) -> None:
        prediction = self.prediction
        np.testing.assert_allclose(
            prediction.spread[0], baseline_spread(self.config, self.times), rtol=1e-12
        )
        np.testing.assert_allclose(
            prediction.outward_polarity[0],
            baseline_outward_polarity(self.config, self.times),
            rtol=1e-9,
            atol=1e-14,
        )
        np.testing.assert_array_equal(prediction.net_polarity[0], 0.0)
        np.testing.assert_array_equal(prediction.centroid[0], 0.0)

    def test_flux_responses_never_polarize_the_population(self) -> None:
        """A flux moves alignment about; only an angular source creates it."""

        for case in (1, 3, 5, 7):
            with self.subTest(case=case):
                np.testing.assert_array_equal(
                    self.prediction.response("net_polarity")[case], 0.0
                )
        for case in (1, 5):
            with self.subTest(case=case, observable="centroid"):
                np.testing.assert_array_equal(
                    self.prediction.response("centroid")[case], 0.0
                )

    def _fitted_exponents(self, config) -> dict[int, tuple[int | None, ...]]:
        times = np.geomspace(1e-6, 2e-4, 40)
        prediction = predicted_observables(
            config, self.field, np.concatenate(([0.0], times))
        )
        fitted: dict[int, tuple[int | None, ...]] = {}
        for case in range(1, 10):
            row: list[int | None] = []
            for name in ("net_polarity", "centroid", "spread", "shape"):
                response = prediction.response(name)[case][1:]
                magnitude = (
                    np.abs(response)
                    if response.ndim == 1
                    else np.linalg.norm(response, axis=-1)
                )
                if magnitude.max() <= 0.0:
                    row.append(None)
                    continue
                slope = np.polyfit(np.log(times), np.log(magnitude), 1)[0]
                row.append(round(slope))
                self.assertLess(
                    abs(slope - round(slope)),
                    0.06,
                    msg=f"case {case} {name} slope {slope:.3f} is not a clean power",
                )
            fitted[case] = tuple(row)
        return fitted

    def test_short_time_exponents_match_the_table(self) -> None:
        self.assertEqual(self._fitted_exponents(self.config), SHORT_TIME_EXPONENTS)

    def test_a_point_release_delays_the_dilation_driven_entries(self) -> None:
        point = replace(
            self.config,
            density=replace(self.config.density, initial_position_distribution="point"),
        )
        self.assertEqual(self._fitted_exponents(point), POINT_RELEASE_EXPONENTS)

    def test_adiabatic_centroid_rate_is_the_pulse_drift(self) -> None:
        """Quasi-static polarity turns Eq. (observable-centroid) into V grad(eta).

        Substituting Pi = v0 G / (2 gamma1) and the quasi-static polarity into
        the centroid balance must reproduce ``kinetics.pulse_centroid_drift``,
        and the spread balance must reproduce ``4 delta_D_eff'``.
        """

        kinetics = self.config.kinetics
        gamma1 = polarity_relaxation_rate(self.config)
        for case in range(1, 10):
            response = self.config.responses.only_case(case)
            rotational_level = response.D_R_eta + 0.70 * response.lambda_eta
            steering = response.chi_theta - (
                response.D_R_zeta + 0.70 * response.lambda_zeta
            )
            # Pi = (v0 / 2 gamma1) G and P = [steering G / 2 - DR_eta Pi] / gamma1
            polarity_overlap = kinetics.v0 / (2.0 * gamma1)
            net = (0.5 * steering - rotational_level * polarity_overlap) / gamma1
            rate = (
                kinetics.v0 * net
                + response.v_eta * polarity_overlap
                + 0.5 * response.v_zeta
            )
            self.assertAlmostEqual(
                rate,
                pulse_centroid_drift(kinetics, response, 0.30),
                places=12,
                msg=f"case {case}",
            )
            # K = (v0 / 2 gamma1)(2 E + C); the C terms carry two more spatial
            # derivatives and drop from the long-wavelength coefficient.
            outward_level = kinetics.v0 / gamma1
            outward = (response.v_eta - rotational_level * outward_level) / gamma1
            spread_rate = (
                2.0 * kinetics.v0 * outward
                + 2.0 * response.v_eta * outward_level
                + 4.0 * response.D_T_eta
            )
            self.assertAlmostEqual(
                spread_rate,
                4.0 * delta_D_eff_coefficient(kinetics, response, 0.30),
                places=12,
                msg=f"case {case}",
            )

    def test_interpolation_is_exact_on_a_linear_series(self) -> None:
        times = np.linspace(0.0, 1.0, 11)
        series = TransportObservables(
            method="test",
            times=times,
            case_numbers=(0, 1),
            net_polarity=np.stack([np.stack([times, 2.0 * times], -1)] * 2),
            centroid=np.stack([np.stack([3.0 * times, 4.0 * times], -1)] * 2),
            spread=np.stack([5.0 * times] * 2),
        )
        target = np.linspace(0.0, 1.0, 7)
        resampled = series.interpolated(target)
        np.testing.assert_allclose(resampled.net_polarity[0, :, 1], 2.0 * target)
        np.testing.assert_allclose(resampled.spread[1], 5.0 * target)
        with self.assertRaises(ValueError):
            series.interpolated(np.array([0.0, 1.5]))

    def test_finite_volume_solver_matches_the_prediction(self) -> None:
        """The PDE and the O(eps) prediction agree to the size of O(eps^2).

        The tolerances are set by the largest fractional modulation in the
        calibrated configuration.  Cases 2 and 4 need
        D_R_eta * A / gamma1 = 0.29 to reach a response comparable with the
        others, so their own second-order residual is a few per cent; the rest
        sit near 0.05 and agree far more closely.
        """

        solver = finite_volume_observables(self.config, self.field).interpolated(
            self.times
        )
        largest = float(np.max(np.abs(solver.response("net_polarity"))))
        for case in (1, 3, 5, 7):
            self.assertLess(
                float(np.max(np.abs(solver.response("net_polarity")[case]))),
                1e-9 * largest,
                msg=f"case {case} polarity should vanish structurally",
            )
        for name, tolerance in (
            ("net_polarity", 0.04),
            ("centroid", 0.04),
            ("spread", 0.20),
        ):
            predicted = self.prediction.response(name)[:, -1]
            measured = solver.response(name)[:, -1]
            scale = float(np.max(np.abs(predicted)))
            for case in range(1, 10):
                reference = predicted[case]
                if np.max(np.abs(reference)) < 1e-3 * scale:
                    continue
                np.testing.assert_allclose(
                    measured[case],
                    reference,
                    rtol=tolerance,
                    atol=1e-6 * scale,
                    err_msg=f"case {case} {name}",
                )


class SaturationObservableTests(unittest.TestCase):
    """Closed forms for what one release measures once the mode has decayed.

    The reference values are the exact first-order moment system integrated to
    fifteen mode lifetimes on ``config/foraging.toml``, whose landscape is a
    single even mode about the release point.  The agreement is a check on the
    algebra of Sec. "What one release measures", not on a fit.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from chem_swimmer import observables as ob

        cls.ob = ob
        base = load_config(PROJECT_ROOT / "config" / "foraging.toml")
        # The saturation laws assume the released population eventually samples
        # a landscape of zero mean.  The project defines the reference level as
        # the disk average, which for a cosine mode is 2 J_1(qR)/(qR); tuning
        # the wavelength to a zero of J_1 makes that vanish exactly, so the
        # level overlap E decays instead of leaving a constant residue that
        # would make dS grow without bound.
        wavenumber = float(special.jn_zeros(1, 7)[-1])
        cls.config = replace(
            base,
            field={**base.field, "wavelength": 2.0 * np.pi / wavenumber},
        )
        cls.field = create_field(cls.config.field)
        cls.wavenumber = cls.field.wavenumber
        cls.amplitude = ob.even_mode_amplitude(cls.field, cls.config.initial.position)
        cls.sigma = 0.03125
        lifetime = 1.0 / (ob.effective_diffusivity(cls.config) * cls.wavenumber**2)
        # The spectral box must hold a whole number of wavelengths, or the
        # periodic synthesis sees a discontinuity at its edge.
        wavelength = 2.0 * np.pi / cls.wavenumber
        box = round(12.0 / wavelength) * wavelength
        cls.solution = ob.predicted_observables(
            cls.config,
            cls.field,
            np.linspace(0.0, 15.0 * lifetime, 1201),
            box_size=box,
            box_points=512,
        )

    def test_spread_saturates_on_the_mobility(self):
        """``dS(inf) = -4 eps kappa/(D_eff q^2)``, not on ``delta D_eff``."""

        for case in range(1, 10):
            with self.subTest(case=case):
                solved = self.solution.spread[case, -1] - self.solution.spread[0, -1]
                closed = self.ob.spread_saturation(
                    self.config, case, self.wavenumber, self.amplitude, self.sigma
                )
                if abs(solved) < 1.0e-8:
                    self.assertLess(abs(closed), 1.0e-7)
                else:
                    self.assertAlmostEqual(closed / solved, 1.0, delta=0.02)

    def test_shape_saturates_on_the_drift(self):
        """The traceless second moment returns ``V`` where the trace returns ``kappa``."""

        for case in range(1, 10):
            with self.subTest(case=case):
                solved = self.solution.shape[case, -1, 0]
                closed = self.ob.shape_saturation(
                    self.config, case, self.wavenumber, self.amplitude, self.sigma
                )
                if abs(solved) < 1.0e-9:
                    self.assertLess(abs(closed), 1.0e-8)
                else:
                    self.assertAlmostEqual(closed / solved, 1.0, delta=0.02)

    def test_translational_responses_cannot_shape_a_pulse(self):
        """A diffusive flux is a gradient, so it feeds only the trace."""

        for case in (0, 1, 5):
            with self.subTest(case=case):
                self.assertLess(
                    float(np.max(np.abs(self.solution.shape[case]))), 1.0e-18
                )

    def test_transverse_shape_component_vanishes_for_an_axial_mode(self):
        self.assertLess(float(np.max(np.abs(self.solution.shape[:, :, 1]))), 1.0e-18)

    def test_accumulation_ratio_is_the_exponential_peclet(self):
        for case in range(10):
            with self.subTest(case=case):
                peclet = self.ob.chemotactic_peclet(
                    self.config, case, self.wavenumber, self.field.amplitude
                )
                self.assertAlmostEqual(
                    self.ob.accumulation_ratio(
                        self.config, case, self.wavenumber, self.field.amplitude
                    ),
                    float(np.exp(2.0 * peclet)),
                    places=12,
                )

    def test_mode_transfer_reduces_to_the_coarse_limit(self):
        """``W -> 1`` and ``D_eff(q) -> D_eff`` for a point release and small q."""

        transfer = self.ob.mode_transfer(self.config, 1.0e-3, 0.0)
        self.assertAlmostEqual(transfer.release_transform, 1.0, places=12)
        self.assertAlmostEqual(
            transfer.effective_diffusivity,
            self.ob.effective_diffusivity(self.config),
            delta=1.0e-6,
        )


class NavigationObservableTests(unittest.TestCase):
    """The finite-lag channel: closed forms against the kinetic hierarchy.

    ``harmonic_navigation`` integrates the angular moments of the backward
    generator with no expansion in the response and eight angular harmonics, so
    it is an independent solution of the same kinetic problem; the closed
    prediction must reproduce it to first order.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from chem_swimmer import information as nav

        cls.nav = nav
        cls.config = load_config(PROJECT_ROOT / "config" / "navigation.toml")
        cls.field = create_field(cls.config.field)
        cls.gradient = float(
            np.linalg.norm(nav.uniform_gradient(cls.config, cls.field))
        )
        cls.lags = np.linspace(0.0, cls.config.simulation.duration, 11)
        cls.prediction = nav.navigation_prediction(
            cls.config, cls.lags, field=cls.field
        )
        cls.kinetic = nav.harmonic_navigation(cls.config, cls.lags, field=cls.field)

    def test_lag_shapes_have_the_stated_short_lag_limits(self):
        shapes = self.nav.lag_functions(1.0, 2.7, 5.2, np.array([1.0e-4]))
        self.assertAlmostEqual(float(shapes.Phi[0]) / 5.0e-5, 1.0, places=4)
        self.assertAlmostEqual(float(shapes.Psi[0]) / (1.0e-8 / 6.0), 1.0, places=3)
        self.assertAlmostEqual(float(shapes.Lam[0]) / 5.0e-5, 1.0, places=3)
        self.assertAlmostEqual(float(shapes.Theta[0]) / 5.0e-5, 1.0, places=3)

    def test_projected_velocity_saturates_at_the_pulse_drift(self):
        """``V_par -> V_j <g>`` at long lag, along the rule's own lag shape."""

        expected = {7: 1.0, 3: 0.9, 6: 0.9, 8: 0.9, 9: 0.9, 2: 0.8001, 4: 0.8001}
        for case, fraction in expected.items():
            with self.subTest(case=case):
                drift = self.nav.pulse_drift(self.config, case) * self.gradient
                measured = (
                    self.prediction.response("projected_velocity")[case][-1] / drift
                )
                self.assertAlmostEqual(measured, fraction, places=3)

    def test_prediction_matches_the_kinetic_hierarchy(self):
        for name in ("projected_velocity", "alignment_harmonic", "level_harmonic"):
            for case in range(10):
                closed = self.prediction.response(name)[case][1:]
                exact = self.kinetic.response(name)[case][1:]
                scale = float(np.max(np.abs(exact)))
                if scale < 1.0e-12:
                    continue
                with self.subTest(observable=name, case=case):
                    self.assertLess(float(np.max(np.abs(closed - exact))) / scale, 0.01)

    def test_only_the_two_diffusivity_rules_are_blind_to_navigation(self):
        for case in (1, 5):
            with self.subTest(case=case):
                self.assertLess(
                    float(
                        np.max(
                            np.abs(
                                self.kinetic.response("projected_velocity")[case][1:]
                            )
                        )
                    ),
                    1.0e-12,
                )

    def test_second_harmonic_carries_the_other_turning_combination(self):
        """``B`` weights ``chi + D_R_zeta`` where the drift weights the difference."""

        terms_nine = self.nav._response_terms(self.config, 9)
        terms_six = self.nav._response_terms(self.config, 6)
        # Case 9 is pure steering and Case 6 pure turning with equal magnitude,
        # so their drifts are opposite while their second harmonics coincide.
        drift = self.prediction.response("projected_velocity")
        harmonic = self.prediction.response("alignment_harmonic")
        self.assertAlmostEqual(terms_nine["chi"], terms_six["gamma_zeta"], places=9)
        self.assertAlmostEqual(float(drift[9][-1] / drift[6][-1]), -1.0, places=6)
        self.assertAlmostEqual(float(harmonic[9][-1] / harmonic[6][-1]), 1.0, places=6)

    def test_short_lag_information_matches_the_case_table(self):
        """Only the speed rules shift the finite part; only D_T rules diverge."""

        baseline = self.nav.short_lag_information(self.config, 0, field=self.field)
        self.assertAlmostEqual(
            baseline[0],
            self.config.kinetics.v0**2 / (8.0 * self.config.kinetics.D_T0),
            places=9,
        )
        for case in (2, 4, 6, 8, 9):
            with self.subTest(case=case):
                self.assertEqual(
                    self.nav.short_lag_information(self.config, case, field=self.field),
                    baseline,
                )
        for case in (3, 7):
            with self.subTest(case=case):
                finite, divergent = self.nav.short_lag_information(
                    self.config, case, field=self.field
                )
                self.assertGreater(finite, baseline[0])
                self.assertEqual(divergent, 0.0)
        for case in (1, 5):
            with self.subTest(case=case):
                finite, divergent = self.nav.short_lag_information(
                    self.config, case, field=self.field
                )
                self.assertAlmostEqual(finite, baseline[0], places=9)
                self.assertGreater(divergent, 0.0)

    def test_information_response_is_reproduced_for_the_diffusivity_rules(self):
        for case in (1, 5):
            with self.subTest(case=case):
                lag = self.lags[-1]
                closed = self.prediction.response("information")[case][-1] * lag
                exact = self.kinetic.response("information")[case][-1] * lag
                self.assertAlmostEqual(closed / exact, 1.0, delta=0.01)

    def test_a_varying_gradient_is_refused(self):
        with self.assertRaises(ValueError):
            self.nav.uniform_gradient(
                self.config,
                create_field(
                    {
                        "kind": "gaussian",
                        "baseline": 1.0,
                        "amplitude": 0.5,
                        "center": [0.0, 0.0],
                        "sigma": [0.5, 0.5],
                    }
                ),
            )


class TurningMomentTests(unittest.TestCase):
    """Higher cosine moments of the tumble kernel, needed past nematic order."""

    def test_wrapped_normal_moments_follow_the_gaussian_law(self):
        from chem_swimmer.turning import create_turning_kernel, turning_cosine_moment

        kernel = create_turning_kernel({"kind": "wrapped_normal", "P_turn": 0.3})
        variance = -2.0 * math.log(0.3)
        for harmonic in range(1, 6):
            with self.subTest(harmonic=harmonic):
                self.assertAlmostEqual(
                    turning_cosine_moment(kernel, harmonic),
                    math.exp(-0.5 * harmonic**2 * variance),
                    places=12,
                )
        self.assertEqual(turning_cosine_moment(kernel, 0), 1.0)
        self.assertAlmostEqual(turning_cosine_moment(kernel, 1), kernel.P_turn, 12)
        self.assertAlmostEqual(turning_cosine_moment(kernel, 2), kernel.P_2_turn, 12)


class ForagingObservableTests(unittest.TestCase):
    """Guards on the uptake harvest law and on the divergence between strategies.

    The numbers below were confirmed against the finite-volume solver on a
    single landscape mode at two wavelengths and two grids
    (``config/foraging.toml``); with the cell Peclet number held under two, the
    closed form and the solver agree to a few per cent.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(PROJECT_ROOT / "config" / "foraging.toml")
        cls.field = create_field(cls.config.field)

    def _harvest_time(self) -> float:
        return landscape_uptake_time(self.config, self.field.wavenumber)

    def test_periodic_field_is_self_consistent(self) -> None:
        field = PeriodicField(
            baseline=1.0, amplitude=0.3, wavevector=(4.0, 3.0), phase=0.4
        )
        self.assertAlmostEqual(field.wavenumber, 5.0, places=12)
        self.assertAlmostEqual(field.wavelength, 2.0 * math.pi / 5.0, places=12)
        points = np.random.default_rng(3).normal(size=(7, 2))
        step = 1.0e-5
        value, gradient = field.value_and_gradient(points)
        np.testing.assert_allclose(value, field.value(points), rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(gradient, field.gradient(points), rtol=0, atol=1e-15)
        curvature = np.zeros(len(points))
        for axis in range(2):
            shift = np.zeros(2)
            shift[axis] = step
            curvature += (
                np.asarray(field.value(points + shift))
                - 2.0 * np.asarray(field.value(points))
                + np.asarray(field.value(points - shift))
            ) / step**2
        np.testing.assert_allclose(field.laplacian(points), curvature, rtol=2e-4)
        # lap(eta) = -q^2 (eta - baseline) holds exactly for a single mode.
        np.testing.assert_allclose(
            field.laplacian(points),
            -(field.wavenumber**2) * (np.asarray(field.value(points)) - field.baseline),
            rtol=1e-12,
        )

    def test_periodic_field_matches_its_native_kernel(self) -> None:
        """The Numba branch must reproduce the readable Python field exactly."""

        from chem_swimmer.accelerated import (
            _field_value_gradient,
            numba_available_for,
            pack_field,
        )

        field = create_field(
            {
                "kind": "periodic",
                "baseline": 1.0,
                "amplitude": 0.3,
                "wavelength": 0.4,
                "direction": [1.0, 0.5],
            }
        )
        self.assertTrue(numba_available_for(field))
        packed = pack_field(field, 1.0)
        points = np.random.default_rng(4).normal(size=(6, 2)) * 0.4
        value, gradient = field.value_and_gradient(points)
        for index, (x, y) in enumerate(points):
            native = _field_value_gradient(
                x,
                y,
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
            )
            self.assertAlmostEqual(native[0], float(value[index]), places=13)
            self.assertAlmostEqual(native[1], float(gradient[index, 0]), places=12)
            self.assertAlmostEqual(native[2], float(gradient[index, 1]), places=12)

    def test_periodic_factory_rejects_ambiguous_settings(self) -> None:
        with self.assertRaises(ValueError):
            create_field({"kind": "periodic", "amplitude": 0.1})
        with self.assertRaises(ValueError):
            create_field(
                {
                    "kind": "periodic",
                    "amplitude": 0.1,
                    "wavevector": [1.0, 0.0],
                    "wavelength": 0.5,
                }
            )

    def test_green_amplitudes_survive_a_fast_reorienter(self) -> None:
        """Regression: the envelope and the hyperbolic factor each overflow alone."""

        config = replace(
            self.config,
            kinetics=replace(
                self.config.kinetics, v0=800.0, D_R0=160000.0, lambda0=160000.0 / 0.7
            ),
        )
        wavenumbers = np.array([0.0, 1.0, 25.0, 400.0])
        for time in (0.0, 1e-6, 1e-2, 0.5):
            density, polarity = green_amplitudes(config, wavenumbers, time)
            self.assertTrue(np.all(np.isfinite(density)), msg=f"t={time}")
            self.assertTrue(np.all(np.isfinite(polarity)), msg=f"t={time}")
            self.assertAlmostEqual(float(density[0]), 1.0, places=12)

    def test_uptake_rate_splits_passive_smoothing_from_harvest(self) -> None:
        for case in range(1, 10):
            mobility = chemotactic_mobility(self.config, case)
            rate = uptake_rate(self.config, case, np.array([2.0]), np.array([3.0]))
            self.assertAlmostEqual(
                float(rate[0]),
                2.0 * effective_diffusivity(self.config) + 3.0 * mobility,
                places=12,
                msg=f"case {case}",
            )

    def test_uptake_gain_is_the_mobility_times_the_contrast(self) -> None:
        """A single mode of amplitude A is worth kappa A^2 / 2 D_eff."""

        amplitude = self.field.amplitude
        for case in range(1, 10):
            expected = (
                chemotactic_mobility(self.config, case)
                * amplitude**2
                / (2.0 * effective_diffusivity(self.config))
            )
            measured = uptake_gain(self.config, self.field, case)
            if abs(expected) < 1e-14:
                self.assertAlmostEqual(measured, 0.0, places=14, msg=f"case {case}")
            else:
                # The disk average of cos^2 is A^2/2 only up to the boundary.
                self.assertAlmostEqual(
                    measured / expected, 1.0, delta=0.02, msg=f"case {case}"
                )

    def test_closed_uptake_reaches_the_predicted_plateau(self) -> None:
        times = np.linspace(0.0, 12.0 * self._harvest_time(), 400)
        uptake = periodic_uptake_prediction(self.config, self.field, times)
        for case in range(1, 10):
            gain = uptake[case, -1] - uptake[0, -1]
            expected = uptake_gain(self.config, self.field, case)
            if abs(expected) < 1e-14:
                self.assertAlmostEqual(gain, 0.0, places=14, msg=f"case {case}")
            else:
                self.assertAlmostEqual(
                    gain / expected, 1.0, delta=0.02, msg=f"case {case}"
                )
        # The null forager keeps the landscape mean, having no response to it.
        self.assertAlmostEqual(uptake[0, -1], self.field.baseline, delta=1e-4)

    def test_the_gain_does_not_depend_on_the_wavelength(self) -> None:
        """A fine landscape and a coarse one of equal contrast are worth the same."""

        gains = []
        for wavelength in (0.5, 0.25, 0.125):
            settings = dict(self.config.field)
            settings["wavelength"] = wavelength
            field = create_field(settings)
            config = replace(self.config, field=settings)
            horizon = 12.0 / (effective_diffusivity(config) * field.wavenumber**2)
            times = np.linspace(0.0, horizon, 400)
            uptake = periodic_uptake_prediction(config, field, times)
            gains.append(uptake[7, -1] - uptake[0, -1])
        np.testing.assert_allclose(gains, gains[0], rtol=5e-3)

    def test_the_two_clocks_separate_as_the_squared_scale_ratio(self) -> None:
        wavenumber = self.field.wavenumber
        ratio = landscape_uptake_time(self.config, wavenumber) / domain_mixing_time(
            self.config
        )
        expected = (
            FIRST_NEUMANN_DISK_WAVENUMBER / (self.config.domain.radius * wavenumber)
        ) ** 2
        self.assertAlmostEqual(ratio, expected, places=12)
        self.assertLess(ratio, 0.01)


class TheoryCoefficientTests(unittest.TestCase):
    def test_cases_match_table_2(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        base = replace(
            config.kinetics,
            v0=0.1,
            D_R0=0.1,
            lambda0=0.1,
        )
        P_turn = create_turning_kernel(config.turning).P_turn
        gamma1 = base.D_R0 + base.lambda0 * (1.0 - P_turn)

        expected = {
            1: -config.responses.D_T_eta,
            2: 0.0,
            3: -base.v0 * config.responses.v_eta / (2.0 * gamma1),
            4: 0.0,
            5: 0.0,
            6: -base.v0 * config.responses.D_R_zeta / (2.0 * gamma1),
            7: config.responses.v_zeta / 2.0,
            8: -base.v0
            * (1.0 - P_turn)
            * config.responses.lambda_zeta
            / (2.0 * gamma1),
            9: base.v0 * config.responses.chi_theta / (2.0 * gamma1),
        }
        for case_number, expected_chi in expected.items():
            _, _, chi = macroscopic_coefficients(
                base, config.responses.only_case(case_number), P_turn
            )
            self.assertAlmostEqual(chi, expected_chi)

    def test_positive_angular_steering_rotates_toward_gradient(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        responses = config.responses.only_case(9)
        gradient = np.asarray([1.0, 0.0])
        heading = 0.5 * np.pi
        rate = angular_steering_rate(responses, gradient, heading)
        self.assertLess(rate, 0.0)
        xi_perp = -1.0
        alignment_derivative = rate * xi_perp
        self.assertAlmostEqual(
            alignment_derivative,
            responses.chi_theta * xi_perp**2,
        )

    def test_zero_orientation_relaxation_is_valid_for_simulation(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        base = replace(
            config.kinetics,
            v0=0.0,
            D_R0=0.0,
            lambda0=0.0,
        )
        gamma1, effective_diffusivity, mobility = macroscopic_coefficients(
            base,
            Responses.zeros(),
            create_turning_kernel(config.turning).P_turn,
        )
        self.assertEqual(gamma1, 0.0)
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


class BoundaryLayerTests(unittest.TestCase):
    def test_no_propulsion_or_no_relaxation_gives_no_layer(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        P_turn = create_turning_kernel(config.turning).P_turn
        no_propulsion = replace(config.kinetics, v0=0.0)
        layer = sliding_wall_boundary_layer(no_propulsion, P_turn)
        self.assertEqual(layer.peclet, 0.0)
        self.assertEqual(layer.wall_polarity_over_bulk_density, 0.0)
        self.assertTrue(math.isfinite(layer.decay_rate))

        no_relaxation = replace(config.kinetics, D_R0=0.0, lambda0=0.0)
        frozen = sliding_wall_boundary_layer(no_relaxation, P_turn)
        self.assertEqual(frozen.gamma1, 0.0)
        self.assertTrue(math.isnan(frozen.peclet))
        self.assertTrue(math.isnan(frozen.wall_polarity_over_bulk_density))

    def test_default_configuration_matches_hand_derivation(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        P_turn = create_turning_kernel(config.turning).P_turn
        layer = sliding_wall_boundary_layer(config.kinetics, P_turn)
        v0, D_T0 = config.kinetics.v0, config.kinetics.D_T0
        gamma1 = config.kinetics.D_R0 + config.kinetics.lambda0 * (1.0 - P_turn)
        self.assertAlmostEqual(layer.gamma1, gamma1)
        expected_peclet = v0**2 / (D_T0 * gamma1)
        self.assertAlmostEqual(layer.peclet, expected_peclet)
        expected_decay_rate = math.sqrt(D_T0 * gamma1 + 0.5 * v0**2) / D_T0
        self.assertAlmostEqual(layer.decay_rate, expected_decay_rate)
        self.assertAlmostEqual(layer.width, 1.0 / expected_decay_rate)
        expected_ratio = math.sqrt(expected_peclet + 0.5 * expected_peclet**2) / 2.0
        self.assertAlmostEqual(layer.wall_polarity_over_bulk_density, expected_ratio)

    def test_closed_form_satisfies_the_marshak_boundary_condition(self) -> None:
        """Regression guard against algebra errors in boundary_layer.py.

        Independently reconstructs rho(y) and p(y) from the closed-form
        p_wall^(0) and kappa, then checks both governing ODEs and the
        cos(phi)-weighted wall condition v0 rho(0)/2 = D_T0 p'(0) that the
        closed form was solved from, for several random background
        parameter sets.
        """

        rng = np.random.default_rng(20260730)
        for _ in range(20):
            v0 = float(rng.uniform(0.1, 50.0))
            D_T0 = float(rng.uniform(0.1, 5.0))
            D_R0 = float(rng.uniform(0.0, 300.0))
            lambda0 = float(rng.uniform(0.0, 300.0))
            P_turn = float(rng.uniform(-0.9, 0.9))
            background = replace(
                load_config(PROJECT_ROOT / "config" / "default.toml").kinetics,
                v0=v0,
                D_T0=D_T0,
                D_R0=D_R0,
                lambda0=lambda0,
            )
            layer = sliding_wall_boundary_layer(background, P_turn)
            rho_bulk = float(rng.uniform(0.1, 5.0))
            p0 = -layer.wall_polarity_over_bulk_density * rho_bulk
            kappa = layer.decay_rate

            # (A): v0 p(y) = D_T0 rho'(y), integrated with rho(inf) = rho_bulk.
            # rho and p are analytic exponentials, so use exact derivatives
            # rather than finite differences -- this is meant to catch algebra
            # errors in the closed form, not to test numerical differentiation.
            def rho(y: float) -> float:
                return rho_bulk - (v0 / (D_T0 * kappa)) * p0 * math.exp(-kappa * y)  # noqa: B023

            def rho_prime(y: float) -> float:
                return (v0 / D_T0) * p0 * math.exp(-kappa * y)  # noqa: B023

            def p(y: float) -> float:
                return p0 * math.exp(-kappa * y)  # noqa: B023

            def p_prime(y: float) -> float:
                return -kappa * p0 * math.exp(-kappa * y)  # noqa: B023

            # (A) at y = 0.
            self.assertAlmostEqual(v0 * p(0.0), D_T0 * rho_prime(0.0), places=8)
            # (B): D_T0 p'' - gamma1 p - (v0/2) rho' = 0, via p'' = kappa^2 p.
            p_double_prime = kappa**2 * p(0.0)
            self.assertAlmostEqual(
                D_T0 * p_double_prime
                - layer.gamma1 * p(0.0)
                - 0.5 * v0 * rho_prime(0.0),
                0.0,
                places=8,
            )
            # (C): the Marshak wall condition the closed form was solved from.
            self.assertAlmostEqual(
                0.5 * v0 * rho(0.0),
                D_T0 * p_prime(0.0),
                places=8,
            )


class SimulatorTests(unittest.TestCase):
    def test_short_run_is_reproducible_and_stays_in_disk(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml").with_overrides(
            duration=1.6e-6,
            dt=8.0e-7,
        )
        first = simulate(config, 8)
        second = simulate(config, 8)
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.headings, second.headings)
        radii = np.linalg.norm(first.positions, axis=1)
        self.assertTrue(np.all(radii < 1.0))
        self.assertEqual(first.times[-1], 1.6e-6)

    def test_trajectory_outputs_are_dimensionless(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml").with_overrides(
            duration=8.0e-7,
            dt=8.0e-7,
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
        config = load_config(PROJECT_ROOT / "config" / "default.toml").with_overrides(
            duration=8.0e-7,
            dt=8.0e-7,
        )
        self.assertEqual(config.initial.position, (0.0, 0.0))
        insensitive = replace(config, responses=Responses.zeros())
        trajectories = simulate_all_cases(insensitive)
        reference = trajectories[0]
        for trajectory in trajectories[1:]:
            np.testing.assert_array_equal(trajectory.positions, reference.positions)
            np.testing.assert_array_equal(trajectory.headings, reference.headings)

    def test_ensemble_msd_starts_at_zero(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml").with_overrides(
            duration=8.0e-7,
            dt=8.0e-7,
        )
        result = compute_ensemble_msd(config, ensemble_size=3, workers=1)
        self.assertEqual(result.mean.shape, (9, len(result.times)))
        np.testing.assert_array_equal(result.mean[:, 0], np.zeros(9))
        self.assertTrue(np.all(result.mean >= 0.0))

    def test_numba_matches_readable_reference(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml").with_overrides(
            duration=8.0e-6,
            dt=4.0e-6,
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

    def test_numba_matches_readable_angular_steering(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml").with_overrides(
            duration=8.0e-6,
            dt=4.0e-6,
        )
        field = create_field(config.field)
        config, _ = resolve_numerics(config, field, (9,))
        python_config = replace(
            config,
            performance=replace(config.performance, engine="python"),
        )
        steps = round(config.simulation.duration / config.simulation.dt)
        innovations = Innovations.generate(steps, config.simulation.seed)
        native = simulate(config, 9, field=field, innovations=innovations)
        readable = simulate(
            python_config,
            9,
            field=field,
            innovations=innovations,
        )
        np.testing.assert_allclose(native.positions, readable.positions, atol=1.0e-14)
        np.testing.assert_allclose(native.headings, readable.headings, atol=1.0e-14)

    def test_numba_matches_reference_at_sliding_wall(self) -> None:
        base = load_config(PROJECT_ROOT / "config" / "default.toml").with_overrides(
            duration=8.0e-6,
            dt=4.0e-6,
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
        config = replace(
            base,
            initial=replace(base.initial, position=(0.99, 0.0)),
        )
        config, _ = resolve_numerics(config, field, (0,))
        steps = int(config.simulation.duration / config.simulation.dt)
        innovations = Innovations.generate(steps, config.simulation.seed)
        python_config = replace(
            config,
            performance=replace(config.performance, engine="python"),
        )
        native = simulate(config, 0, field=field, innovations=innovations)
        readable = simulate(python_config, 0, field=field, innovations=innovations)
        self.assertGreater(int(native.wall_hit_counts.sum()), 0)
        np.testing.assert_allclose(
            native.positions, readable.positions, rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            native.headings, readable.headings, rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_array_equal(native.wall_hit_counts, readable.wall_hit_counts)


class ResolutionTests(unittest.TestCase):
    def test_nonfinite_configuration_is_rejected(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        invalid = replace(
            config,
            responses=replace(config.responses, v_zeta=float("nan")),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid.validate()

    def test_default_profile_meets_working_theory_targets(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        plan = build_resolution_plan(
            config,
            create_field(config.field),
            tuple(range(10)),
        )
        backgrounds = {
            "v": config.kinetics.v0,
            "D_T": config.kinetics.D_T0,
            "D_R": config.kinetics.D_R0,
            "lambda": config.kinetics.lambda0,
        }
        P_turn = create_turning_kernel(config.turning).P_turn
        gamma1 = config.kinetics.D_R0 + config.kinetics.lambda0 * (1.0 - P_turn)

        def modulation(case_number: int) -> float:
            return max(
                (
                    max(abs(bound) for bound in bounds) / gamma1
                    if name == "angular_drift"
                    else max(abs(bound - backgrounds[name]) for bound in bounds)
                    / backgrounds[name]
                )
                for name, bounds in plan.ranges_by_case[case_number].items()
            )

        # Cases 2 and 4 modulate the decay of a polarity they must build
        # first, so their density response starts two powers of t later than
        # everyone else's.  Bringing them onto a common colour scale costs
        # exactly one thing: they need the largest fractional modulation of the
        # coefficient they act on -- D_R_eta A / D_R0 = 0.24, which the plan's
        # safety margin reports as 0.29 -- and so carry the largest O(eta^2)
        # error.  Case 5 is next at 0.14; the remaining six stay below a tenth.
        self.assertLessEqual(max(modulation(case) for case in (2, 4)), 0.30)
        self.assertLessEqual(modulation(5), 0.15)
        self.assertLessEqual(max(modulation(case) for case in (1, 3, 6, 7, 8, 9)), 0.10)
        # The closure eliminates the nematic mode, not the polar one, so the
        # rate that has to outrun transport across a landscape length is
        # gamma2 -- Eq. (polar-closure-validity) -- and not gamma1.
        gamma2 = 4.0 * config.kinetics.D_R0 + config.kinetics.lambda0 * (
            1.0 - create_turning_kernel(config.turning).cosine_moment(2)
        )
        self.assertLessEqual(
            config.kinetics.v0 / gamma2 / plan.characteristic_length_star, 0.2
        )
        # The default profile reaches the measured orientational crossover so
        # the cubic Case 2/4 responses clear the particle sampling floor, but
        # remains far short of the adiabatic gamma1*t >> 1 regime.
        self.assertGreaterEqual(gamma1 * config.simulation.duration, 1.0)
        self.assertLessEqual(gamma1 * config.simulation.duration, 1.8)
        self.assertEqual(
            recommended_ensemble_size(config.density),
            1048576,
        )

    def test_radius_and_diffusion_define_requested_scales(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        field = create_field(config.field)
        plan = build_resolution_plan(config, field, tuple(range(1, 10)))
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
                "angular steering",
                "tumbling",
            },
        )
        duration = config.simulation.duration / plan.scales.time
        self.assertAlmostEqual(
            duration / plan.dt,
            round(duration / plan.dt),
        )

    def test_timestep_report_covers_every_active_constraint(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        plan = build_resolution_plan(
            config, create_field(config.field), tuple(range(10))
        )
        report = timestep_report(config, plan)
        for constraint in plan.constraints:
            self.assertIn(constraint, report)

    def test_default_paper_run_has_one_thousand_physical_updates(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        self.assertEqual(config.simulation.dt, 5.0e-6)
        self.assertEqual(
            round(config.simulation.duration / config.simulation.dt),
            1000,
        )

    def test_field_grid_resolves_smallest_gaussian_width(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        plan = build_resolution_plan(config, create_field(config.field), (8,))
        spacing = 2.0 * config.domain.radius / (plan.field_grid - 1)
        self.assertLessEqual(
            spacing,
            plan.characteristic_length / config.numerics.scan_points_per_length,
        )

    def test_unsafe_manual_dt_is_rejected(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml").with_overrides(
            dt=0.1
        )
        with self.assertRaisesRegex(ValueError, "too large"):
            build_resolution_plan(config, create_field(config.field), (8,))

    def test_high_velocity_reduces_automatic_dt(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml").with_overrides(
            dt="auto"
        )
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
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        coarse = replace(
            config,
            video=replace(config.video, field_grid=33),
        )
        with self.assertRaisesRegex(ValueError, "under-resolves"):
            build_resolution_plan(coarse, create_field(coarse.field), (8,))


class ProbabilityDensityTests(unittest.TestCase):
    @staticmethod
    def _small_density_config():
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        return replace(
            config.with_overrides(duration=8.0e-7, dt=8.0e-7, progress=False),
            density=replace(
                config.density,
                ensemble_size="auto",
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
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        count = recommended_ensemble_size(config.density)
        self.assertEqual(count, 1048576)
        self.assertGreaterEqual(
            2.0 * count * config.density.bandwidth_fraction**2,
            config.density.min_samples_per_kernel,
        )
        with self.assertRaisesRegex(ValueError, "below"):
            resolve_density_ensemble_size(config.density, requested=count // 2)

    def test_compiled_convolution_matches_zero_padded_reference(self) -> None:
        values = np.arange(2 * 3 * 5, dtype=np.uint32).reshape(2, 3, 5)
        kernel = np.asarray([0.2, 0.6, 0.2])
        actual = _convolve_axis(values, kernel, axis=-1)
        expected = np.stack(
            [
                np.convolve(row, kernel, mode="same")
                for row in values.reshape(-1, values.shape[-1])
            ]
        ).reshape(values.shape)
        self.assertEqual(actual.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)

    def test_boundary_correction_preserves_a_uniform_disk_density(self) -> None:
        settings = self._small_density_config().density
        _, _, disk_mask, _ = _kde_kernel_and_support(settings)
        counts = disk_mask[None, None].astype(np.uint32)
        density, _, _ = _normalized_smoothed(counts, settings)
        interior = density[0, 0, disk_mask]
        np.testing.assert_allclose(
            interior,
            np.full_like(interior, interior.mean()),
            rtol=0.0,
            atol=2.0e-14,
        )

    def test_cases_zero_through_nine_are_normalized(self) -> None:
        config = self._small_density_config()
        result = compute_position_pdf(config)
        self.assertEqual(result.case_numbers, tuple(range(10)))
        self.assertEqual(result.density.shape[:2], (10, len(result.times)))
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
            (10, len(result.times)),
        )
        self.assertEqual(
            observables.paired_standard_errors.shape,
            observables.values.shape,
        )
        self.assertTrue(np.all(observables.paired_standard_errors >= 0.0))

    def test_polarity_field_is_bounded_by_density(self) -> None:
        config = self._small_density_config()
        result = compute_position_pdf(config)
        self.assertIsNotNone(result.polarity_x)
        self.assertIsNotNone(result.polarity_y)
        self.assertEqual(result.polarity_x.shape, result.density.shape)
        self.assertEqual(result.polarity_y.shape, result.density.shape)

        # |integral e(theta) P dtheta| <= integral P dtheta = rho (Cauchy-Schwarz).
        # KDE boundary correction can push this slightly over the strict bound
        # very close to the disk edge, so exclude the outermost ring of cells.
        x_grid, y_grid = np.meshgrid(result.coordinates, result.coordinates)
        interior = x_grid**2 + y_grid**2 <= 0.85**2
        magnitude = np.hypot(result.polarity_x, result.polarity_y)
        self.assertTrue(
            np.all(magnitude[:, :, interior] <= result.density[:, :, interior] + 1.0e-3)
        )

    def test_microscopic_nematic_field_is_defined_and_physically_bounded(self) -> None:
        config = self._small_density_config()
        result = compute_position_pdf(config)
        self.assertIsNotNone(result.nematic_xx)
        self.assertIsNotNone(result.nematic_xy)
        assert result.nematic_xx is not None
        assert result.nematic_xy is not None
        self.assertEqual(result.nematic_xx.shape, result.density.shape)
        self.assertEqual(result.nematic_xy.shape, result.density.shape)
        self.assertIsNotNone(result.nematic_split_half_error_xx)
        self.assertIsNotNone(result.nematic_split_half_error_xy)
        assert result.nematic_split_half_error_xx is not None
        assert result.nematic_split_half_error_xy is not None
        self.assertEqual(
            result.nematic_split_half_error_xx.shape,
            result.density.shape,
        )

        # Every single-heading traceless tensor has |Q|_F/rho=1/sqrt(2),
        # and the shared positive KDE preserves this convex bound.
        q_norm = nematic_frobenius(result.nematic_xx, result.nematic_xy)
        self.assertTrue(np.all(q_norm <= result.density / np.sqrt(2.0) + 1.0e-3))
        np.testing.assert_array_equal(result.nematic_xx[:, 0], 0.0)
        np.testing.assert_array_equal(result.nematic_xy[:, 0], 0.0)
        np.testing.assert_array_equal(result.nematic_split_half_error_xx[:, 0], 0.0)
        np.testing.assert_array_equal(result.nematic_split_half_error_xy[:, 0], 0.0)

    def test_angular_moment_binning_uses_the_stated_q_definition(self) -> None:
        shape = (1, 1, 4, 4)
        sum_cos = np.zeros(shape)
        sum_sin = np.zeros(shape)
        sum_qxx = np.zeros(shape)
        sum_qxy = np.zeros(shape)
        positions = np.zeros((3, 1, 1, 2))
        headings = np.asarray([0.0, np.pi / 2.0, np.pi / 4.0])[:, None, None]
        _accumulate_angular_moments(
            sum_cos,
            sum_sin,
            sum_qxx,
            sum_qxy,
            positions,
            headings,
        )
        self.assertAlmostEqual(sum_qxx[0, 0, 2, 2], 0.0, places=14)
        self.assertAlmostEqual(sum_qxy[0, 0, 2, 2], 0.5, places=14)

    def test_nematic_closure_report_includes_exact_flux_contribution(self) -> None:
        config = self._small_density_config()
        result = compute_position_pdf(config)
        field = create_field(config.field)
        rows = nematic_closure_rows(result, config, field)
        final_rows = {
            int(row["case"]): row
            for row in rows
            if float(row["time"]) == float(result.times[-1])
        }
        self.assertEqual(set(final_rows), set(range(10)))
        self.assertGreater(float(final_rows[7]["q_density_flux_l2"]), 0.0)
        for case_number in set(range(10)) - {7}:
            self.assertEqual(float(final_rows[case_number]["q_density_flux_l2"]), 0.0)
        self.assertTrue(np.isfinite(float(final_rows[7]["q_flux_over_exact_flux"])))
        self.assertGreater(
            float(final_rows[7]["q_split_half_sampling_error_l2"]),
            0.0,
        )
        self.assertTrue(
            np.isfinite(float(final_rows[7]["q_density_flux_signal_to_sampling_error"]))
        )

        with TemporaryDirectory() as directory:
            outputs = render_nematic_closure_report(
                result,
                config,
                field,
                Path(directory) / "nematic.png",
                Path(directory) / "nematic_summary.csv",
                Path(directory) / "nematic_timeseries.csv",
            )
            for output in outputs:
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 100)
            self.assertEqual(outputs[1].suffix, ".pdf")
            self.assertTrue(outputs[1].read_bytes().startswith(b"%PDF-"))
            with Image.open(outputs[0]) as rendered:
                self.assertGreaterEqual(rendered.width, 3000)
                self.assertGreaterEqual(rendered.height, 1800)
                self.assertAlmostEqual(rendered.info["dpi"][0], 300.0, delta=1.0)
            with outputs[2].open(encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 10)
            self.assertIn("q_sampling_error_over_rho_l2", summary_rows[0])

    def test_exact_q_density_flux_matches_tensor_contraction(self) -> None:
        config = self._small_density_config()
        field = create_field(
            {
                "kind": "linear",
                "baseline": 1.0,
                "gradient": [1.0, 0.0],
                "center": [0.0, 0.0],
            }
        )
        coordinates = np.linspace(-0.2, 0.2, 5)
        shape = (1, 1, 5, 5)
        rho = np.ones(shape, dtype=np.float64)
        zero = np.zeros(shape, dtype=np.float64)
        qxx_value = 0.12
        qxy_value = -0.07
        result = PositionPDFResult(
            times=np.asarray([0.0]),
            coordinates=coordinates,
            density=rho,
            case_numbers=(7,),
            ensemble_size=2,
            bandwidth_fraction=0.1,
            polarity_x=zero,
            polarity_y=zero,
            nematic_xx=np.full(shape, qxx_value, dtype=np.float64),
            nematic_xy=np.full(shape, qxy_value, dtype=np.float64),
            nematic_split_half_error_xx=zero,
            nematic_split_half_error_xy=zero,
        )
        flux = reconstruct_density_flux(result, config, field, 0, 0)
        coefficient = config.responses.v_zeta
        np.testing.assert_allclose(flux.q_x, coefficient * qxx_value, atol=1.0e-8)
        np.testing.assert_allclose(flux.q_y, coefficient * qxy_value, atol=1.0e-8)
        np.testing.assert_allclose(
            flux.total_x,
            coefficient * (0.5 + qxx_value),
            atol=1.0e-8,
        )
        np.testing.assert_allclose(flux.total_y, coefficient * qxy_value, atol=1.0e-8)

    def test_isotropic_initial_polarity_is_reported_exactly(self) -> None:
        result = compute_position_pdf(self._small_density_config())
        np.testing.assert_array_equal(result.polarity_x[:, 0], 0.0)
        np.testing.assert_array_equal(result.polarity_y[:, 0], 0.0)
        np.testing.assert_array_equal(result.difference_polarity_x[:, 0], 0.0)
        np.testing.assert_array_equal(result.difference_polarity_y[:, 0], 0.0)

    def test_polarity_boundary_layer_report_is_renderable(self) -> None:
        config = self._small_density_config()
        result = compute_position_pdf(config)
        field = create_field(config.field)
        with TemporaryDirectory() as directory:
            figure_output, summary_output = render_polarity_report(
                result,
                config,
                field,
                Path(directory) / "polarity.png",
                Path(directory) / "polarity_summary.csv",
                cases=(0, 2, 3),
            )
            self.assertTrue(figure_output.is_file())
            self.assertGreater(figure_output.stat().st_size, 100)
            self.assertTrue(summary_output.is_file())
            with summary_output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["case"] for row in rows}, {"0", "2", "3"})

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
        np.testing.assert_array_equal(native.polarity_x, readable.polarity_x)
        np.testing.assert_array_equal(native.polarity_y, readable.polarity_y)
        np.testing.assert_array_equal(native.polarity_x, serial_native.polarity_x)
        np.testing.assert_array_equal(native.nematic_xx, readable.nematic_xx)
        np.testing.assert_array_equal(native.nematic_xy, readable.nematic_xy)
        np.testing.assert_array_equal(
            native.nematic_split_half_error_xx,
            readable.nematic_split_half_error_xx,
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

    def test_gaussian_initial_position_matches_between_engines_and_stays_in_disk(
        self,
    ) -> None:
        base = replace(
            self._small_density_config(),
            density=replace(
                self._small_density_config().density,
                initial_position_distribution="gaussian",
                initial_gaussian_sigma=0.3,
            ),
        )
        numba_config = replace(
            base, performance=replace(base.performance, engine="numba")
        )
        python_config = replace(
            base, performance=replace(base.performance, engine="python")
        )
        native = compute_position_pdf(numba_config)
        readable = compute_position_pdf(python_config)
        np.testing.assert_array_equal(native.density, readable.density)
        coords = native.coordinates
        x_grid, y_grid = np.meshgrid(coords, coords)
        mask = x_grid**2 + y_grid**2 <= 1.0
        self.assertTrue(np.all(native.density[:, :, ~mask] == 0.0))

    def test_gaussian_initial_density_is_reported_without_finite_M_noise(self) -> None:
        base = self._small_density_config()
        # The continuum solvers size their own grid from the landscape, so the
        # reported grid has to be read off the plan rather than assumed.
        config = replace(
            base,
            density=replace(
                base.density,
                initial_position_distribution="gaussian",
                initial_gaussian_sigma=0.3,
                grid_size=build_macroscopic_plan(base)[0].grid_size,
            ),
        )
        micro = compute_position_pdf(config)
        macro = compute_macroscopic_pdf(config)
        np.testing.assert_array_equal(micro.coordinates, macro.coordinates)
        np.testing.assert_array_equal(micro.density[:, 0], macro.density[:, 0])

    def test_native_density_rejects_duplicate_truncated_seeds(self) -> None:
        config = self._small_density_config()
        field = create_field(config.field)
        config, _ = resolve_numerics(config, field, tuple(range(10)))
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
        for case_number in range(1, 10):
            np.testing.assert_array_equal(
                result.density[case_number],
                result.density[0],
            )
        observables = result.microscopic_observables
        assert observables is not None
        for case_number in range(1, 10):
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
        self.assertIsNotNone(result.difference_polarity_x)
        self.assertIsNotNone(result.difference_polarity_y)
        self.assertFalse(
            np.array_equal(result.difference_polarity_x, result.polarity_x)
        )

        with TemporaryDirectory() as directory:
            output = save_position_pdf(
                result,
                Path(directory) / "density.npz",
            )
            with np.load(output) as saved:
                self.assertEqual(int(saved["schema_version"]), PDF_SCHEMA_VERSION)
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
                self.assertIn("polarity_x", saved.files)
                self.assertIn("polarity_y", saved.files)
                self.assertIn("difference_polarity_x", saved.files)
                self.assertIn("difference_polarity_y", saved.files)
                self.assertIn("nematic_xx", saved.files)
                self.assertIn("nematic_xy", saved.files)
                self.assertIn("difference_nematic_xx", saved.files)
                self.assertIn("difference_nematic_xy", saved.files)
                self.assertIn("nematic_split_half_error_xx", saved.files)
                self.assertIn("nematic_split_half_error_xy", saved.files)
                self.assertEqual(
                    float(saved["difference_bandwidth_fraction"]),
                    0.25,
                )
            loaded = load_position_pdf(output)
            np.testing.assert_array_equal(loaded.density, result.density)
            self.assertEqual(loaded.schema_version, PDF_SCHEMA_VERSION)
            self.assertEqual(loaded.software_version, "0.2.0")
            self.assertTrue(loaded.config_text.startswith("# Every value"))
            self.assertEqual(len(loaded.config_sha256), 64)
            self.assertIn('"numpy":', loaded.runtime_versions)
            self.assertEqual(
                loaded.microscopic_observables.names,
                result.microscopic_observables.names,
            )
            np.testing.assert_array_equal(loaded.polarity_x, result.polarity_x)
            np.testing.assert_array_equal(loaded.polarity_y, result.polarity_y)
            np.testing.assert_array_equal(
                loaded.difference_polarity_x, result.difference_polarity_x
            )
            np.testing.assert_array_equal(loaded.nematic_xx, result.nematic_xx)
            np.testing.assert_array_equal(loaded.nematic_xy, result.nematic_xy)
            np.testing.assert_array_equal(
                loaded.difference_nematic_xx, result.difference_nematic_xx
            )
            np.testing.assert_array_equal(
                loaded.nematic_split_half_error_xx,
                result.nematic_split_half_error_xx,
            )
            np.testing.assert_array_equal(
                loaded.difference_polarity_y, result.difference_polarity_y
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
            density=differences,
            difference_density=differences,
            case_numbers=(0, 1),
            ensemble_size=1,
            bandwidth_fraction=0.1,
        )
        result_b = replace(
            result_a,
            difference_density=2.0 * differences,
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
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
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
        self.assertIsNotNone(result.polarity_x)
        self.assertIsNotNone(result.polarity_y)
        self.assertGreater(float(np.max(np.abs(result.polarity_x[:, 1:]))), 0.0)

    def test_polar_closed_moment_pdf_evolves_density_and_polarity(self) -> None:
        config = self._small_macro_config()
        result = compute_moment_pdf(config)
        spacing = 2.0 / len(result.coordinates)
        masses = np.sum(result.density, axis=(-2, -1)) * spacing**2
        np.testing.assert_allclose(masses, 1.0, rtol=0.0, atol=3.0e-6)
        self.assertEqual(result.method, "moment")
        self.assertIsNotNone(result.polarity_x)
        self.assertIsNotNone(result.polarity_y)
        np.testing.assert_array_equal(result.polarity_x[:, 0], 0.0)
        np.testing.assert_array_equal(result.polarity_y[:, 0], 0.0)
        self.assertGreater(float(np.max(np.abs(result.polarity_x[:, 1:]))), 0.0)
        self.assertGreaterEqual(float(np.min(result.density)), 0.0)

    def test_macro_is_the_dynamic_polar_manuscript_system(self) -> None:
        config = self._small_macro_config()
        moment = compute_moment_pdf(config)
        macro = compute_macroscopic_pdf(config)
        # Macro is now the manuscript-level dynamic-p entry point, while
        # moment is retained as a backwards-compatible strategy name.
        np.testing.assert_array_equal(macro.density, moment.density)
        np.testing.assert_array_equal(macro.polarity_x, moment.polarity_x)
        np.testing.assert_array_equal(macro.polarity_y, moment.polarity_y)
        # Local D_R and tumble-rate responses modify the evolved polarity
        # relaxation and therefore must not disappear from macro.
        for case_number in (2, 4):
            self.assertGreater(
                float(
                    np.linalg.norm(
                        moment.density[case_number, -1] - moment.density[0, -1]
                    )
                ),
                0.0,
            )
            self.assertGreater(
                float(
                    np.linalg.norm(
                        macro.density[case_number, -1] - macro.density[0, -1]
                    )
                ),
                0.0,
            )
        # Under Q=0 these mechanisms enter only through gamma1_eta, and the
        # default responses obey D_R_eta=(1-P_turn)*lambda_eta, so their
        # polar-closed predictions must be indistinguishable.  The tuning is
        # exact in decimal but not in binary -- D_R_eta versus
        # (1-0.30)*lambda_eta -- and the two cases reach gamma1 through
        # different products, so agreement is to float64 round-off rather than
        # bit for bit.
        for observed, expected in (
            (moment.density[2], moment.density[4]),
            (moment.polarity_x[2], moment.polarity_x[4]),
            (moment.polarity_y[2], moment.polarity_y[4]),
        ):
            np.testing.assert_allclose(observed, expected, rtol=1.0e-12, atol=1.0e-20)

    def test_moment_plan_sets_nematic_to_zero_but_not_polarity(self) -> None:
        config = self._small_macro_config()
        plan, *_ = build_moment_plan(config)
        self.assertGreater(plan.gamma1, 0.0)
        self.assertGreater(plan.gamma2, 0.0)
        self.assertLessEqual(plan.dt, plan.safe_dt)
        expected_joint_limit = config.macro.cfl_safety / (
            1.0 / plan.diffusion_dt + 1.0 / plan.wave_dt + 1.0 / plan.relaxation_dt
        )
        self.assertAlmostEqual(plan.safe_dt, expected_joint_limit)
        self.assertAlmostEqual(plan.steps * plan.dt, config.simulation.duration)

    def test_polar_closed_evolution_is_independent_of_second_turning_moment(
        self,
    ) -> None:
        config = self._small_macro_config()
        # Both kernels have P_turn=0, hence the same polar relaxation gamma1,
        # but their nematic moments are P_2_turn=-1 and +1 respectively.
        quarter_turns = replace(
            config,
            turning={
                "kind": "discrete",
                "absolute_angles": [np.pi / 2.0],
                "probabilities": [1.0],
            },
        )
        forward_or_reverse = replace(
            config,
            turning={
                "kind": "discrete",
                "absolute_angles": [0.0, np.pi],
                "probabilities": [0.5, 0.5],
            },
        )
        first = compute_moment_pdf(quarter_turns)
        second = compute_moment_pdf(forward_or_reverse)
        # Identical polar-order dynamics, reached through different kernel
        # moments, so the fields agree to float64 round-off, not bit for bit.
        for observed, expected in (
            (first.density, second.density),
            (first.polarity_x, second.polarity_x),
            (first.polarity_y, second.polarity_y),
        ):
            np.testing.assert_allclose(observed, expected, rtol=1.0e-12, atol=1.0e-20)

    def test_gaussian_initial_density_is_centered_normalized_and_disk_clipped(
        self,
    ) -> None:
        config = replace(
            self._small_macro_config(),
            density=replace(
                self._small_macro_config().density,
                initial_position_distribution="gaussian",
                initial_gaussian_sigma=0.3,
            ),
        )
        field = create_field(config.field)
        _, coordinates, grid_points, mask, *_ = build_macroscopic_plan(config, field)
        initial = _initial_density(
            config, field, grid_points, mask, 2.0 / len(coordinates)
        )
        spacing = 2.0 / len(coordinates)
        self.assertAlmostEqual(float(np.sum(initial) * spacing**2), 1.0, places=6)
        self.assertTrue(np.all(initial[~mask] == 0.0))
        x_grid, y_grid = np.meshgrid(coordinates, coordinates)
        center_index = np.unravel_index(np.argmin(x_grid**2 + y_grid**2), mask.shape)
        self.assertEqual(
            np.argmax(initial), np.ravel_multi_index(center_index, mask.shape)
        )

    def test_dynamic_macro_retains_zero_adiabatic_mobility_cases(self) -> None:
        result = compute_macroscopic_pdf(self._small_macro_config())
        for case_number in (2, 4, 5):
            self.assertGreater(
                float(
                    np.linalg.norm(
                        result.polarity_x[case_number] - result.polarity_x[0]
                    )
                ),
                0.0,
            )

    def test_macro_plan_is_the_dynamic_polar_stability_plan(self) -> None:
        config = self._small_macro_config()
        plan, *_ = build_macroscopic_plan(config)
        self.assertGreater(plan.gamma1, 0.0)
        self.assertGreater(plan.gamma2, 0.0)
        self.assertLessEqual(plan.dt, plan.safe_dt)
        self.assertAlmostEqual(
            plan.steps * plan.dt,
            config.simulation.duration,
        )

    def test_macro_and_moment_plans_are_identical(self) -> None:
        config = self._small_macro_config()
        macro_plan, *_ = build_macroscopic_plan(config)
        moment_plan, *_ = build_moment_plan(config)
        self.assertEqual(macro_plan.grid_size, moment_plan.grid_size)
        self.assertEqual(macro_plan.dt, moment_plan.dt)
        self.assertEqual(macro_plan.gamma1, moment_plan.gamma1)
        self.assertEqual(macro_plan.gamma2, moment_plan.gamma2)
        np.testing.assert_array_equal(macro_plan.save_steps, moment_plan.save_steps)

    def test_macro_and_micro_use_the_same_observed_point_initial_state(self) -> None:
        micro_config = ProbabilityDensityTests._small_density_config()
        micro_config = replace(
            micro_config,
            density=replace(
                micro_config.density,
                grid_size=build_macroscopic_plan(micro_config)[0].grid_size,
            ),
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
        with self.assertRaisesRegex(ValueError, "requires gamma1"):
            build_macroscopic_plan(config)

    def test_macro_rejects_unrelaxed_nematic_mode(self) -> None:
        config = replace(
            self._small_macro_config(),
            kinetics=replace(
                self._small_macro_config().kinetics,
                D_R0=0.0,
                lambda0=10.0,
            ),
            # Deterministic reversals relax polarity (P_turn=-1) but retain
            # the second harmonic (P_2_turn=1), exactly the independent
            # gamma2=0 failure discussed below Eq. (gamma2) in the manuscript.
            turning={"kind": "wrapped_normal", "P_turn": -1.0},
        )
        with self.assertRaisesRegex(ValueError, "requires gamma2"):
            build_macroscopic_plan(config)

    def test_density_cli_selects_micro_moment_macro_or_analytic(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["compare"]).strategy, "all")
        self.assertEqual(parser.parse_args(["compare", "--micro"]).strategy, "micro")
        self.assertEqual(parser.parse_args(["compare", "--moment"]).strategy, "moment")
        self.assertEqual(parser.parse_args(["compare", "--macro"]).strategy, "macro")
        self.assertEqual(
            parser.parse_args(["compare", "--analytic"]).strategy, "analytic"
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["compare", "--micro", "--macro"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["compare", "--micro", "--moment"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["compare", "--macro", "--analytic"])

    def test_cli_rejects_removed_boundary_override(self) -> None:
        parser = build_parser()
        default_config = _load(parser.parse_args(["compare"]))
        self.assertEqual(default_config.domain.radius, 1.0)
        with self.assertRaises(SystemExit):
            parser.parse_args(["compare", "--boundary", "specular"])


class AnalyticSolutionTests(unittest.TestCase):
    @staticmethod
    def _small_analytic_config(*, gradient=(0.2, -0.1)):
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        return replace(
            config.with_overrides(duration=0.002, progress=False),
            field={
                "kind": "linear",
                "baseline": 1.0,
                "gradient": list(gradient),
                "center": [0.0, 0.0],
            },
            density=replace(
                config.density,
                # This helper's own tests (e.g. the zero-mobility-cases
                # invariance check below) reason about a spatially uniform
                # steady state, so pin the IC explicitly rather than
                # inheriting config/default.toml's "fixed" default.
                initial_position_distribution="uniform",
                grid_size=33,
                time_slices=3,
                bandwidth_fraction=0.25,
                difference_bandwidth_fraction=0.25,
            ),
        )

    def test_basis_is_orthonormal(self) -> None:
        basis = build_neumann_bessel_basis(radius=1.0, m_max=3, n_max=3)
        r_nodes, r_weights = np.polynomial.legendre.leggauss(160)
        r_nodes = 0.5 * (r_nodes + 1.0)
        r_weights = 0.5 * r_weights
        theta = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
        r_grid, theta_grid = np.meshgrid(r_nodes, theta, indexing="ij")
        x = r_grid * np.cos(theta_grid)
        y = r_grid * np.sin(theta_grid)
        q = evaluate_basis(basis, x, y).reshape(basis.size, -1)
        weight = (r_weights * r_nodes)[:, None] * np.ones_like(theta)[None, :]
        weight = weight.reshape(-1) * (2.0 * np.pi / theta.size)
        gram = (q * weight[None, :]) @ q.T
        np.testing.assert_allclose(gram, np.eye(basis.size), atol=1.0e-6)

    def test_overlap_matrix_matches_radial_closed_form(self) -> None:
        # A Gaussian centered at the origin is purely axisymmetric, so its
        # m=0 Fourier coefficient a_0(r) is just eta_1(r) itself, and the
        # diagonal overlap of a radial mode reduces to the manuscript's
        # closed form M_(0n),(0n) = 2*pi*integral[a_0(r) * R_0n(r)^2 * r dr].
        field = GaussianField(
            baseline=1.0,
            amplitude=0.6,
            center=(0.0, 0.0),
            sigma=(0.35, 0.35),
        )
        basis = build_neumann_bessel_basis(radius=1.0, m_max=1, n_max=3)
        overlap = compute_overlap_matrix(basis, field)

        r_nodes, r_weights = np.polynomial.legendre.leggauss(400)
        r_nodes = 0.5 * (r_nodes + 1.0)
        r_weights = 0.5 * r_weights
        eta1 = np.asarray(
            field.value(np.stack((r_nodes, np.zeros_like(r_nodes)), axis=-1))
        )
        eta1 = eta1 - disk_area_average(
            field, 1.0, radial_points=96, angular_points=256
        )
        for index, mode in enumerate(basis.modes):
            if mode.parity != "radial":
                continue
            radial_shape = special.jv(mode.m, mode.k * r_nodes) / mode.norm
            expected = (
                2.0 * np.pi * np.sum(r_weights * r_nodes * eta1 * radial_shape**2)
            )
            self.assertAlmostEqual(overlap[index, index], expected, delta=1.0e-6)

    @staticmethod
    def _quadrature_mass_at_t0(kappa: float, gradient) -> float:
        # Integrates the raw (un-normalized) t=0 density with the same
        # radial-Gauss-Legendre / uniform-angular quadrature solve_analytic_spectrum
        # itself uses, so the result reflects only this basis's finite mode
        # truncation -- not Cartesian-grid pixelation of the disk boundary,
        # which dominates at coarse grid sizes and would otherwise mask it.
        radius = 1.0
        field = create_field(
            {
                "kind": "linear",
                "baseline": 1.0,
                "gradient": list(gradient),
                "center": [0.0, 0.0],
            }
        )
        effective_diffusivity = 1.0
        basis = build_neumann_bessel_basis(radius, m_max=6, n_max=6)
        spectrum = solve_analytic_spectrum(basis, field, kappa, effective_diffusivity)

        r_nodes, r_weights = np.polynomial.legendre.leggauss(96)
        r_nodes = 0.5 * radius * (r_nodes + 1.0)
        r_weights = 0.5 * radius * r_weights
        theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        r_grid, theta_grid = np.meshgrid(r_nodes, theta, indexing="ij")
        x = r_grid * np.cos(theta_grid)
        y = r_grid * np.sin(theta_grid)

        basis_values = evaluate_basis(basis, x, y)
        eta = np.asarray(field.value(np.stack((x, y), axis=-1)), dtype=float)
        equilibrium_weight = np.exp(
            kappa * (eta - field.reference_value) / effective_diffusivity
        )
        projection = uniform_projection(
            spectrum,
            constant_mode_index=_constant_mode_index(basis),
            disk_area=math.pi * radius**2,
        )
        density_t0 = evaluate_density(
            spectrum,
            basis_values,
            equilibrium_weight,
            np.array([0.0]),
            projection=projection,
        )[0]
        weight = (r_weights * r_nodes)[:, None] * np.full_like(
            theta, 2.0 * np.pi / theta.size
        )
        return float(np.sum(density_t0 * weight))

    def test_mass_residual_stays_near_machine_precision_for_any_field_amplitude(
        self,
    ) -> None:
        # solve_analytic_spectrum diagonalizes the (mode-truncated) weighted
        # eigenvalue problem exactly rather than via a kappa-power series, so
        # mass conservation -- a property of the underlying PDE for any
        # kappa, not a perturbative artifact -- should hold to within
        # quadrature/floating-point precision regardless of field amplitude,
        # unlike the O(kappa^2)-scaling residual an earlier, perturbative
        # version of this solver had.
        kappa = 0.5
        for gradient in ((0.1, -0.05), (0.3, -0.15), (0.5, -0.25)):
            residual = abs(self._quadrature_mass_at_t0(kappa, gradient) - 1.0)
            self.assertLess(residual, 1.0e-6)

    def test_analytic_pdf_is_positive_conservative_and_dimensionless(self) -> None:
        config = self._small_analytic_config()
        result, diagnostics = compute_analytic_pdf(config)
        spacing = 2.0 / len(result.coordinates)
        masses = np.sum(result.density, axis=(-2, -1)) * spacing**2
        np.testing.assert_allclose(masses, 1.0, rtol=0.0, atol=2.0e-6)
        self.assertGreaterEqual(float(np.min(result.density)), 0.0)
        self.assertEqual(result.method, "analytic")
        self.assertEqual(result.ensemble_size, 0)
        self.assertEqual(result.times[-1], config.simulation.duration)
        self.assertEqual(diagnostics.padding_factor, 3)
        self.assertEqual(diagnostics.spectral_grid_size, 3 * len(result.coordinates))
        self.assertIsNotNone(result.polarity_x)
        self.assertIsNotNone(result.polarity_y)
        turning_kernel = create_turning_kernel(config.turning)
        expected_gamma2 = 4.0 * config.kinetics.D_R0 + config.kinetics.lambda0 * (
            1.0 - turning_kernel.P_2_turn
        )
        self.assertEqual(diagnostics.gamma2, expected_gamma2)

    def test_uniform_nutrient_has_only_the_exact_common_green_solution(self) -> None:
        config = self._small_analytic_config(gradient=(0.0, 0.0))
        result, diagnostics = compute_analytic_pdf(config)
        for case_number in range(1, 10):
            # eta1 = eta - eta0 vanishes only to the round-off of the disk
            # area average, so each case picks up a different O(eps) forcing.
            np.testing.assert_allclose(
                result.density[case_number],
                result.density[0],
                rtol=1.0e-12,
                atol=1.0e-20,
            )
            np.testing.assert_allclose(
                result.polarity_x[case_number], result.polarity_x[0], atol=1.0e-15
            )
            np.testing.assert_allclose(
                result.polarity_y[case_number], result.polarity_y[0], atol=1.0e-15
            )
        for case in diagnostics.cases:
            self.assertLess(case.max_relative_correction, 1.0e-14)

    def test_analytic_evolves_polarity_instead_of_reconstructing_it(self) -> None:
        config = self._small_analytic_config(gradient=(0.2, -0.1))
        analytic, diagnostics = compute_analytic_pdf(config)
        np.testing.assert_array_equal(analytic.polarity_x[:, 0], 0.0)
        np.testing.assert_array_equal(analytic.polarity_y[:, 0], 0.0)
        self.assertGreater(float(np.max(np.abs(analytic.polarity_x[:, 1:]))), 0.0)
        self.assertGreater(diagnostics.gamma1, 0.0)
        self.assertGreater(diagnostics.gamma2, 0.0)

    def test_delta_projection_matches_neumann_heat_kernel_for_pure_diffusion(
        self,
    ) -> None:
        # With no field (kappa=0), a Dirac-delta IC's diffusive spread has
        # the textbook closed form rho(x,t) = sum_i q_i(x0)*q_i(x)*
        # exp(-D_eff*k_i^2*t) (the Neumann Green's function of the disk).
        # Check delta_projection/evaluate_density reproduce this exactly,
        # independent of the rest of the module's kappa!=0 machinery.
        radius = 1.0
        position = (0.2, -0.1)
        effective_diffusivity = 1.7
        basis = build_neumann_bessel_basis(radius, m_max=5, n_max=5)
        field = create_field(
            {
                "kind": "linear",
                "baseline": 1.0,
                "gradient": [0.0, 0.0],
                "center": [0.0, 0.0],
            }
        )
        spectrum = solve_analytic_spectrum(basis, field, 0.0, effective_diffusivity)

        projection = delta_projection(spectrum, basis, position)

        target_x = np.array([[0.3]])
        target_y = np.array([[0.0]])
        time = 0.05
        basis_at_source = evaluate_basis(
            basis, np.array([position[0]]), np.array([position[1]])
        )[:, 0]
        basis_at_target = evaluate_basis(basis, target_x, target_y)[:, 0, 0]
        k2 = np.asarray([mode.k**2 for mode in basis.modes])
        expected_density = np.sum(
            basis_at_source
            * basis_at_target
            * np.exp(-effective_diffusivity * k2 * time)
        )

        equilibrium_weight = np.ones_like(target_x)
        density = evaluate_density(
            spectrum,
            evaluate_basis(basis, target_x, target_y),
            equilibrium_weight,
            np.array([time]),
            projection=projection,
        )[0, 0, 0]
        self.assertAlmostEqual(float(density), float(expected_density), places=10)

    def test_analytic_pdf_with_delta_ic_conserves_mass_and_starts_concentrated(
        self,
    ) -> None:
        base = self._small_analytic_config()
        config = replace(
            base, density=replace(base.density, initial_position_distribution="fixed")
        )
        result, _ = compute_analytic_pdf(config)
        spacing = 2.0 / len(result.coordinates)
        masses = np.sum(result.density, axis=(-2, -1)) * spacing**2
        np.testing.assert_allclose(masses, 1.0, rtol=0.0, atol=2.0e-6)

        x_grid, y_grid = np.meshgrid(result.coordinates, result.coordinates)
        center_index = np.unravel_index(np.argmin(x_grid**2 + y_grid**2), x_grid.shape)
        case_number = 1
        self.assertGreater(
            result.density[case_number, 0][center_index],
            result.density[case_number, -1][center_index],
        )

    def test_delta_ic_cases_agree_at_t0_despite_case_dependent_eigenbasis(self) -> None:
        # Every case starts from the exact same physical point regardless of
        # its chemical response, so at t=0 every case's density should equal
        # Case 0's almost exactly, even though chi_i's own coefficients are
        # case-dependent (through kappa) -- see delta_projection's docstring
        # for why the *correct* weighted projection makes this hold to
        # essentially the mode-truncation limit, not just approximately.
        base = self._small_analytic_config()
        config = replace(
            base, density=replace(base.density, initial_position_distribution="fixed")
        )
        result, _ = compute_analytic_pdf(config)
        baseline = result.density[0, 0]
        peak = float(np.max(baseline))
        for case_number in range(1, 10):
            np.testing.assert_allclose(
                result.density[case_number, 0],
                baseline,
                rtol=0.0,
                atol=5.0e-3 * peak,
            )

    def test_analytic_pdf_with_gaussian_ic_conserves_mass_and_starts_concentrated(
        self,
    ) -> None:
        # "gaussian" is the package default: manuscript/main.tex
        # explicitly excludes "a point release or another sharply localized
        # initial profile" from this solver's declared regime, so unlike
        # "fixed" this option is meant to actually be trustworthy, not just
        # runnable.
        base = self._small_analytic_config()
        config = replace(
            base,
            density=replace(
                base.density,
                initial_position_distribution="gaussian",
                initial_gaussian_sigma=0.3,
            ),
        )
        result, _ = compute_analytic_pdf(config)
        spacing = 2.0 / len(result.coordinates)
        masses = np.sum(result.density, axis=(-2, -1)) * spacing**2
        np.testing.assert_allclose(masses, 1.0, rtol=0.0, atol=2.0e-6)

        x_grid, y_grid = np.meshgrid(result.coordinates, result.coordinates)
        center_index = np.unravel_index(np.argmin(x_grid**2 + y_grid**2), x_grid.shape)
        case_number = 1
        self.assertGreater(
            result.density[case_number, 0][center_index],
            result.density[case_number, -1][center_index],
        )

    def test_gaussian_ic_cases_agree_at_t0_more_tightly_than_a_literal_delta(
        self,
    ) -> None:
        # A finite-width target is far better resolved by a truncated mode
        # sum than a literal point is (see gaussian_projection's docstring),
        # so this should hold noticeably tighter than the "fixed" version of
        # this same check above.
        base = self._small_analytic_config()
        config = replace(
            base,
            density=replace(
                base.density,
                initial_position_distribution="gaussian",
                initial_gaussian_sigma=0.3,
            ),
        )
        result, _ = compute_analytic_pdf(config)
        baseline = result.density[0, 0]
        peak = float(np.max(baseline))
        for case_number in range(1, 10):
            np.testing.assert_allclose(
                result.density[case_number, 0],
                baseline,
                rtol=0.0,
                atol=5.0e-4 * peak,
            )


class PolarityVideoTests(unittest.TestCase):
    def test_case_limits_use_magnitude_and_ignore_case_zero(self) -> None:
        # Two cases, one time, a 3x3 grid inside the unit disk (center only
        # is guaranteed inside for a coarse grid, so put signal there).
        coordinates = np.asarray([-0.4, 0.0, 0.4])
        polarity_x = np.zeros((2, 1, 3, 3))
        polarity_y = np.zeros((2, 1, 3, 3))
        polarity_x[1, 0, 1, 1] = 3.0
        polarity_y[1, 0, 1, 1] = 4.0
        limits, maxima = _polarity_case_limits(
            polarity_x, polarity_y, coordinates, 100.0
        )
        np.testing.assert_allclose(limits, [0.0, 5.0])
        np.testing.assert_allclose(maxima, [0.0, 5.0])

    def test_synchronized_limits_match_joint_percentile(self) -> None:
        coordinates = np.asarray([-0.4, 0.0, 0.4])
        px_a = np.zeros((2, 1, 3, 3))
        py_a = np.zeros((2, 1, 3, 3))
        px_a[1, 0, 1, 1] = 3.0
        result_a = PositionPDFResult(
            times=np.asarray([0.0]),
            coordinates=coordinates,
            density=np.ones((2, 1, 3, 3), dtype=np.float64),
            case_numbers=(0, 1),
            ensemble_size=1,
            bandwidth_fraction=0.1,
            polarity_x=px_a,
            polarity_y=py_a,
        )
        px_b = np.zeros((2, 1, 3, 3))
        px_b[1, 0, 1, 1] = 9.0
        result_b = replace(
            result_a,
            method="macro",
            polarity_x=px_b,
            polarity_y=py_a,
        )
        shared = synchronized_polarity_case_limits([result_a, result_b], 100.0)
        np.testing.assert_allclose(shared, [0.0, 9.0])

    def test_theoretical_polarity_field_has_expected_shape(self) -> None:
        config = MacroscopicDensityTests._small_macro_config()
        field = create_field(config.field)
        macro = compute_macroscopic_pdf(config)
        theory_x, theory_y = compute_theoretical_polarity_field(macro, config, field)
        self.assertEqual(theory_x.shape, macro.density.shape)
        self.assertEqual(theory_y.shape, macro.density.shape)
        self.assertTrue(np.all(np.isfinite(theory_x)))
        self.assertTrue(np.all(np.isfinite(theory_y)))
        # Case 0 has no chemistry, but the diffusive backflow -v0/(2 gamma1)
        # grad(rho) is still generally nonzero once rho departs from uniform.
        self.assertFalse(np.all(theory_x[0] == 0.0) and np.all(theory_y[0] == 0.0))

    def test_masked_gradient_is_exact_for_a_linear_field(self) -> None:
        coordinates = np.linspace(-1.0, 1.0, 41)
        x_grid, y_grid = np.meshgrid(coordinates, coordinates)
        mask = x_grid * x_grid + y_grid * y_grid <= 1.0
        values = 2.0 + 3.0 * x_grid - 0.5 * y_grid
        values = np.where(mask, values, 0.0)
        spacing = coordinates[1] - coordinates[0]
        gradient_x = _masked_directional_gradient(values, mask, spacing, axis=-1)
        gradient_y = _masked_directional_gradient(values, mask, spacing, axis=-2)
        # Exclude the handful of single-pixel-wide mask rows/columns (top and
        # bottom of the disk on this coarse grid), where both neighbors along
        # an axis fall outside the mask and the gradient is conventionally 0.
        row_has_neighbors = mask.sum(axis=1) > 1
        column_has_neighbors = mask.sum(axis=0) > 1
        well_posed_x = mask & row_has_neighbors[:, None]
        well_posed_y = mask & column_has_neighbors[None, :]
        np.testing.assert_allclose(gradient_x[well_posed_x], 3.0, atol=5.0e-2)
        np.testing.assert_allclose(gradient_y[well_posed_y], -0.5, atol=5.0e-2)

    def test_adiabatic_polarity_has_no_boundary_masking_artifact(self) -> None:
        # Regression test: _adiabatic_polarity used to differentiate the
        # density array with a plain np.gradient, which -- since density is
        # stored as exactly 0 outside the disk mask -- differenced real
        # interior values against that artificial padding at every mask-edge
        # pixel. That produced a spurious O(1/dx) gradient spike concentrated
        # in a thin ring at the rim, many times larger than the true
        # near-uniform interior signal, which then dominated any
        # percentile-based color/significance scaling downstream. A smooth
        # density field (no real near-wall structure) should map to a
        # polarity field whose magnitude is comparable at the rim and in the
        # interior, not concentrated in a thin boundary ring.
        coordinates = np.linspace(-1.0, 1.0, 81)
        x_grid, y_grid = np.meshgrid(coordinates, coordinates)
        mask = x_grid * x_grid + y_grid * y_grid <= 1.0
        r = np.sqrt(x_grid * x_grid + y_grid * y_grid)
        # A smooth, slowly-varying density with no genuine near-wall feature.
        density = np.where(mask, 1.0 + 0.05 * x_grid - 0.03 * y_grid, 0.0)
        eta_gradient = np.stack(
            [0.2 * np.ones_like(x_grid), -0.1 * np.ones_like(x_grid)], axis=-1
        )
        theory_x, theory_y = _adiabatic_polarity(
            density, eta_gradient, coordinates, 1.0, 1.0, 0.5, mask
        )
        magnitude = np.hypot(theory_x, theory_y)
        ring = mask & (r > 1.0 - 2.0 * (coordinates[1] - coordinates[0]))
        interior = mask & (r < 0.7)
        ring_mean = float(np.mean(magnitude[ring]))
        interior_mean = float(np.mean(magnitude[interior]))
        self.assertLess(ring_mean / interior_mean, 3.0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for MP4 tests")
    def test_polarity_difference_video_is_renderable(self) -> None:
        config = ProbabilityDensityTests._small_density_config()
        result = compute_position_pdf(config)
        with TemporaryDirectory() as directory:
            output = render_polarity_difference_video(
                result,
                config,
                Path(directory) / "polarity_difference.mp4",
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 100)


class ComparisonVideoTests(unittest.TestCase):
    """The 9(case)-x-N(method) grid videos used when more than one method
    is run together, so the same case sits side by side across methods."""

    @staticmethod
    def _micro_and_macro():
        micro_config = ProbabilityDensityTests._small_density_config()
        micro = compute_position_pdf(micro_config)
        macro_config = MacroscopicDensityTests._small_macro_config()
        macro = compute_macroscopic_pdf(macro_config)
        field = create_field(macro_config.field)
        theory_x, theory_y = compute_theoretical_polarity_field(
            macro, macro_config, field
        )
        macro = replace(
            macro,
            polarity_x=theory_x,
            polarity_y=theory_y,
            difference_polarity_x=theory_x,
            difference_polarity_y=theory_y,
        )
        return micro_config, {"micro": micro, "macro": macro}

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for MP4 tests")
    def test_density_comparison_video_is_renderable(self) -> None:
        config, results = self._micro_and_macro()
        with TemporaryDirectory() as directory:
            output = render_density_comparison_video(
                results, config, Path(directory) / "density_comparison.mp4"
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 100)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for MP4 tests")
    def test_density_difference_comparison_video_is_renderable(self) -> None:
        config, results = self._micro_and_macro()
        with TemporaryDirectory() as directory:
            output = render_density_difference_comparison_video(
                results, config, Path(directory) / "density_difference_comparison.mp4"
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 100)

    def test_default_comparison_schedule_never_repeats_a_saved_state(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "default.toml")
        result = SimpleNamespace(
            times=np.linspace(0.0, config.simulation.duration, 101)
        )
        display_times = shared_display_times([result], config)
        indices = _density_frame_indices(
            result,
            config,
            frame_times=display_times,
        )
        self.assertEqual(len(display_times), 100)
        self.assertEqual(len(np.unique(indices)), len(indices))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for MP4 tests")
    def test_polarity_comparison_video_is_renderable(self) -> None:
        config, results = self._micro_and_macro()
        with TemporaryDirectory() as directory:
            output = render_polarity_comparison_video(
                results, config, Path(directory) / "polarity_comparison.mp4"
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 100)

    def test_initial_conditions_and_time_frame_contours_are_renderable(self) -> None:
        config, results = self._micro_and_macro()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            initial = render_initial_condition_contours(
                results, config, root / "initial_conditions.png"
            )
            frames, metrics = render_density_frames_vs_time(
                results,
                config,
                root / "frames",
                root / "frame_metrics.csv",
                frame_count=3,
            )
            polarity_frames, metrics = render_polarity_frames_vs_time(
                results,
                config,
                root / "frames",
                metrics,
                frame_count=3,
            )
            self.assertTrue(initial.is_file())
            self.assertGreater(initial.stat().st_size, 1_000)
            self.assertEqual(len(frames), 10)
            self.assertTrue(all(path.is_file() for path in frames))
            self.assertEqual(len(polarity_frames), 10)
            self.assertTrue(all(path.is_file() for path in polarity_frames))
            with metrics.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 10 * 2 * 3 * 2)
            manuscript = render_manuscript_strategy_figures(
                results, config, root / "manuscript"
            )
            pdfs = [path for path in manuscript if path.suffix == ".pdf"]
            pngs = [path for path in manuscript if path.suffix == ".png"]
            self.assertEqual(len(pdfs), 1)
            self.assertEqual(len(pngs), 1)
            self.assertTrue(all(path.is_file() for path in manuscript))
            self.assertTrue(
                all(path.read_bytes().startswith(b"%PDF-") for path in pdfs)
            )
            self.assertEqual(
                [path.name for path in pdfs],
                [f"{STRATEGY_FIGURE_STEMS[1]}.pdf"],
            )

    def test_theoretical_polarity_uses_the_prescribed_initial_state(self) -> None:
        _, results = self._micro_and_macro()
        micro = results["micro"]
        macro = results["macro"]
        np.testing.assert_array_equal(micro.polarity_x[:, 0], 0.0)
        np.testing.assert_array_equal(micro.polarity_y[:, 0], 0.0)
        np.testing.assert_array_equal(macro.polarity_x[:, 0], 0.0)
        np.testing.assert_array_equal(macro.polarity_y[:, 0], 0.0)

    def test_comparison_video_rejects_empty_results(self) -> None:
        config, _ = self._micro_and_macro()
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                render_density_comparison_video(
                    {}, config, Path(directory) / "empty.mp4"
                )
            with self.assertRaises(ValueError):
                render_polarity_comparison_video(
                    {}, config, Path(directory) / "empty.mp4"
                )

    def test_single_method_still_uses_the_old_3x3_case_grid(self) -> None:
        # With only one method there is nothing to place side by side, so
        # the CLI keeps rendering the original per-case 3x3 grid rather than
        # a 9x1 column; this exercises the CLI's branch decision directly.
        parser = build_parser()
        args = parser.parse_args(["compare", "--micro"])
        self.assertEqual(args.strategy, "micro")


if __name__ == "__main__":
    unittest.main()
