from dataclasses import replace

import numpy as np

from model.solvers.chem_swimmer.config import load_config
from model.solvers.chem_swimmer.density import PositionPDFResult, save_position_pdf
from model.solvers.chem_swimmer.result_cache import (
    load_cached_results,
    simulation_fingerprint,
    write_cache_manifest,
)


def test_fingerprint_ignores_style_but_tracks_physics():
    config = load_config("model/parameters/default.toml")
    styled = replace(config, video=replace(config.video, dpi=config.video.dpi + 10))
    changed = replace(
        config,
        simulation=replace(config.simulation, duration=config.simulation.duration * 2),
    )
    assert simulation_fingerprint(styled) == simulation_fingerprint(config)
    assert simulation_fingerprint(changed) != simulation_fingerprint(config)


def test_manifest_reuses_only_matching_archives(tmp_path):
    config = load_config("model/parameters/default.toml")
    density = np.ones((1, 2, 3, 3))
    result = PositionPDFResult(
        times=np.array([0.0, config.simulation.duration]),
        coordinates=np.linspace(-1, 1, 3),
        density=density,
        case_numbers=(0,),
        ensemble_size=8,
        bandwidth_fraction=0.1,
        method="micro",
    )
    archive = save_position_pdf(result, tmp_path / "micro.npz")
    write_cache_manifest(tmp_path, config, {"micro": archive})
    assert load_cached_results(tmp_path, config, ("micro",)) is not None

    changed = replace(
        config,
        simulation=replace(config.simulation, seed=config.simulation.seed + 1),
    )
    assert load_cached_results(tmp_path, changed, ("micro",)) is None
