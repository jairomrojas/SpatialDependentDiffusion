"""Validated reuse of expensive density archives."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ._version import PDF_SCHEMA_VERSION
from .config import ProjectConfig
from .density import PositionPDFResult, load_position_pdf

CACHE_MANIFEST = "density-cache-manifest.json"


def _numerical_source_fingerprint() -> str:
    package_root = Path(__file__).resolve().parent
    ignored = {"cli.py", "provenance.py", "result_cache.py"}
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        if "visualization" in path.parts or path.name in ignored:
            continue
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def simulation_fingerprint(config: ProjectConfig, ensemble: int | None = None) -> str:
    """Hash numerical inputs while deliberately ignoring presentation settings."""

    inputs: dict[str, Any] = asdict(config)
    inputs.pop("source_path", None)
    inputs.pop("video", None)
    inputs.pop("performance", None)
    density = inputs["density"]
    for key in (
        "video_fps",
        "frame_size",
        "max_video_seconds",
        "difference_case_percentile",
    ):
        density.pop(key, None)
    if ensemble is not None:
        density["ensemble_size"] = ensemble
    inputs["numerical_source_sha256"] = _numerical_source_fingerprint()
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_cache_manifest(
    data_root: Path,
    config: ProjectConfig,
    archives: dict[str, Path],
    *,
    ensemble: int | None = None,
) -> Path:
    manifest = data_root / CACHE_MANIFEST
    payload = {
        "schema_version": PDF_SCHEMA_VERSION,
        "simulation_fingerprint": simulation_fingerprint(config, ensemble),
        "archives": {name: path.name for name, path in archives.items()},
    }
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=data_root, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def load_cached_results(
    data_root: Path,
    config: ProjectConfig,
    methods: tuple[str, ...],
    *,
    ensemble: int | None = None,
) -> dict[str, PositionPDFResult] | None:
    manifest = data_root / CACHE_MANIFEST
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != PDF_SCHEMA_VERSION:
            return None
        if payload.get("simulation_fingerprint") != simulation_fingerprint(
            config, ensemble
        ):
            return None
        archives = payload["archives"]
        results = {
            method: load_position_pdf(data_root / archives[method])
            for method in methods
        }
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if any(result.method != method for method, result in results.items()):
        return None
    return results
