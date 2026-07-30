"""Portable provenance recorded inside numerical result archives."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np

from ._version import PDF_SCHEMA_VERSION, __version__


@dataclass(frozen=True)
class ArtifactProvenance:
    schema_version: int
    software_version: str
    config_text: str
    config_sha256: str
    runtime_versions: str


def build_provenance(config_source: str) -> ArtifactProvenance:
    """Capture the actual input text plus versions needed to audit an archive."""

    config_text = ""
    if config_source:
        path = Path(config_source).expanduser()
        if path.is_file():
            config_text = path.read_text(encoding="utf-8")
    config_sha256 = (
        hashlib.sha256(config_text.encode("utf-8")).hexdigest()
        if config_text
        else ""
    )
    runtime_versions = json.dumps(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return ArtifactProvenance(
        schema_version=PDF_SCHEMA_VERSION,
        software_version=__version__,
        config_text=config_text,
        config_sha256=config_sha256,
        runtime_versions=runtime_versions,
    )
