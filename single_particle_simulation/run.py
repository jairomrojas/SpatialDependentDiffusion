#!/usr/bin/env python3
"""Run the project without installing it first."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    from chem_swimmer.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
