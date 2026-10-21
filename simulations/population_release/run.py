#!/usr/bin/env python3
"""Run the project from the repository root: python -m simulations.population_release.run <command>."""

from __future__ import annotations


def main() -> None:
    from model.solvers.chem_swimmer.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
