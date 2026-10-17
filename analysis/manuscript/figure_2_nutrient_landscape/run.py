#!/usr/bin/env python3
"""Write the nutrient-landscape figure and the design numbers behind it.

The whole numerical comparison runs on one cosine mode, because every closed
form in the manuscript is a function of a single wavenumber.  This script draws
that landscape and the three dimensionless numbers that select it.

Run from the simulation directory:

    python scripts/generate_figure_2_nutrient_landscape.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from model.solvers.chem_swimmer.config import load_config
from model.solvers.chem_swimmer.fields import create_field
from model.solvers.chem_swimmer.visualization.landscape_plot import (
    landscape_summary,
    render_landscape_report,
)

FIGURE = ROOT / "manuscript" / "figures" / "figure-2-nutrient-landscape.pdf"
SUPPLEMENT = (
    ROOT
    / "outputs"
    / "simulations"
    / "population_release"
    / "figures"
    / "supplementary"
    / "figure-s09-nutrient-landscape.pdf"
)
METRICS = (
    ROOT
    / "outputs"
    / "simulations"
    / "population_release"
    / "metrics"
    / "nutrient-landscape.csv"
)


def main() -> None:
    config = load_config(ROOT / "model" / "parameters" / "default.toml")
    field = create_field(config.field)
    summary = landscape_summary(config, field)

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    with METRICS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("quantity", "value"))
        for key, value in summary.items():
            writer.writerow((key, f"{value:.8g}"))

    render_landscape_report(config, FIGURE, field, also_png=False)
    render_landscape_report(config, SUPPLEMENT, field)
    width = max(len(key) for key in summary)
    for key, value in summary.items():
        print(f"  {key:<{width}}  {value:.6g}")
    print(f"wrote {METRICS}\nwrote {FIGURE}\nwrote {SUPPLEMENT}")


if __name__ == "__main__":
    main()
