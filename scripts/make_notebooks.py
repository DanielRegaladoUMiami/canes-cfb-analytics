"""Generate the notebook tree from the market registry.

Idempotent: existing notebooks and READMEs are never overwritten, so this is safe to re-run
after adding a market or period to ``canes_cfb.markets``.

    uv run python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

from canes_cfb.markets import MARKETS, PERIOD_COMPONENTS, MarketSpec, Period, notebook_name
from canes_cfb.paths import NOTEBOOKS

SETUP_CELL = """\
import numpy as np
import pandas as pd

from canes_cfb.markets import MARKETS, PERIOD_COMPONENTS, MarketType, Period
from canes_cfb.paths import PREDICTIONS, PROCESSED

pd.set_option("display.max_columns", 50)"""

# Notebooks outside the market grid: (folder, file, title, objective, sections)
SUPPORT_NOTEBOOKS = [
    (
        "00_data",
        "01_ingest_games.ipynb",
        "Ingest games and scores",
        "Pull every FBS game (schedule, venue, neutral site, final score) into `data/raw/`.",
        ["Pull seasons", "Validate (one row per game, no duplicates)", "Save to data/raw"],
    ),
    (
        "00_data",
        "02_ingest_line_scores.ipynb",
        "Ingest scoring by quarter",
        "Pull quarter-by-quarter line scores (Q1-Q4 + OT) so every period target can be built.",
        ["Pull line scores", "Check quarters sum to the final score", "Save to data/raw"],
    ),
    (
        "00_data",
        "03_ingest_betting_lines.ipynb",
        "Ingest betting lines",
        "Pull opening and closing lines for every market and period we model.",
        ["Pull lines", "Normalize to home-perspective conventions", "Save to data/raw"],
    ),
    (
        "00_data",
        "04_build_feature_table.ipynb",
        "Build the shared feature table",
        "One row per game with as-of features (known before kickoff) plus every period target. "
        "Every model notebook reads this table.",
        ["Targets per period", "Team ratings and efficiency features", "Save to data/processed"],
    ),
    (
        "99_evaluation",
        "01_backtest_all_markets.ipynb",
        "Backtest all markets",
        "Walk-forward backtest of every model against closing lines, in one comparable table.",
        ["Load predictions", "Metrics by market and period", "ROI and CLV"],
    ),
    (
        "99_evaluation",
        "02_calibration.ipynb",
        "Calibration",
        "Are predicted probabilities (win, cover, over) honest? Reliability curves per market.",
        ["Reliability curves", "Calibration by season and by line size"],
    ),
    (
        "99_evaluation",
        "03_weekly_card.ipynb",
        "Weekly card",
        "This week's predictions from every model, ranked by edge against the current line.",
        ["Load this week's predictions", "Join current lines", "Rank by edge"],
    ),
]

SUPPORT_READMES = {
    "00_data": (
        "Data",
        "Ingestion and the shared feature table. Run these in order before any model notebook.",
    ),
    "99_evaluation": (
        "Evaluation",
        "Cross-market backtests, calibration, and the weekly card of picks.",
    ),
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    # Sequential cell ids match what nbstripout writes, so the pre-commit hook is a no-op.
    cells = [{**cell, "id": str(i)} for i, cell in enumerate(cells)]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def to_json(nb: dict) -> str:
    # Same layout nbstripout writes, so the pre-commit hook leaves generated files untouched.
    return json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def model_notebook(spec: MarketSpec, period: Period) -> dict:
    components = " + ".join(PERIOD_COMPONENTS[period])
    title = f"{spec.market.value.replace('_', ' ').title()} — {period.value}"
    header = (
        f"# {title}\n\n"
        f"**Target:** {spec.target}  \n"
        f"**Period:** {period.value} = {components}  \n"
        f"**Kind:** {spec.target_kind}  \n"
        f"**Priced against:** {spec.line}  \n"
        f"**Metrics:** {', '.join(spec.metrics)}\n\n"
        "Status: not started. Log results in `docs/experiments/` when a version is done."
    )
    return notebook(
        [
            md(header),
            code(SETUP_CELL),
            md("## 1. Load data\nRead the shared feature table from `data/processed/`."),
            code(""),
            md(
                "## 2. Target and features\n"
                "Build the target for this period. Use only features known before kickoff."
            ),
            code(""),
            md("## 3. Baseline\nThe line itself (or a naive rule) is the benchmark to beat."),
            code(""),
            md("## 4. Model\nTrain with walk-forward splits by season/week. No random splits."),
            code(""),
            md("## 5. Evaluate against the line\n" + ", ".join(spec.metrics)),
            code(""),
            md(
                "## 6. This week's predictions\n"
                f"Write to `data/predictions/{spec.market.value}_{period.value}.parquet`."
            ),
            code(""),
            md("## Notes\n- "),
        ]
    )


def support_notebook(title: str, objective: str, sections: list[str]) -> dict:
    cells = [md(f"# {title}\n\n{objective}"), code(SETUP_CELL)]
    for i, section in enumerate(sections, 1):
        cells += [md(f"## {i}. {section}"), code("")]
    return notebook(cells)


def market_readme(spec: MarketSpec) -> str:
    rows = "\n".join(
        f"| [`{notebook_name(p)}`]({notebook_name(p)}) | {p.value} | "
        f"{' + '.join(PERIOD_COMPONENTS[p])} | not started |"
        for p in spec.periods
    )
    title = spec.market.value.replace("_", " ").title()
    return (
        f"# {title}\n\n"
        f"**Target:** {spec.target} ({spec.target_kind})  \n"
        f"**Priced against:** {spec.line}  \n"
        f"**Metrics:** {', '.join(spec.metrics)}\n\n"
        "| Notebook | Period | Points counted | Status |\n"
        "|---|---|---|---|\n"
        f"{rows}\n\n"
        "Conventions: see [`docs/markets.md`](../../docs/markets.md).\n"
    )


def write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def main() -> None:
    created = []
    for spec in MARKETS.values():
        folder = NOTEBOOKS / spec.folder
        if write_if_missing(folder / "README.md", market_readme(spec)):
            created.append(folder / "README.md")
        for period in spec.periods:
            nb = to_json(model_notebook(spec, period))
            if write_if_missing(folder / notebook_name(period), nb):
                created.append(folder / notebook_name(period))

    for folder_name, (title, blurb) in SUPPORT_READMES.items():
        names = [f for d, f, *_ in SUPPORT_NOTEBOOKS if d == folder_name]
        listing = "\n".join(f"- [`{n}`]({n})" for n in names)
        path = NOTEBOOKS / folder_name / "README.md"
        if write_if_missing(path, f"# {title}\n\n{blurb}\n\n{listing}\n"):
            created.append(path)

    for folder_name, file, title, objective, sections in SUPPORT_NOTEBOOKS:
        nb = to_json(support_notebook(title, objective, sections))
        path = NOTEBOOKS / folder_name / file
        if write_if_missing(path, nb):
            created.append(path)

    for path in created:
        print(f"created {path.relative_to(NOTEBOOKS.parent)}")
    print(f"{len(created)} files created")


if __name__ == "__main__":
    main()
