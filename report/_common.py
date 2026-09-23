"""Shared setup for every chapter: data loading and one consistent figure style.

Colors are the validated reference palette (categorical slots 1-3 and text inks).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "report" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 10,
        "lines.linewidth": 2,
        "figure.figsize": (6.5, 3.2),
        "figure.dpi": 150,
    }
)
pd.set_option("display.precision", 2)

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def games() -> pd.DataFrame:
    return pd.read_parquet(RAW / "games.parquet")


def features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "team_games.parquet")


def home_rows() -> pd.DataFrame:
    """One row per game (home team's row) with results vs the lines."""
    f = features().merge(games()[["game_id", "home_id"]], on="game_id")
    h = f[(f.team_id == f.home_id) & f.completed & ~f.shortened].copy()
    h["total"] = h.points + h.points_allowed
    h["margin"] = h.points - h.points_allowed
    return h


def bar_ends(ax):
    """Recessive axes: no top/right spines, light grid under the data."""
    ax.set_axisbelow(True)
    return ax
