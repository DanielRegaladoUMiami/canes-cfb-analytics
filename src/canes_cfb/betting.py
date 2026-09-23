"""From team-points predictions to graded spread and total bets.

Conventions (docs/markets.md): margin = home - away; the spread is the home team's line
(negative = home favored); home covers when ``margin + spread > 0``. Pushes return the
stake and are excluded from win rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BREAK_EVEN_110 = 110 / 210  # 52.38%
PAYOUT_110 = 100 / 110


def to_games(team_rows: pd.DataFrame, games: pd.DataFrame, pred: str = "pred") -> pd.DataFrame:
    """Pair the two team rows of each game into one row: predicted margin and total.

    ``team_rows`` needs ``game_id``, ``team_id`` and ``pred``; ``games`` needs
    ``game_id``, ``home_id``, ``home_points``, ``away_points``. Games missing either
    team's prediction are dropped.
    """
    rows = team_rows.merge(
        games[["game_id", "home_id", "home_points", "away_points"]], on="game_id"
    )
    is_home = rows["team_id"] == rows["home_id"]
    home = rows[is_home].drop(columns=["team_id", "home_id"])
    away = rows.loc[~is_home, ["game_id", pred]].rename(columns={pred: f"{pred}_away"})
    out = home.merge(away, on="game_id")
    out["pred_margin"] = out[pred] - out[f"{pred}_away"]
    out["pred_total"] = out[pred] + out[f"{pred}_away"]
    out["margin"] = out["home_points"] - out["away_points"]
    out["total"] = out["home_points"] + out["away_points"]
    return out


def spread_edge(g: pd.DataFrame, line: str) -> pd.Series:
    """Points by which the model likes home against ``line`` (negative = likes away)."""
    return g["pred_margin"] + g[line]


def spread_result(g: pd.DataFrame, line: str) -> pd.Series:
    """+1 home covered, -1 away covered, 0 push."""
    return np.sign(g["margin"] + g[line])


def total_edge(g: pd.DataFrame, line: str) -> pd.Series:
    """Points by which the model likes the over (negative = likes the under)."""
    return g["pred_total"] - g[line]


def total_result(g: pd.DataFrame, line: str) -> pd.Series:
    return np.sign(g["total"] - g[line])


def spread_clv(edge: pd.Series, g: pd.DataFrame) -> pd.Series:
    """Points the spread moved toward our side between open and close.

    Taking home at open (-3) and seeing it close at -4.5 is +1.5: the market agreed.
    """
    return np.sign(edge) * (g["spread_open"] - g["spread_close"])


def total_clv(edge: pd.Series, g: pd.DataFrame) -> pd.Series:
    return np.sign(edge) * (g["total_close"] - g["total_open"])


def grade(
    edge: pd.Series,
    result: pd.Series,
    clv: pd.Series | None = None,
    thresholds: tuple[float, ...] = (0, 2, 3, 4, 5, 7),
) -> pd.DataFrame:
    """Win rate, ROI at -110 and average CLV for bets with ``|edge| >= threshold``."""
    rows = []
    for threshold in thresholds:
        take = (edge.abs() >= threshold) & (result != 0) & edge.notna() & result.notna()
        n = int(take.sum())
        win = float((np.sign(edge[take]) == result[take]).mean()) if n else np.nan
        row = {
            "min edge": threshold,
            "bets": n,
            "win %": 100 * win,
            "ROI %": 100 * (win * (1 + PAYOUT_110) - 1),
        }
        if clv is not None:
            row["avg CLV"] = float(clv[take].mean()) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
