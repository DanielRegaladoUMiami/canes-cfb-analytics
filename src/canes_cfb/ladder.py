"""Probabilities for any line on any period, from the model's out-of-sample errors.

For a stat (team points, total or home margin) in a period (game, halves, quarters), the
chance it lands above a line L is how often the model's real 2021-2025 errors would have
put it there: P(actual > L) = share of residuals r with pred + r > L. Empirical, so it
keeps the lumpy shape of football scores (3s and 7s, scoreless quarters) that a normal
curve misses. Residuals are stored as quantiles in models/period_calibration.json.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIOD_KEYS = ["Game", "1H", "2H", "Q1", "Q2", "Q3", "Q4"]
STATS = ["team", "total", "margin"]
QUANTILES = np.linspace(0, 1, 401)


def game_level(oos_team: pd.DataFrame) -> pd.DataFrame:
    """Team-period rows (game_id, is_home, period, pred, actual) → one row per game and
    period with home/away predictions and actuals."""
    home = oos_team[oos_team.is_home].drop(columns="is_home")
    away = oos_team[~oos_team.is_home].drop(columns="is_home")
    return home.merge(away, on=["game_id", "season", "period"], suffixes=("_home", "_away"))


OT_SHARE = 0.0096  # share of points scored in overtime, 2016-2025 (quarters exclude OT)


def market_team_points(total_line, spread_line):
    """Each team's points implied by the sportsbook line: home = (T - S) / 2, away =
    (T + S) / 2, with S the home spread."""
    return (total_line - spread_line) / 2, (total_line + spread_line) / 2


def split_market(preds: dict, market_pts) -> dict:
    """One team's points per period: the sportsbook's full-game level, split across the
    game by the model's shares.

    ``preds`` maps "1H", "2H", "Q1".."Q4" to the model's period predictions. Halves are
    scaled to add up to the market's points, quarters to the market's points minus the
    usual overtime share. The period models alone come out low (they predict medians, and
    quarter scoring is lumpy), and their level is worse than the market's: out of sample
    2021-2025 this split beats them in every period
    (docs/experiments/2026-09-24_kalshi_periods.md)."""
    h = preds["1H"] + preds["2H"]
    q = preds["Q1"] + preds["Q2"] + preds["Q3"] + preds["Q4"]
    out = {"Game": market_pts}
    for p in ("1H", "2H"):
        out[p] = market_pts * preds[p] / h
    for p in ("Q1", "Q2", "Q3", "Q4"):
        out[p] = market_pts * (1 - OT_SHARE) * preds[p] / q
    return out


def residuals(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """Residuals (actual - pred) per stat for game-level rows of one period."""
    return {
        "team": np.concatenate([g.actual_home - g.pred_home, g.actual_away - g.pred_away]),
        "total": ((g.actual_home + g.actual_away) - (g.pred_home + g.pred_away)).to_numpy(),
        "margin": ((g.actual_home - g.actual_away) - (g.pred_home - g.pred_away)).to_numpy(),
    }


def quantiles(r: np.ndarray) -> list[float]:
    return [round(float(v), 3) for v in np.quantile(r, QUANTILES)]


def p_over(pred: float | np.ndarray, line: float | np.ndarray, q: list[float]) -> np.ndarray:
    """P(actual > line) given the prediction and the stat's residual quantiles."""
    need = np.asarray(line, dtype=float) - np.asarray(pred, dtype=float)
    # share of residuals above `need` = 1 - CDF(need), CDF by interpolating the quantiles
    cdf = np.interp(need, np.asarray(q), QUANTILES, left=0.0, right=1.0)
    return np.clip(1 - cdf, 0.005, 0.995)
