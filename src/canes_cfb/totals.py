"""Game-level totals: over/under vs the market line, with the nonlinear features found in
notebooks/00_data/05_nonlinearity_insights.ipynb.

One row per game (home team's perspective). Features come from both teams' as-of rows in
the feature table; nothing here uses the game's result except the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHOOTOUT_EXP_TOTAL = 63.5  # top 20% of exp_total, 2016-2023 (paper rule B)

# Per-team columns pulled from both sides of the game.
_SIDE_COLS = ["exp_points", "exp_plays", "exp_ppa", "exp_sr", "exp_expl", "talent", "ret_ppa"]

BASE_FEATURES = [
    "line", "exp_total", "gap", "abs_exp_margin", "game_pace", "exp_plays_sum", "exp_ppa_sum",
    "exp_sr_mean", "exp_expl_mean", "talent_gap", "ret_min", "week", "neutral_site", "indoor",
    "conference_game",
]  # fmt: skip

# Explicit nonlinear terms: thresholds and tails suggested by SHAP and the market lens.
NONLINEAR_FEATURES = [
    "shootout", "exp_total_z2", "line_high", "pace_tail", "talent_gap_tail", "ret_low",
    "early_season", "blowout_script", "gap_x_shootout",
]  # fmt: skip


def build_totals_table(features: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Home row + away row → one game row with both teams' as-of features and lines."""
    f = features.merge(games[["game_id", "home_id"]], on="game_id")
    home = f[f.team_id == f.home_id]
    away = f[f.team_id != f.home_id][["game_id", *_SIDE_COLS]].rename(
        columns={c: f"away_{c}" for c in _SIDE_COLS}
    )
    g = home.merge(away, on="game_id")
    out = pd.DataFrame(
        {
            "game_id": g.game_id,
            "season": g.season,
            "season_type": g.season_type,
            "week": g.week,
            "completed": g.completed,
            "shortened": g.shortened,
            "total": g.points + g.points_allowed,
            "total_open": g.total_open,
            "total_close": g.total_close,
            "exp_total": g.exp_total,
            "abs_exp_margin": g.exp_margin.abs(),
            "game_pace": g.game_pace,
            "exp_plays_sum": g.exp_plays + g.away_exp_plays,
            "exp_ppa_sum": g.exp_ppa + g.away_exp_ppa,
            "exp_sr_mean": (g.exp_sr + g.away_exp_sr) / 2,
            "exp_expl_mean": (g.exp_expl + g.away_exp_expl) / 2,
            "talent_gap": (g.talent - g.away_talent).abs(),
            "ret_min": np.fmin(g.ret_ppa, g.away_ret_ppa),
            "neutral_site": g.neutral_site,
            "indoor": g.indoor,
            "conference_game": g.conference_game,
        }
    )
    return out.reset_index(drop=True)


def add_line_features(t: pd.DataFrame, line: str) -> pd.DataFrame:
    """Features that depend on the market line the bet is graded against.

    Thresholds are fixed constants chosen from 2016-2023 (never from later seasons), so
    they add no look-ahead inside the walk-forward folds after 2023.
    """
    out = t.copy()
    out["line"] = out[line]
    out["gap"] = out.exp_total - out.line
    out["shootout"] = (out.exp_total > SHOOTOUT_EXP_TOTAL).astype(float)
    out["exp_total_z2"] = ((out.exp_total - 55.0) / 8.0) ** 2  # curvature at both extremes
    out["line_high"] = (out.line > 62.5).astype(float)
    out["pace_tail"] = (out.game_pace > 4.8).astype(float)  # top 20% of 2016-2023
    out["talent_gap_tail"] = np.clip(out.talent_gap - 250.0, 0, None)
    out["ret_low"] = (out.ret_min < 0.35).astype(float)
    out["early_season"] = ((out.season_type == 2) & (out.week <= 3)).astype(float)
    out["blowout_script"] = np.clip(out.abs_exp_margin - 14.0, 0, None)
    out["gap_x_shootout"] = out.gap * out.shootout
    decided = np.sign(out.total - out.line)
    out["under"] = np.where(decided == 0, np.nan, (decided < 0).astype(float))
    return out
