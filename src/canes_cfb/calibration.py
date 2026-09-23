"""Turn point predictions into honest probabilities.

The model predicts points, not probabilities. Probabilities here come from how its
**out-of-sample** predictions actually did (walk-forward, 2021-2025):

- P(home wins) = Phi(pred_margin / sigma_margin), with sigma from out-of-sample margin
  errors, checked with a reliability table.
- P(cover) and P(over) come from a logistic fit of real outcomes against the model's edge
  vs the opening line. That's how often bets with that edge actually won, which is far
  less confident than the model's own point spread would suggest.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from canes_cfb import betting
from canes_cfb.modeling import SPECS, fit_predict

SEASONS = range(2021, 2026)


def out_of_sample_games(features: pd.DataFrame, games: pd.DataFrame, recipe_params: dict):
    """Game rows with walk-forward predictions of the final model for 2021-2025."""
    base = features[features.completed & ~features.shortened & features.season.between(2016, 2025)]
    parts = []
    for season in SEASONS:
        train, test = base[base.season < season], base[base.season == season].copy()
        test["pred"] = fit_predict(SPECS["random_forest"], recipe_params, train, test)
        parts.append(test)
    rows = pd.concat(parts)
    keep = ["game_id", "team_id", "pred", "season", "season_type", "week", "spread_open",
            "spread_close", "total_open", "total_close", "exp_total"]  # fmt: skip
    return betting.to_games(rows[keep], games)


SHOOTOUT_EXP_TOTAL = 63.5  # rule B threshold (top 20% of exp_total, 2016-2023)


def _logit(x: pd.DataFrame, result: pd.Series) -> dict:
    """Logistic fit of P(home covers / over) on ``x``; coefficients per column."""
    ok = x.notna().all(axis=1) & result.notna() & (result != 0)
    model = LogisticRegression(C=1.0).fit(x[ok], (result[ok] > 0).astype(int))
    coefs = {c: float(v) for c, v in zip(x.columns, model.coef_[0], strict=True)}
    return {"intercept": float(model.intercept_[0]), "coef": coefs, "n": int(ok.sum())}


def total_inputs(g: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge": betting.total_edge(g, "total_open"),
            "shootout": (g["exp_total"] > SHOOTOUT_EXP_TOTAL).astype(float),
        },
        index=g.index,
    )


def spread_inputs(g: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"edge": betting.spread_edge(g, "spread_open")}, index=g.index)


def fit(g: pd.DataFrame) -> dict:
    """Calibration parameters from out-of-sample game rows."""
    sigma_margin = float((g.margin - g.pred_margin).std())
    sigma_total = float((g.total - g.pred_total).std())
    home_p = norm.cdf(g.pred_margin / sigma_margin)
    won = (g.margin > 0).astype(float).where(g.margin != 0)
    bins = pd.cut(home_p, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    reliability = (
        pd.DataFrame({"predicted": home_p, "actual": won, "bin": bins})
        .dropna()
        .groupby("bin", observed=True)
        .agg(predicted=("predicted", "mean"), actual=("actual", "mean"), n=("actual", "size"))
        .reset_index(drop=True)
    )
    spread = _logit(spread_inputs(g), betting.spread_result(g, "spread_open"))
    total = _logit(total_inputs(g), betting.total_result(g, "total_open"))
    return {
        "seasons": f"{min(SEASONS)}-{max(SEASONS)}",
        "games": int(len(g)),
        "sigma_margin": sigma_margin,
        "sigma_total": sigma_total,
        "mae_margin": float((g.margin - g.pred_margin).abs().mean()),
        "mae_total": float((g.total - g.pred_total).abs().mean()),
        "win_reliability": reliability.round(4).to_dict(orient="records"),
        "spread_vs_open": spread,
        "total_vs_open": total,
    }


def win_probability(pred_margin, cal: dict):
    return norm.cdf(np.asarray(pred_margin, dtype=float) / cal["sigma_margin"])


def probability(x: pd.DataFrame, params: dict) -> np.ndarray:
    """P(home covers) or P(over) from the historical fit; callers flip for the other side."""
    z = params["intercept"] + sum(params["coef"][c] * x[c].to_numpy(float) for c in params["coef"])
    return 1 / (1 + np.exp(-z))


def expected_value(p_win):
    """Expected profit per 1 unit risked at -110."""
    p = np.asarray(p_win, dtype=float)
    return p * betting.PAYOUT_110 - (1 - p)


def save(cal: dict, path) -> None:
    path.write_text(json.dumps(cal, indent=2) + "\n")
