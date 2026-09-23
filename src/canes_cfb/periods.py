"""Halves and quarters: per-period team points targets and as-of scoring-share features.

A team's points in a period ≈ its expected game points × the share of its points it tends
to score in that period. Shares are team-specific (fast starters, strong closers) and so
are the shares a defense allows. Both are exponentially weighted over prior games only
(shifted), carried across seasons.

Periods follow docs/markets.md: quarters are regulation only, 1H = Q1 + Q2,
2H = Q3 + Q4 + OT.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from canes_cfb.markets import PERIOD_COMPONENTS, Period

PERIODS = [p for p in Period if p is not Period.FULL_GAME]  # 1H, 2H, Q1..Q4


def _cols(period: Period) -> list[str]:
    return [c.lower() for c in PERIOD_COMPONENTS[period]]


def add_period_targets(tg: pd.DataFrame) -> pd.DataFrame:
    """``pts_<period>``: team points in each period (NaN for unplayed or shortened games)."""
    out = tg.copy()
    for p in PERIODS:
        out[f"pts_{p.value}"] = out[_cols(p)].sum(axis=1, min_count=len(_cols(p)))
        out.loc[~out["completed"] | out["shortened"], f"pts_{p.value}"] = np.nan
    return out


def add_period_shares(tg: pd.DataFrame, halflife: float = 8.0) -> pd.DataFrame:
    """As-of share features per period: ``share_<p>`` (team's own scoring share),
    ``opp_allowed_share_<p>`` (share the opponent's defense usually allows in that period),
    and ``exp_pts_<p>`` = exp_points × the average of the two.

    Needs ``add_period_targets`` first, plus ``exp_points`` from the feature table.
    """
    out = tg.sort_values("start_utc").copy()
    for p in PERIODS:
        own = out[f"pts_{p.value}"] / out["points"].where(out["points"] > 0)
        out[f"_own_{p.value}"] = own
    # The share a team's defense allows = the opponent's own share in that game.
    own_cols = [f"_own_{p.value}" for p in PERIODS]
    allowed = out[["game_id", "team_id", *own_cols]].rename(
        columns={"team_id": "opp_id", **{c: c.replace("_own_", "_allowed_") for c in own_cols}}
    )
    out = out.merge(allowed, on=["game_id", "opp_id"], how="left").sort_values("start_utc")

    by_team = out.groupby("team_id")
    for p in PERIODS:
        for kind in ("own", "allowed"):
            col = f"_{kind}_{p.value}"
            out[f"{kind}_ewm_{p.value}"] = by_team[col].transform(
                lambda s: s.ewm(halflife=halflife, ignore_na=True).mean().shift()
            )
    # Opponent's allowed share comes from the opponent's own row.
    opp = out[["game_id", "team_id"] + [f"allowed_ewm_{p.value}" for p in PERIODS]].rename(
        columns={
            "team_id": "opp_id",
            **{f"allowed_ewm_{p.value}": f"opp_allowed_share_{p.value}" for p in PERIODS},
        }
    )
    out = out.merge(opp, on=["game_id", "opp_id"], how="left")
    for p in PERIODS:
        out = out.rename(columns={f"own_ewm_{p.value}": f"share_{p.value}"})
        blended = out[[f"share_{p.value}", f"opp_allowed_share_{p.value}"]].mean(axis=1)
        out[f"exp_pts_{p.value}"] = out["exp_points"] * blended
    drop = [c for c in out.columns if c.startswith(("_own_", "_allowed_", "allowed_ewm_"))]
    return out.drop(columns=drop)


def period_features(p: Period, base: list[str]) -> list[str]:
    return [*base, f"share_{p.value}", f"opp_allowed_share_{p.value}", f"exp_pts_{p.value}"]


def fit_predict_period(
    p: Period, train: pd.DataFrame, test: pd.DataFrame, base_features: list[str], params: dict
) -> np.ndarray:
    """Team points in period ``p``: exp_points × league share (from ``train``) + LightGBM
    on the residual. Team-specific shares stay available as features, but as a base
    they're worse than the league share (noise).
    """
    target = f"pts_{p.value}"
    fit_rows = train[train[target].notna()]
    share = fit_rows[target].sum() / fit_rows["points"].sum()
    cols = period_features(p, base_features)
    model = lgb.LGBMRegressor(
        subsample_freq=1, random_state=7, verbose=-1, objective="l1", **params
    ).fit(fit_rows[cols], fit_rows[target] - fit_rows["exp_points"] * share)
    return test["exp_points"].to_numpy() * share + model.predict(test[cols])
