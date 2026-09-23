"""Market layer: when should we trust the model's disagreement with the opening line?

Opening lines only exist from 2021 on, so this layer is deliberately tiny: a regularized
logistic regression on three inputs. It turns the points model's edge against the opener
into a probability of winning the bet, learning how much the edge is worth (and whether
it's worth less early in the season or on big lines) instead of trusting it at face value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from canes_cfb import betting

INPUTS = ["edge", "edge_x_early", "abs_line"]


def design(g: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Inputs and outcome for ``kind`` in {"spread", "total"} against the opening line.

    ``side_won`` is 1 when home covered (spread) or the over hit (total), 0 otherwise,
    NaN on pushes or missing lines.
    """
    if kind == "spread":
        edge = betting.spread_edge(g, "spread_open")
        result = betting.spread_result(g, "spread_open")
        abs_line = g["spread_open"].abs()
    elif kind == "total":
        edge = betting.total_edge(g, "total_open")
        result = betting.total_result(g, "total_open")
        abs_line = g["total_open"]
    else:
        raise ValueError(kind)
    early = ((g["season_type"] == 2) & (g["week"] <= 4)).astype(float)
    out = pd.DataFrame(
        {
            "edge": edge,
            "edge_x_early": edge * early,
            "abs_line": abs_line,
            "side_won": np.where(result == 0, np.nan, (result > 0).astype(float)),
        },
        index=g.index,
    )
    out.loc[result.isna() | edge.isna(), "side_won"] = np.nan
    return out


def fit(g: pd.DataFrame, kind: str, c: float = 0.05):
    """Fit on games with a decided result (no pushes)."""
    d = design(g, kind).dropna()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c))
    return model.fit(d[INPUTS], d["side_won"].astype(int))


def side_probability(model, g: pd.DataFrame, kind: str) -> pd.Series:
    """P(home covers) for spreads or P(over) for totals; NaN where inputs are missing."""
    d = design(g, kind)
    ok = d[INPUTS].notna().all(axis=1)
    p = pd.Series(np.nan, index=g.index)
    p[ok] = model.predict_proba(d.loc[ok, INPUTS])[:, 1]
    return p


def grade_by_confidence(
    p: pd.Series,
    g: pd.DataFrame,
    kind: str,
    thresholds: tuple[float, ...] = (0.5, 0.525, 0.54, 0.56, 0.58),
) -> pd.DataFrame:
    """Bet the likelier side when its probability clears each threshold."""
    d = design(g, kind)
    confidence = np.maximum(p, 1 - p)
    side = np.sign(p - 0.5)
    result = 2 * d["side_won"] - 1  # +1 home/over won, -1 lost, NaN push
    if kind == "spread":
        clv = betting.spread_clv(side, g)
    else:
        clv = betting.total_clv(side, g)
    rows = []
    for t in thresholds:
        take = (confidence >= t) & result.notna() & p.notna()
        n = int(take.sum())
        win = float((side[take] == result[take]).mean()) if n else np.nan
        rows.append(
            {
                "min prob": t,
                "bets": n,
                "win %": 100 * win,
                "ROI %": 100 * (win * (1 + betting.PAYOUT_110) - 1),
                "avg CLV": float(clv[take].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)
