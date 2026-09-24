"""Out-of-sample errors for every period and stat, and a check that the probabilities
built from them are honest.

    uv run python scripts/period_calibration.py   → models/period_calibration.json

Walk-forward 2021-2025, same as the report card: full-game team points from the final
model (random forest), halves and quarters from the period models (LightGBM). The check
fits on 2021-2023 errors and scores 2024-2025 on lines near each prediction.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from canes_cfb import ladder
from canes_cfb.modeling import FEATURES
from canes_cfb.paths import PROCESSED, RAW, ROOT
from canes_cfb.periods import PERIODS, add_period_shares, add_period_targets, fit_predict_period

SEASONS = range(2021, 2026)
CACHE = PROCESSED / "oos_periods_2021_2025.parquet"


def oos_team_periods() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    params = json.loads((ROOT / "models" / "team_points_params.json").read_text())
    features = pd.read_parquet(PROCESSED / "team_games.parquet")
    games = pd.read_parquet(RAW / "games.parquet")[["game_id", "home_id"]]
    f = add_period_shares(add_period_targets(features)).merge(games, on="game_id")
    f["is_home"] = f.team_id == f.home_id
    base = f[f.completed & ~f.shortened & f.season.between(2016, 2025)]
    parts = []
    for p in PERIODS:
        for season in SEASONS:
            train, test = base[base.season < season], base[base.season == season]
            pred = fit_predict_period(p, train, test, FEATURES, params["lightgbm"]["params"])
            parts.append(
                pd.DataFrame(
                    {
                        "game_id": test.game_id.to_numpy(),
                        "season": season,
                        "is_home": test.is_home.to_numpy(),
                        "period": p.value,
                        "pred": pred,
                        "actual": test[f"pts_{p.value}"].to_numpy(),
                    }
                )
            )
    # full game: the final model's out-of-sample team points (scripts/clv_history.py cache)
    g = pd.read_parquet(PROCESSED / "oos_2021_2025.parquet")
    for is_home, pred, pts in ((True, "pred", "home_points"), (False, "pred_away", "away_points")):
        parts.append(
            pd.DataFrame(
                {
                    "game_id": g.game_id,
                    "season": g.season,
                    "is_home": is_home,
                    "period": "Game",
                    "pred": g[pred],
                    "actual": g[pts],
                }  # fmt: skip
            )
        )
    out = pd.concat(parts, ignore_index=True).dropna(subset=["actual"])
    out.to_parquet(CACHE, index=False)
    return out


def check(fit: pd.DataFrame, test: pd.DataFrame, stat: str, step: float) -> dict:
    """Score probabilities on lines at the prediction ± a few steps (x.5 lines)."""
    q = ladder.quantiles(ladder.residuals(fit)[stat])
    if stat == "team":
        pred = np.concatenate([test.pred_home, test.pred_away])
        act = np.concatenate([test.actual_home, test.actual_away])
    elif stat == "total":
        pred = (test.pred_home + test.pred_away).to_numpy()
        act = (test.actual_home + test.actual_away).to_numpy()
    else:
        pred = (test.pred_home - test.pred_away).to_numpy()
        act = (test.actual_home - test.actual_away).to_numpy()
    rows = []
    for k in (-3, -2, -1, 0, 1, 2, 3):
        line = np.floor(pred) + 0.5 + k * step
        rows.append(pd.DataFrame({"p": ladder.p_over(pred, line, q), "hit": act > line}))
    d = pd.concat(rows)
    d["bin"] = pd.cut(d.p, np.linspace(0, 1, 11))
    rel = d.groupby("bin", observed=True).agg(p=("p", "mean"), hit=("hit", "mean"), n=("p", "size"))
    ece = float((rel.p - rel.hit).abs().mul(rel.n).sum() / rel.n.sum())
    brier = float(((d.p - d.hit) ** 2).mean())
    base = float(((d.hit.mean() - d.hit) ** 2).mean())
    return {"ece": round(ece, 4), "brier": round(brier, 4), "brier_no_skill": round(base, 4),
            "n": int(len(d))}  # fmt: skip


def anchored(team: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Game rows per period with predictions = the opening line split by the model."""
    lines = pd.read_parquet(PROCESSED / "oos_2021_2025.parquet")[
        ["game_id", "spread_open", "total_open"]
    ]
    wide = team.pivot_table(index=["game_id", "season", "is_home"], columns="period",
                            values=["pred", "actual"]).reset_index()  # fmt: skip
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide = wide.merge(lines, on="game_id").dropna(subset=["spread_open", "total_open"])
    mh, ma = ladder.market_team_points(wide.total_open, wide.spread_open)
    wide["market"] = np.where(wide.is_home, mh, ma)
    preds = {p: wide[f"pred_{p}"] for p in ladder.PERIOD_KEYS if p != "Game"}
    split = ladder.split_market(preds, wide["market"])
    out = {}
    for p in ladder.PERIOD_KEYS:
        t = wide[["game_id", "season", "is_home"]].assign(
            period=p, pred=split[p].to_numpy(), actual=wide[f"actual_{p}"].to_numpy()
        )
        out[p] = ladder.game_level(t.dropna(subset=["actual"]))
    return out


def main() -> None:
    team = oos_team_periods()
    out = {
        "source": "walk-forward 2021-2025, opening line split across periods by the model",
        "quantiles": {},
        "check_2024_25": {},
    }
    steps = {"Game": 3, "1H": 2, "2H": 2, "Q1": 1, "Q2": 1, "Q3": 1, "Q4": 1}
    by_period = anchored(team)
    for p in ladder.PERIOD_KEYS:
        g = by_period[p]
        out["quantiles"][p] = {s: ladder.quantiles(r) for s, r in ladder.residuals(g).items()}
        fit, test = g[g.season <= 2023], g[g.season >= 2024]
        out["check_2024_25"][p] = {s: check(fit, test, s, steps[p]) for s in ladder.STATS}
        print(p, len(g), {s: out["check_2024_25"][p][s]["ece"] for s in ladder.STATS})
    (ROOT / "models" / "period_calibration.json").write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
