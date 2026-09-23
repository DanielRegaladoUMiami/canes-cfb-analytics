"""Round 2: improve accuracy without overfitting. Pre-registered experiments only.

    uv run python scripts/round2_experiments.py

Protocol (fixed before running):
- Candidates: A (current: random forest team points), B (average of the six tuned models),
  C (dedicated game-level margin and total models), D (B and C averaged), E (random forest
  with the training residual winsorized at ±21 points, so blowouts teach less).
- Decision seasons 2021-2023: a candidate is adopted only if its margin MAE beats A in at
  least 2 of the 3 seasons and the paired bootstrap 95% interval of the improvement
  excludes zero. 2024-2025 are reported as a check, never used to choose.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from canes_cfb.betting import to_games
from canes_cfb.modeling import FEATURES, SPECS, fit_predict
from canes_cfb.paths import PROCESSED, RAW, ROOT

SEASONS = range(2021, 2026)
DECIDE = (2021, 2023)
rng = np.random.default_rng(7)


def team_preds(base: pd.DataFrame, params: dict, models: list[str], clip: float | None = None):
    """Walk-forward team-points predictions averaged over ``models``."""
    parts = []
    for season in SEASONS:
        train, test = base[base.season < season].copy(), base[base.season == season].copy()
        if clip is not None:  # winsorize what the trees learn: points - exp_points
            resid = (train.points - train.exp_points).clip(-clip, clip)
            train["points"] = train.exp_points + resid
        preds = [fit_predict(SPECS[m], params[m]["params"], train, test) for m in models]
        test["pred"] = np.mean(preds, axis=0)
        parts.append(test)
    return pd.concat(parts)


def game_rows(team: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    keep = ["game_id", "team_id", "pred", "season", "season_type", "week", "spread_open",
            "spread_close", "total_open", "total_close"]  # fmt: skip
    return to_games(team[keep], games)


def dedicated(base: pd.DataFrame, games: pd.DataFrame, lgb_params: dict) -> pd.DataFrame:
    """Game-level models on the home row: margin (base exp_margin) and total (base exp_total)."""
    home = base.merge(games[["game_id", "home_id"]], on="game_id")
    home = home[home.team_id == home.home_id].copy()
    home["margin"] = home.points - home.points_allowed
    home["total"] = home.points + home.points_allowed
    out = []
    for season in SEASONS:
        train, test = home[home.season < season], home[home.season == season].copy()
        for target, basecol in (("margin", "exp_margin"), ("total", "exp_total")):
            model = lgb.LGBMRegressor(
                subsample_freq=1, random_state=7, verbose=-1, objective="l1", **lgb_params
            ).fit(train[FEATURES], train[target] - train[basecol])
            test[f"ded_{target}"] = test[basecol] + model.predict(test[FEATURES])
        out.append(test[["game_id", "ded_margin", "ded_total"]])
    return pd.concat(out)


def metrics(g: pd.DataFrame) -> dict:
    res_s = np.sign(g.margin + g.spread_open)
    res_t = np.sign(g.total - g.total_open)
    ok_s, ok_t = (res_s != 0) & g.spread_open.notna(), (res_t != 0) & g.total_open.notna()
    return {
        "margin MAE": (g.margin - g.pred_margin).abs().mean(),
        "total MAE": (g.total - g.pred_total).abs().mean(),
        "winner %": 100 * ((g.pred_margin > 0) == (g.margin > 0)).mean(),
        "ATS open %": 100 * (np.sign(g.pred_margin + g.spread_open)[ok_s] == res_s[ok_s]).mean(),
        "O/U open %": 100 * (np.sign(g.pred_total - g.total_open)[ok_t] == res_t[ok_t]).mean(),
    }


def bootstrap_gain(a_err: np.ndarray, b_err: np.ndarray, n: int = 2000) -> tuple[float, float]:
    """95% interval of mean(a_err - b_err) (positive = candidate better), paired by game."""
    d = a_err - b_err
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    params = json.loads((ROOT / "models" / "team_points_params.json").read_text())
    f = pd.read_parquet(PROCESSED / "team_games.parquet")
    games = pd.read_parquet(RAW / "games.parquet")
    base = f[f.completed & ~f.shortened & f.season.between(2016, 2025)]
    all_models = list(SPECS)

    cands = {
        "A current (RF)": game_rows(team_preds(base, params, ["random_forest"]), games),
        "B average of 6": game_rows(team_preds(base, params, all_models), games),
        "E RF, blowouts clipped": game_rows(
            team_preds(base, params, ["random_forest"], clip=21.0), games
        ),
    }
    ded = dedicated(base, games, params["lightgbm"]["params"])
    c = cands["A current (RF)"].merge(ded, on="game_id")
    c["pred_margin"], c["pred_total"] = c.ded_margin, c.ded_total
    cands["C dedicated margin/total"] = c
    d = cands["B average of 6"].merge(ded, on="game_id")
    d["pred_margin"] = (d.pred_margin + d.ded_margin) / 2
    d["pred_total"] = (d.pred_total + d.ded_total) / 2
    cands["D = B + C"] = d

    a = cands["A current (RF)"].set_index("game_id")
    rows, verdicts = [], {}
    for name, g in cands.items():
        g = g.set_index("game_id").loc[a.index]
        for label, yrs in (("decide 2021-23", DECIDE), ("check 2024-25", (2024, 2025))):
            m = g.season.between(*yrs)
            rows.append({"candidate": name, "period": label, **metrics(g[m])})
        dec = g.season.between(*DECIDE)
        a_err = (a.margin - a.pred_margin).abs()[dec].to_numpy()
        c_err = (g.margin - g.pred_margin).abs()[dec].to_numpy()
        seasons_won = sum(
            (a[dec & (a.season == s)].margin - a[dec & (a.season == s)].pred_margin).abs().mean()
            > (g[dec & (g.season == s)].margin - g[dec & (g.season == s)].pred_margin).abs().mean()
            for s in range(DECIDE[0], DECIDE[1] + 1)
        )
        lo, hi = bootstrap_gain(a_err, c_err)
        adopt = name != "A current (RF)" and seasons_won >= 2 and lo > 0
        verdicts[name] = {
            "seasons better than A (of 3)": seasons_won,
            "margin MAE gain 95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
            "adopt": adopt,
        }
    table = pd.DataFrame(rows).set_index(["period", "candidate"]).sort_index().round(2)
    print(table.to_string())
    print()
    print(pd.DataFrame(verdicts).T.to_string())
    out = {"table": table.reset_index().to_dict(orient="records"), "verdicts": verdicts}
    (ROOT / "models" / "round2_experiments.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
