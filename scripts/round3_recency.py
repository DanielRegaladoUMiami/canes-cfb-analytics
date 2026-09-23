"""Round 3: make recent information count more. Pre-registered experiments only.

    uv run python scripts/round3_recency.py

Candidates (fixed before running), all with the current random forest recipe:
- A: current model.
- R1: season recency weights in training (a season 2 years back counts half).
- R2: short-memory "form" ratings (points and EPA per play, half-life 30 days over the
  last 150 days) plus momentum = form rating minus the long-memory rating.
- R3: R1 + R2.
Same rule as round 2: adopt only if margin MAE beats A in at least 2 of the 3 decision
seasons (2021-2023) and the paired bootstrap 95% interval of the gain excludes zero.
2024-2025 are a check only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from canes_cfb import features as fx
from canes_cfb.betting import to_games
from canes_cfb.modeling import FEATURES, SPECS
from canes_cfb.paths import PROCESSED, RAW, ROOT

SEASONS = range(2021, 2026)
DECIDE = (2021, 2023)
HALF_LIFE_SEASONS = 2.0
rng = np.random.default_rng(7)


def form_features(games: pd.DataFrame) -> pd.DataFrame:
    """Short-memory ratings per (slate, team): points and EPA/play, half-life 30 days."""
    all_tg = fx.team_games(games)
    adv = pd.read_parquet(RAW / "advanced.parquet")
    teams = pd.read_parquet(RAW / "teams.parquet")
    with_stats = fx.attach_advanced(all_tg, adv, teams)
    with_stats.loc[~with_stats["completed"], list(fx.ADJUSTED_STATS)] = np.nan
    out = None
    for target, name in (("points", "form_pts"), ("ppa", "form_ppa")):
        src = all_tg if target == "points" else with_stats
        rr = fx.ridge_ratings(src, alpha=1.0, half_life_days=30, window_days=150, target=target)
        rr = rr.rename(columns={"off": f"{name}_off", "def": f"{name}_def",
                                "intercept": f"{name}_base"}).drop(columns="hfa_pts")  # fmt: skip
        out = rr if out is None else out.merge(rr, on=[*fx.SLATE, "team_id"], how="outer")
    return out


def add_form(base: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    b = base.merge(form, on=[*fx.SLATE, "team_id"], how="left")
    opp = form[[*fx.SLATE, "team_id", "form_pts_def", "form_ppa_def"]].rename(
        columns={"team_id": "opp_id", "form_pts_def": "opp_form_pts_def",
                 "form_ppa_def": "opp_form_ppa_def"}
    )  # fmt: skip
    b = b.merge(opp, on=[*fx.SLATE, "opp_id"], how="left")
    b["exp_points_form"] = b.form_pts_base + b.form_pts_off - b.opp_form_pts_def + b.hfa_pts * b.hfa
    b["momentum_pts"] = b.exp_points_form - b.exp_points
    b["exp_ppa_form"] = b.form_ppa_base + b.form_ppa_off - b.opp_form_ppa_def
    b["momentum_ppa"] = b.exp_ppa_form - b.exp_ppa
    return b


FORM_COLS = ["form_pts_off", "form_pts_def", "opp_form_pts_def", "exp_points_form",
             "momentum_pts", "form_ppa_off", "form_ppa_def", "opp_form_ppa_def",
             "exp_ppa_form", "momentum_ppa"]  # fmt: skip


def predict(base, params, cols, weighted: bool) -> pd.DataFrame:
    spec = SPECS["random_forest"]
    parts = []
    for season in SEASONS:
        train, test = base[base.season < season], base[base.season == season].copy()
        model = spec.build(params["random_forest"]["params"])
        target = (train.points - train.exp_points).to_numpy()
        fit_kw = {}
        if weighted:
            age = season - train.season.to_numpy()
            w = 0.5 ** ((age - 1) / HALF_LIFE_SEASONS)
            key = (
                f"{model.steps[-1][0]}__sample_weight"
                if isinstance(model, Pipeline)
                else "sample_weight"
            )
            fit_kw[key] = w
        model.fit(train[cols], target, **fit_kw)
        test["pred"] = test.exp_points.to_numpy() + model.predict(test[cols])
        parts.append(test)
    return pd.concat(parts)


def to_game(team: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    keep = ["game_id", "team_id", "pred", "season", "season_type", "week", "spread_open",
            "total_open"]  # fmt: skip
    return to_games(team[keep], games)


def metrics(g: pd.DataFrame) -> dict:
    res_s, res_t = np.sign(g.margin + g.spread_open), np.sign(g.total - g.total_open)
    ok_s, ok_t = (res_s != 0) & g.spread_open.notna(), (res_t != 0) & g.total_open.notna()
    wk = g[(g.season_type == 2) & (g.week <= 4)]
    return {
        "margin MAE": (g.margin - g.pred_margin).abs().mean(),
        "total MAE": (g.total - g.pred_total).abs().mean(),
        "wk 0-4 margin MAE": (wk.margin - wk.pred_margin).abs().mean(),
        "winner %": 100 * ((g.pred_margin > 0) == (g.margin > 0)).mean(),
        "ATS open %": 100 * (np.sign(g.pred_margin + g.spread_open)[ok_s] == res_s[ok_s]).mean(),
        "O/U open %": 100 * (np.sign(g.pred_total - g.total_open)[ok_t] == res_t[ok_t]).mean(),
    }


def main() -> None:
    params = json.loads((ROOT / "models" / "team_points_params.json").read_text())
    games = pd.read_parquet(RAW / "games.parquet")
    f = pd.read_parquet(PROCESSED / "team_games.parquet")
    base = f[f.completed & ~f.shortened & f.season.between(2016, 2025)]
    base = add_form(base, form_features(games))
    with_form = FEATURES + FORM_COLS
    cands = {
        "A current": to_game(predict(base, params, FEATURES, False), games),
        "R1 recency weights": to_game(predict(base, params, FEATURES, True), games),
        "R2 form ratings": to_game(predict(base, params, with_form, False), games),
        "R3 R1 + R2": to_game(predict(base, params, with_form, True), games),
    }
    a = cands["A current"].set_index("game_id")
    rows, verdicts = [], {}
    for name, g in cands.items():
        g = g.set_index("game_id").loc[a.index]
        for label, yrs in (("decide 2021-23", DECIDE), ("check 2024-25", (2024, 2025))):
            rows.append({"candidate": name, "period": label, **metrics(g[g.season.between(*yrs)])})
        dec = g.season.between(*DECIDE)
        err_a, err_c = (a.margin - a.pred_margin).abs(), (g.margin - g.pred_margin).abs()
        won = sum(err_a[dec & (a.season == s)].mean() > err_c[dec & (g.season == s)].mean()
                  for s in range(DECIDE[0], DECIDE[1] + 1))  # fmt: skip
        d = (err_a - err_c)[dec].to_numpy()
        boots = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        verdicts[name] = {"seasons better (of 3)": won, "gain 95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                          "adopt": name != "A current" and won >= 2 and lo > 0}  # fmt: skip
    table = pd.DataFrame(rows).set_index(["period", "candidate"]).sort_index().round(2)
    print(table.to_string(), "\n")
    print(pd.DataFrame(verdicts).T.to_string())
    (ROOT / "models" / "round3_recency.json").write_text(json.dumps(
        {"table": table.reset_index().to_dict(orient="records"), "verdicts": verdicts},
        indent=2, default=str))  # fmt: skip


if __name__ == "__main__":
    main()
