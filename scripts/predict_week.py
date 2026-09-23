"""Predict the next slate of games and print the weekly card.

    uv run python scripts/predict_week.py [--no-refresh] [--no-log]

1. Refresh the current season (ESPN scores + CFBD lines and advanced stats, ~3 CFBD calls).
2. Rebuild the feature table.
3. Refit every base model in the final recipe on all completed games since 2016.
4. Predict both teams' points for the next slate, derive spread and total.
5. Price edges against the opening lines and flag the pre-registered paper bets.

Paper-bet rules (fixed 2026-09-23, before any 2026 prediction was graded), totals only,
tracked on paper through 2026, never real money:
  A. "edge4": bet the side the model likes when |model total - opening total| >= 4.
     Only rule above break-even in both 2024 (validation) and 2025 (test).
  B. "shootout_under": bet the under when the ratings expect a shootout
     (exp_total > 63.5, the top 20% of 2016-2023). Under 55-60% every season 2021-2025;
     56.4% vs the opening total over 2021-2025 (509 games). See
     notebooks/00_data/05_nonlinearity_insights.ipynb.
Spread edges are listed for tracking only.

Output: data/predictions/<season>_<type>_w<week>.parquet (full card, local) and new paper
bets appended to paper_trading/<season>_bets.csv (committed, so each pick is timestamped
in git before kickoff).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from canes_cfb import betting, cfbd, espn
from canes_cfb.features import build_all
from canes_cfb.modeling import (
    FEATURES,
    SPECS,
    ensemble_predict,
    fit_predict,
    models_needed,
)
from canes_cfb.paths import PREDICTIONS, PROCESSED, RAW, ROOT
from canes_cfb.periods import PERIODS, add_period_shares, add_period_targets, fit_predict_period

SEASON = 2026
PAPER_TOTAL_EDGE = 4.0  # rule A: points vs the opening total
SHOOTOUT_EXP_TOTAL = 63.5  # rule B: top 20% of exp_total, 2016-2023


def refresh() -> None:
    seasons = list(range(2015, SEASON + 1))
    RAW.mkdir(parents=True, exist_ok=True)
    espn.load_seasons(seasons).to_parquet(RAW / "games.parquet", index=False)
    cfbd.load_lines(seasons, current_season=SEASON).to_parquet(RAW / "lines.parquet", index=False)
    cfbd.load_advanced(seasons, current_season=SEASON).to_parquet(
        RAW / "advanced.parquet", index=False
    )
    # Season-level priors: cached after the first run, so these cost CFBD calls only once.
    cfbd.load_talent(seasons).to_parquet(RAW / "talent.parquet", index=False)
    cfbd.load_returning(seasons).to_parquet(RAW / "returning.parquet", index=False)
    cfbd.load_teams().to_parquet(RAW / "teams.parquet", index=False)
    # Preseason information (portal, recruiting, coaches, preseason AP poll).
    cfbd.load_portal(list(range(2021, SEASON + 1))).to_parquet(RAW / "portal.parquet", index=False)
    cfbd.load_recruiting(list(range(2011, SEASON + 1))).to_parquet(
        RAW / "recruiting.parquet", index=False
    )
    cfbd.load_coaches(seasons).to_parquet(RAW / "coaches.parquet", index=False)
    cfbd.load_preseason_ap(seasons).to_parquet(RAW / "preseason_ap.parquet", index=False)


def main() -> None:
    if "--no-refresh" not in sys.argv:
        refresh()
    features = build_all(RAW)
    features.to_parquet(PROCESSED / "team_games.parquet", index=False)

    upcoming = features[~features.completed & (features.start_utc >= pd.Timestamp.now(tz="UTC"))]
    if upcoming.empty:
        print("No upcoming FBS-vs-FBS games.")
        return
    slate = upcoming.sort_values("slate_start").iloc[0][["season", "season_type", "week"]]
    rows = upcoming[
        (upcoming.season == slate.season)
        & (upcoming.season_type == slate.season_type)
        & (upcoming.week == slate.week)
    ].copy()

    recipe = json.loads((ROOT / "models" / "team_points_final.json").read_text())
    params = json.loads((ROOT / "models" / "team_points_params.json").read_text())
    train = features[features.completed & ~features.shortened & (features.season >= 2016)]
    base_preds = pd.DataFrame(
        {m: fit_predict(SPECS[m], params[m]["params"], train, rows) for m in models_needed(recipe)},
        index=rows.index,
    )
    rows["pred"] = ensemble_predict(base_preds, recipe)

    # Shadow model (not used for picks): average of all six tuned models. It led in round 2
    # but missed the pre-registered adoption rule, so 2026 decides. See docs/experiments.
    rows["pred_shadow"] = np.mean(
        [fit_predict(SPECS[m], params[m]["params"], train, rows) for m in SPECS], axis=0
    )

    # Halves and quarters (saved to the parquet; no period lines to price them yet, #4).
    with_periods = add_period_shares(add_period_targets(features))
    period_train = with_periods[with_periods.completed & ~with_periods.shortened]
    period_train = period_train[period_train.season >= 2016]
    period_rows = (
        with_periods.set_index(["game_id", "team_id"])
        .loc[rows.set_index(["game_id", "team_id"]).index]
        .reset_index()
    )
    for p in PERIODS:
        rows[f"pred_{p.value}"] = fit_predict_period(
            p, period_train, period_rows, FEATURES, params["lightgbm"]["params"]
        )

    games = pd.read_parquet(RAW / "games.parquet")
    keep = ["game_id", "team_id", "team", "opp", "pred", "season", "season_type", "week",
            "start_utc", "spread_open", "spread_close", "total_open", "total_close",
            "exp_total"]  # fmt: skip
    g = betting.to_games(rows[keep], games).rename(columns={"team": "home", "opp": "away"})
    shadow = betting.to_games(
        rows[["game_id", "team_id", "pred_shadow"]], games, pred="pred_shadow"
    )
    g = g.merge(
        shadow[["game_id", "pred_margin", "pred_total"]].rename(
            columns={"pred_margin": "shadow_margin", "pred_total": "shadow_total"}
        ),
        on="game_id",
        how="left",
    )
    for p in PERIODS:
        side = rows[["game_id", "team_id", f"pred_{p.value}"]].merge(
            games[["game_id", "home_id"]], on="game_id"
        )
        home = side[side.team_id == side.home_id].set_index("game_id")[f"pred_{p.value}"]
        away = side[side.team_id != side.home_id].set_index("game_id")[f"pred_{p.value}"]
        g[f"pred_margin_{p.value}"] = g.game_id.map(home - away)
        g[f"pred_total_{p.value}"] = g.game_id.map(home + away)

    g["spread_edge"] = betting.spread_edge(g, "spread_open")
    g["total_edge"] = betting.total_edge(g, "total_open")
    g["spread_pick"] = g["home"].where(g.spread_edge > 0, g["away"]).where(g.spread_edge.notna())
    g["total_pick"] = (
        (g.total_edge > 0).map({True: "over", False: "under"}).where(g.total_edge.notna())
    )
    rule_a = g.assign(rule="edge4")[g.total_edge.abs() >= PAPER_TOTAL_EDGE]
    rule_b = g.assign(rule="shootout_under", total_pick="under")[
        (g.exp_total > SHOOTOUT_EXP_TOTAL) & g.total_open.notna()
    ]
    bets = pd.concat([rule_a, rule_b], ignore_index=True)

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    name = f"{int(slate.season)}_{int(slate.season_type)}_w{int(slate.week)}.parquet"
    g.to_parquet(PREDICTIONS / name, index=False)

    g["kickoff_et"] = g.start_utc.dt.tz_convert("America/New_York").dt.strftime("%a %m/%d %I:%M%p")
    card = g[["kickoff_et", "away", "home", "pred_away", "pred", "pred_margin", "pred_total",
              "spread_open", "spread_edge", "spread_pick", "total_open", "total_edge",
              "total_pick"]].rename(columns={"pred": "pred_home"})  # fmt: skip
    pd.set_option("display.width", 220)
    print(f"\nSlate: {int(slate.season)} week {int(slate.week)} — {len(g)} FBS-vs-FBS games")
    order = g.start_utc.sort_values().index
    print(card.loc[order].round(1).to_string(index=False))

    print(f"\nPaper bets: {len(bets)} (A: |edge| >= {PAPER_TOTAL_EDGE}; "
          f"B: under when exp_total > {SHOOTOUT_EXP_TOTAL})")  # fmt: skip
    if len(bets):
        cols = ["rule", "away", "home", "total_open", "total_close", "pred_total", "exp_total",
                "total_edge", "total_pick"]  # fmt: skip
        print(bets[cols].round(1).to_string(index=False))
    if "--no-log" not in sys.argv:
        log_paper_bets(bets)
    print(f"\nsaved {PREDICTIONS / name}")


def log_paper_bets(bets: pd.DataFrame) -> None:
    """Append new paper bets; a (game, rule) already logged keeps its first pre-kickoff pick."""
    path = ROOT / "paper_trading" / f"{SEASON}_bets.csv"
    new = bets.assign(logged_utc=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"))[
        ["logged_utc", "rule", "game_id", "season", "week", "start_utc", "away", "home",
         "total_open", "pred_total", "exp_total", "total_edge", "total_pick"]
    ].round(2)  # fmt: skip
    if path.exists():
        old = pd.read_csv(path)
        if "rule" not in old.columns:  # log started before rule B existed: all were rule A
            old.insert(1, "rule", "edge4")
        seen = set(zip(old.game_id, old.rule, strict=True))
        new = new[[k not in seen for k in zip(new.game_id, new.rule, strict=True)]]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


if __name__ == "__main__":
    main()
