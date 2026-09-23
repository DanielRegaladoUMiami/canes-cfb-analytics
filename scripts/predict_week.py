"""Predict the next slate of games and print the weekly card.

    uv run python scripts/predict_week.py [--no-refresh]

1. Refresh the current season (ESPN scores + CFBD lines and advanced stats, ~3 CFBD calls).
2. Rebuild the feature table.
3. Refit every base model in the final recipe on all completed games since 2016.
4. Predict both teams' points for the next slate, derive spread and total.
5. Price edges against the opening lines and flag the pre-registered paper bets.

Paper-bet rule (fixed 2026-09-23, before any 2026 prediction was graded): totals only,
bet the side the model likes when |model total - opening total| >= 4 points. It's the
only rule above break-even in both 2024 (validation) and 2025 (test). The sample is
small, so it's tracked on paper through 2026, not bet with money. Spread edges are
listed for tracking only.

Output: data/predictions/<season>_<type>_w<week>.parquet (full card, local) and new paper
bets appended to paper_trading/<season>_bets.csv (committed, so each pick is timestamped
in git before kickoff).
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from canes_cfb import betting, cfbd, espn
from canes_cfb.features import build_all
from canes_cfb.modeling import SPECS, ensemble_predict, fit_predict
from canes_cfb.paths import PREDICTIONS, PROCESSED, RAW, ROOT

SEASON = 2026
PAPER_TOTAL_EDGE = 4.0  # points vs the opening total


def refresh() -> None:
    seasons = list(range(2015, SEASON + 1))
    espn.load_seasons(seasons).to_parquet(RAW / "games.parquet", index=False)
    cfbd.load_lines(seasons, current_season=SEASON).to_parquet(RAW / "lines.parquet", index=False)
    cfbd.load_advanced(seasons, current_season=SEASON).to_parquet(
        RAW / "advanced.parquet", index=False
    )


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
        {m: fit_predict(SPECS[m], params[m]["params"], train, rows) for m in recipe["models"]},
        index=rows.index,
    )
    rows["pred"] = ensemble_predict(base_preds, recipe)

    games = pd.read_parquet(RAW / "games.parquet")
    keep = ["game_id", "team_id", "team", "opp", "pred", "season", "season_type", "week",
            "start_utc", "spread_open", "spread_close", "total_open", "total_close"]  # fmt: skip
    g = betting.to_games(rows[keep], games).rename(columns={"team": "home", "opp": "away"})

    g["spread_edge"] = betting.spread_edge(g, "spread_open")
    g["total_edge"] = betting.total_edge(g, "total_open")
    g["spread_pick"] = g["home"].where(g.spread_edge > 0, g["away"]).where(g.spread_edge.notna())
    g["total_pick"] = (
        (g.total_edge > 0).map({True: "over", False: "under"}).where(g.total_edge.notna())
    )
    g["paper_bet"] = g.total_edge.abs() >= PAPER_TOTAL_EDGE

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

    bets = g[g.paper_bet].sort_values("total_edge", key=abs, ascending=False)
    print(f"\nPaper bets (totals, |edge| >= {PAPER_TOTAL_EDGE} vs opening total): {len(bets)}")
    if len(bets):
        cols = ["away", "home", "total_open", "total_close", "pred_total", "total_edge",
                "total_pick"]  # fmt: skip
        print(bets[cols].round(1).to_string(index=False))
    log_paper_bets(bets)
    print(f"\nsaved {PREDICTIONS / name}")


def log_paper_bets(bets: pd.DataFrame) -> None:
    """Append new paper bets; a game already logged keeps its first (pre-kickoff) pick."""
    path = ROOT / "paper_trading" / f"{SEASON}_bets.csv"
    new = bets.assign(logged_utc=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"))[
        ["logged_utc", "game_id", "season", "week", "start_utc", "away", "home", "total_open",
         "pred_total", "total_edge", "total_pick"]
    ].round(2)  # fmt: skip
    if path.exists():
        old = pd.read_csv(path)
        new = new[~new.game_id.isin(old.game_id)]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


if __name__ == "__main__":
    main()
