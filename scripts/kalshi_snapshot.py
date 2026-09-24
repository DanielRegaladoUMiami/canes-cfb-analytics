"""Snapshot Kalshi prices for this week's games and price them with the model.

    uv run python scripts/kalshi_snapshot.py

Runs after predict_week.py (Wednesday) and again on Friday, when quarter markets exist.
Saves data/predictions/<season>_<type>_w<week>_kalshi.parquet for the site. The Best and
Lean tiers are logged on paper to paper_trading/<season>_kalshi.csv (first price kept per
market and side) so the season can test them; they are **not** recommended picks: the
backtest found no edge (docs/experiments/2026-09-24_kalshi_periods.md).
"""

from __future__ import annotations

import json

import pandas as pd

from canes_cfb import kalshi
from canes_cfb.paths import PREDICTIONS, RAW, ROOT


def main() -> None:
    latest = sorted(p for p in PREDICTIONS.glob("*_w*.parquet") if "_kalshi" not in p.name)[-1]
    g = pd.read_parquet(latest)
    games = pd.read_parquet(RAW / "games.parquet")
    snap = kalshi.snapshot()
    m = kalshi.match_games(snap, games[games.game_id.isin(g.game_id)])
    cal = json.loads((ROOT / "models" / "period_calibration.json").read_text())
    priced = kalshi.picks(kalshi.price(m, g, cal))
    if priced.empty:
        print("No Kalshi markets matched this week's games.")
        return
    names = g.set_index("game_id")[["home", "away"]]
    priced["subject_name"] = [
        names.at[r.game_id, r.subject] if r.subject in ("home", "away") else None
        for r in priced.itertuples()
    ]
    priced["bet"] = [kalshi.describe(r) for r in priced.itertuples()]
    out = latest.with_name(latest.stem + "_kalshi.parquet")
    priced.to_parquet(out, index=False)

    log = ROOT / "paper_trading" / f"{int(g.season.iloc[0])}_kalshi.csv"
    new = priced[priced.tier.isin(["best", "light"])].assign(
        season=int(g.season.iloc[0]), week=int(g.week.iloc[0])
    )[["season", "week", "game_id", "ticker", "bet", "side", "strike", "period", "stat",
       "subject_name", "price", "cost", "p_side", "edge", "tier", "snapshot_utc"]]  # fmt: skip
    before = 0
    if log.exists():
        old = pd.read_csv(log)
        before = len(old)
        key = old.ticker + "|" + old.side
        new = new[~(new.ticker + "|" + new.side).isin(key)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(log, index=False)
    print(f"{out.name}: {len(priced)} main lines; paper log +{len(new) - before}")
    print(priced.tier.value_counts().to_string())


if __name__ == "__main__":
    main()
