"""Backfill predictions for the 2026 weeks played before live predictions started.

    uv run python scripts/backfill_season.py

For each finished week, every model is refit on games that kicked off before that week's
first game and the week is predicted exactly as scripts/predict_week.py would have done.
Rows are saved to paper_trading/2026_predictions.csv with source="backfill" and never
replace a live prediction. Honest about what they are: made after the fact, but the model
never saw the games it predicts. Uses the cached data (no API calls).
"""

from __future__ import annotations

import pandas as pd
from predict_week import SEASON, predict_slate, record_predictions

from canes_cfb.paths import PROCESSED


def main() -> None:
    features = pd.read_parquet(PROCESSED / "team_games.parquet")
    season = features[(features.season == SEASON) & (features.season_type == 2)]
    done = season.groupby("week").completed.all()
    for week in done[done].index:
        rows = season[season.week == week].copy()
        g = predict_slate(features, rows)
        record_predictions(g, "backfill")
        print(f"week {week}: {len(g)} games backfilled (trained on games before "
              f"{rows.start_utc.min():%Y-%m-%d %H:%M} UTC)")  # fmt: skip


if __name__ == "__main__":
    main()
