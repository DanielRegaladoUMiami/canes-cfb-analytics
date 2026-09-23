"""Fit probability calibration from out-of-sample predictions (2021-2025).

uv run python scripts/calibrate.py   → models/calibration.json
"""

from __future__ import annotations

import json

import pandas as pd

from canes_cfb import calibration
from canes_cfb.paths import PROCESSED, RAW, ROOT


def main() -> None:
    params = json.loads((ROOT / "models" / "team_points_params.json").read_text())
    features = pd.read_parquet(PROCESSED / "team_games.parquet")
    games = pd.read_parquet(RAW / "games.parquet")
    g = calibration.out_of_sample_games(features, games, params["random_forest"]["params"])
    cal = calibration.fit(g)
    calibration.save(cal, ROOT / "models" / "calibration.json")
    print(json.dumps({k: v for k, v in cal.items() if k != "win_reliability"}, indent=2))
    print(pd.DataFrame(cal["win_reliability"]).to_string(index=False))


if __name__ == "__main__":
    main()
