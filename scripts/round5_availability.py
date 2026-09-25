"""Round 5: key-player availability (injury proxy). Pre-registered in
docs/experiments/2026-09-24_availability_preregistration.md (committed before running).

    uv run python scripts/round5_availability.py   → models/round5_availability.json

A = current model (with weather); I1 = A + miss_qb, miss_skill, miss_def for both teams.
Primary metric margin MAE; same adoption rule as every round.
"""

from __future__ import annotations

import json

import pandas as pd
from round3_recency import metrics, predict, to_game
from round4_weather_qb import DECIDE, verdict

from canes_cfb.availability import MISS_COLS
from canes_cfb.modeling import FEATURES
from canes_cfb.paths import PROCESSED, RAW, ROOT

I1 = MISS_COLS + [f"opp_{c}" for c in MISS_COLS]


def add_availability(base: pd.DataFrame) -> pd.DataFrame:
    a = pd.read_parquet(PROCESSED / "availability_features.parquet")
    b = base.merge(a, on=["game_id", "team_id"], how="left")
    opp = a.rename(columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in MISS_COLS}})
    return b.merge(opp, on=["game_id", "opp_id"], how="left")


def main() -> None:
    params = json.loads((ROOT / "models" / "team_points_params.json").read_text())
    games = pd.read_parquet(RAW / "games.parquet")
    f = pd.read_parquet(PROCESSED / "team_games.parquet")
    base = add_availability(f[f.completed & ~f.shortened & f.season.between(2016, 2025)])
    preds = {
        "A current": to_game(predict(base, params, FEATURES, False), games),
        "I1 availability": to_game(predict(base, params, FEATURES + I1, False), games),
    }
    a = preds["A current"].set_index("game_id")
    rows = []
    for name, g in preds.items():
        g = g.set_index("game_id").loc[a.index]
        for label, yrs in (("decide 2021-23", DECIDE), ("check 2024-25", (2024, 2025))):
            rows.append({"candidate": name, "period": label, **metrics(g[g.season.between(*yrs)])})
    g = preds["I1 availability"].set_index("game_id").loc[a.index]
    verdicts = {
        "I1 availability": verdict(a, g, "margin", "pred_margin"),
        "I1 (total, info only)": verdict(a, g, "total", "pred_total"),
    }
    table = pd.DataFrame(rows).set_index(["period", "candidate"]).sort_index().round(2)
    print(table.to_string(), "\n")
    print(pd.DataFrame(verdicts).T.to_string())
    (ROOT / "models" / "round5_availability.json").write_text(json.dumps(
        {"table": table.reset_index().to_dict(orient="records"), "verdicts": verdicts},
        indent=2, default=str))  # fmt: skip


if __name__ == "__main__":
    main()
