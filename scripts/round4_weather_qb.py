"""Round 4: weather and quarterback features. Pre-registered in
docs/experiments/2026-09-24_weather_qb_preregistration.md (committed before running).

    uv run python scripts/round4_weather_qb.py   → models/round4_weather_qb.json

Candidates: A (current), W1 (raw weather), W2 (weather thresholds), Q1 (quarterbacks).
Adopt only if the primary metric (total MAE for W, margin MAE for Q) beats A in at least
2 of 3 decision seasons (2021-2023) and the paired bootstrap 95% interval of the gain
excludes zero. 2024-2025 are a check only.

Deviations from the pre-registration: gust_mph is dropped (station data lacks gusts in
95% of games) and indoor is not added (the current model already has it).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from round3_recency import metrics, predict, to_game

from canes_cfb.modeling import FEATURES
from canes_cfb.paths import PROCESSED, RAW, ROOT
from canes_cfb.quarterbacks import QB_COLS

DECIDE = (2021, 2023)
rng = np.random.default_rng(7)

W1 = ["wind_mph", "precip_in", "temp_f"]
W2 = ["wind_15", "wind_over_10", "wet", "cold"]
Q1 = QB_COLS + [f"opp_{c}" for c in QB_COLS]


def add_weather(base: pd.DataFrame) -> pd.DataFrame:
    w = pd.read_parquet(RAW / "weather.parquet").drop(columns="indoor", errors="ignore")
    b = base.merge(w, on="game_id", how="left")
    b["wind_15"] = (b.wind_mph >= 15).astype(float).where(b.wind_mph.notna())
    b["wind_over_10"] = (b.wind_mph - 10).clip(lower=0)
    b["wet"] = (b.precip_in >= 0.1).astype(float).where(b.precip_in.notna())
    b["cold"] = (b.temp_f < 40).astype(float).where(b.temp_f.notna())
    return b


def add_qb(base: pd.DataFrame) -> pd.DataFrame:
    q = pd.read_parquet(PROCESSED / "qb_features.parquet")
    b = base.merge(q, on=["game_id", "team_id"], how="left")
    opp = q.rename(columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in QB_COLS}})
    return b.merge(opp, on=["game_id", "opp_id"], how="left")


def verdict(a: pd.DataFrame, g: pd.DataFrame, target: str, pred: str) -> dict:
    dec = g.season.between(*DECIDE)
    err_a, err_c = (a[target] - a[pred]).abs(), (g[target] - g[pred]).abs()
    won = sum(err_a[dec & (a.season == s)].mean() > err_c[dec & (g.season == s)].mean()
              for s in range(DECIDE[0], DECIDE[1] + 1))  # fmt: skip
    d = (err_a - err_c)[dec].to_numpy()
    boots = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"metric": f"{target} MAE", "seasons better (of 3)": int(won),
            "gain 95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
            "adopt": bool(won >= 2 and lo > 0)}  # fmt: skip


def main() -> None:
    params = json.loads((ROOT / "models" / "team_points_params.json").read_text())
    games = pd.read_parquet(RAW / "games.parquet")
    f = pd.read_parquet(PROCESSED / "team_games.parquet")
    base = f[f.completed & ~f.shortened & f.season.between(2016, 2025)]
    base = add_qb(add_weather(base))
    cands = {
        "A current": (FEATURES, None),
        "W1 weather (raw)": (FEATURES + W1, "total"),
        "W2 weather (thresholds)": (FEATURES + W2, "total"),
        "Q1 quarterbacks": (FEATURES + Q1, "margin"),
    }
    preds = {
        n: to_game(predict(base, params, cols, False), games) for n, (cols, _) in cands.items()
    }
    a = preds["A current"].set_index("game_id")
    rows, verdicts = [], {}
    for name, g in preds.items():
        g = g.set_index("game_id").loc[a.index]
        for label, yrs in (("decide 2021-23", DECIDE), ("check 2024-25", (2024, 2025))):
            rows.append({"candidate": name, "period": label, **metrics(g[g.season.between(*yrs)])})
        target = cands[name][1]
        if target:
            verdicts[name] = verdict(a, g, target, f"pred_{target}")
    table = pd.DataFrame(rows).set_index(["period", "candidate"]).sort_index().round(2)
    print(table.to_string(), "\n")
    print(pd.DataFrame(verdicts).T.to_string())
    (ROOT / "models" / "round4_weather_qb.json").write_text(json.dumps(
        {"table": table.reset_index().to_dict(orient="records"), "verdicts": verdicts},
        indent=2, default=str))  # fmt: skip


if __name__ == "__main__":
    main()
