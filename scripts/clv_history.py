"""Closing-line value of the practice picks, 2021-2025 (out of sample).

    uv run python scripts/clv_history.py   → models/clv_history.json

CLV = points the total moved toward the pick between the open (where we bet) and the
close. Positive average CLV means the market later agreed with us: the fastest honest
signal that an edge is real (it needs no game results). Out-of-sample predictions are
cached in data/processed/oos_2021_2025.parquet.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from canes_cfb import calibration
from canes_cfb.paths import PROCESSED, RAW, ROOT

OOS = PROCESSED / "oos_2021_2025.parquet"


def oos_games() -> pd.DataFrame:
    if OOS.exists():
        return pd.read_parquet(OOS)
    params = json.loads((ROOT / "models" / "team_points_params.json").read_text())
    features = pd.read_parquet(PROCESSED / "team_games.parquet")
    games = pd.read_parquet(RAW / "games.parquet")
    g = calibration.out_of_sample_games(features, games, params["random_forest"]["params"])
    g.to_parquet(OOS, index=False)
    return g


def clv_table(g: pd.DataFrame) -> pd.DataFrame:
    g = g[g.total_open.notna() & g.total_close.notna()].copy()
    edge = g.pred_total - g.total_open
    shootout = g.exp_total > calibration.SHOOTOUT_EXP_TOTAL
    rules = {
        "edge4 (model side, |edge| >= 4)": (edge.abs() >= 4, np.sign(edge)),
        "shootout_under": (shootout, pd.Series(-1.0, index=g.index)),
        "all games, model side": (edge != 0, np.sign(edge)),
    }
    rows = []
    for name, (take, side) in rules.items():
        d = g[take]
        s = side[take]
        clv = s * (d.total_close - d.total_open)
        won = s * (d.total - d.total_open)
        rows.append(
            {
                "rule": name,
                "bets": int(len(d)),
                "avg CLV (pts)": round(float(clv.mean()), 2),
                "CLV > 0 %": round(100 * float((clv > 0).mean()), 1),
                "CLV < 0 %": round(100 * float((clv < 0).mean()), 1),
                "win % vs open": round(100 * float((won > 0).sum() / (won != 0).sum()), 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    g = oos_games()
    out = {"all": clv_table(g).to_dict("records")}
    out["by_season"] = {int(s): clv_table(d).to_dict("records") for s, d in g.groupby("season")}
    (ROOT / "models" / "clv_history.json").write_text(json.dumps(out, indent=2) + "\n")
    print(pd.DataFrame(out["all"]).to_string(index=False))
    for s, rows in out["by_season"].items():
        print(
            s, [(r["rule"][:14], r["bets"], r["avg CLV (pts)"], r["win % vs open"]) for r in rows]
        )


if __name__ == "__main__":
    main()
