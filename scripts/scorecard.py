"""Report card for everything we predict, on seasons the model never trained on.

    uv run python scripts/scorecard.py   → models/scorecard.json (+ printed tables)

Walk-forward: each season 2021-2025 is predicted by models trained only on earlier
seasons. Every target is compared with what the sportsbooks implied (closing lines)
where a line exists, and the errors are broken down by segment to show where the
model falls furthest behind the market: that's where to improve.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from canes_cfb import calibration
from canes_cfb.modeling import FEATURES
from canes_cfb.paths import PROCESSED, RAW, ROOT
from canes_cfb.periods import PERIODS, add_period_shares, add_period_targets, fit_predict_period

SEASONS = range(2021, 2026)


def devig(home_ml: pd.Series, away_ml: pd.Series) -> pd.Series:
    def implied(ml):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))

    h, a = implied(home_ml.to_numpy(float)), implied(away_ml.to_numpy(float))
    return pd.Series(h / (h + a), index=home_ml.index)


def winner_metrics(y: pd.Series, p: pd.Series) -> dict:
    return {
        "accuracy": float(((p > 0.5) == (y == 1)).mean()),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p.clip(1e-4, 1 - 1e-4))),
    }


def side_rate(pick: pd.Series, result: pd.Series) -> dict:
    ok = (result != 0) & result.notna() & pick.notna() & (pick != 0)
    return {"win_rate": float((np.sign(pick[ok]) == result[ok]).mean()), "n": int(ok.sum())}


def main() -> None:
    params = json.loads((ROOT / "models" / "team_points_params.json").read_text())
    cal = json.loads((ROOT / "models" / "calibration.json").read_text())
    features = pd.read_parquet(PROCESSED / "team_games.parquet")
    games = pd.read_parquet(RAW / "games.parquet")
    lines = pd.read_parquet(RAW / "lines.parquet")[["game_id", "home_ml", "away_ml"]]

    g = calibration.out_of_sample_games(features, games, params["random_forest"]["params"])
    g = g.merge(lines, on="game_id", how="left")
    g = g[g.margin != 0]  # college football has no ties; guards bad rows
    g["home_won"] = (g.margin > 0).astype(int)
    g["p_model"] = calibration.win_probability(g.pred_margin, cal)
    g["p_market"] = devig(g.home_ml, g.away_ml)
    g["market_margin"] = -g.spread_close
    g["market_home_pts"] = g.total_close / 2 - g.spread_close / 2
    g["market_away_pts"] = g.total_close / 2 + g.spread_close / 2

    card: dict = {"seasons": f"{min(SEASONS)}-{max(SEASONS)}", "games": int(len(g))}

    # Winner
    both = g[g.p_market.notna()]
    card["winner"] = {
        "games": int(len(both)),
        "model": winner_metrics(both.home_won, both.p_model),
        "market": winner_metrics(both.home_won, both.p_market),
    }

    # Spread / margin
    card["spread"] = {
        "mae_margin_model": float((g.margin - g.pred_margin).abs().mean()),
        "mae_margin_market": float((g.margin - g.market_margin).abs().mean()),
        "ats_vs_close": side_rate(
            g.pred_margin + g.spread_close, np.sign(g.margin + g.spread_close)
        ),
        "ats_vs_open": side_rate(g.pred_margin + g.spread_open, np.sign(g.margin + g.spread_open)),
    }

    # Game total
    card["total"] = {
        "mae_model": float((g.total - g.pred_total).abs().mean()),
        "mae_market": float((g.total - g.total_close).abs().mean()),
        "ou_vs_close": side_rate(g.pred_total - g.total_close, np.sign(g.total - g.total_close)),
        "ou_vs_open": side_rate(g.pred_total - g.total_open, np.sign(g.total - g.total_open)),
    }

    # Team totals (each team's points)
    team_err = pd.concat([(g.home_points - g.pred).abs(), (g.away_points - g.pred_away).abs()])
    mkt_err = pd.concat(
        [(g.home_points - g.market_home_pts).abs(), (g.away_points - g.market_away_pts).abs()]
    )
    card["team_total"] = {"mae_model": float(team_err.mean()), "mae_market": float(mkt_err.mean())}

    # Halves and quarters: walk-forward, vs a constant (no sportsbook lines yet)
    f = add_period_shares(add_period_targets(features))
    base = f[f.completed & ~f.shortened & f.season.between(2016, 2025)]
    lgb_params = params["lightgbm"]["params"]
    periods = {}
    for p in PERIODS:
        target = f"pts_{p.value}"
        errs, const = [], []
        for season in SEASONS:
            train, test = base[base.season < season], base[base.season == season]
            pred = fit_predict_period(p, train, test, FEATURES, lgb_params)
            errs.append(np.abs(test[target] - pred))
            const.append(np.abs(test[target] - train[target].mean()))
        periods[p.value] = {
            "mae_model": float(pd.concat(errs).mean()),
            "mae_constant": float(pd.concat(const).mean()),
        }
    card["periods"] = periods

    # Where the model falls behind the market (margin error), by segment
    g["abs_spread"] = g.spread_close.abs()
    g["err_model"] = (g.margin - g.pred_margin).abs()
    g["err_market"] = (g.margin - g.market_margin).abs()
    g["model_right"] = ((g.pred_margin > 0) == (g.home_won == 1)).astype(float)
    g["market_right"] = ((g.market_margin > 0) == (g.home_won == 1)).astype(float)
    g["week_bucket"] = pd.cut(
        g.week.where(g.season_type == 2, 99), [0, 2, 4, 8, 20, 100],
        labels=["Weeks 0-2", "Weeks 3-4", "Weeks 5-8", "Weeks 9+", "Bowls/CFP"],
    )  # fmt: skip
    g["spread_bucket"] = pd.cut(
        g.abs_spread, [-0.1, 3, 7, 14, 21, 80],
        labels=["Pick'em to 3", "3.5 to 7", "7.5 to 14", "14.5 to 21", "21.5+"],
    )  # fmt: skip
    segments = {}
    for col in ("week_bucket", "spread_bucket", "season"):
        seg = g.groupby(col, observed=True).agg(
            games=("margin", "size"),
            model=("err_model", "mean"),
            market=("err_market", "mean"),
            model_acc=("model_right", "mean"),
            market_acc=("market_right", "mean"),
        )  # fmt: skip
        seg["gap"] = seg.model - seg.market
        segments[col] = {
            str(k): {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in seg.iterrows()
        }
    card["segments"] = segments

    (ROOT / "models" / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    w = card["winner"]
    print(f"WINNER ({w['games']} games): model {w['model']} | market {w['market']}")
    print("SPREAD:", card["spread"])
    print("TOTAL:", card["total"])
    print("TEAM TOTAL:", card["team_total"])
    print(
        "PERIODS:",
        {k: (round(v["mae_model"], 2), round(v["mae_constant"], 2)) for k, v in periods.items()},
    )
    for col, rows in segments.items():
        print(f"\n{col}:")
        print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
