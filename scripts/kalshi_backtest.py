"""Backtest the Kalshi period picks on 2026 weeks already played.

    uv run python scripts/kalshi_backtest.py   → models/kalshi_backtest.json

For every settled Kalshi market on a game we predicted (paper_trading/2026_predictions.csv),
take the market's bid/ask 4 hours before kickoff (hourly candles), run the exact live rules
(kalshi.price + kalshi.picks, pre-registered 2026-09-24) and grade on the settled result.
Also compares how well the model's chances vs Kalshi's mid prices predicted outcomes
(Brier score) on every main line: the direct test of who prices these markets better.
Kalshi keeps only recent history, so this covers what's still available.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from canes_cfb import kalshi
from canes_cfb.paths import RAW, ROOT

HOURS_BEFORE = 4  # price snapshot: last hourly candle ending ≥ 4 h before kickoff
CACHE = RAW / "kalshi" / "backtest_prices.parquet"


def settled_with_prices(hist: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    raw = RAW / "kalshi" / "settled_markets.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        kalshi.snapshot(status="settled").to_parquet(raw, index=False)
    snap = pd.read_parquet(raw)
    m = kalshi.match_games(snap, games[games.game_id.isin(hist.game_id)])
    # only strikes near the model's projection can be the main line; price them
    dummy = m.assign(yes_bid=0.5, yes_ask=0.5, no_bid=0.5, no_ask=0.5)
    near = kalshi.price(dummy, hist, json.loads((ROOT / "models" / "period_calibration.json")
                                                .read_text()))  # fmt: skip
    width = np.where(near.period.isin(["Game", "1H", "2H"]), 6, 4)
    near = near[((near.strike - near.model_mean).abs() <= width) | (near.strike <= 3.5)]
    m = m[m.ticker.isin(near.ticker)].merge(games[["game_id", "start_utc"]], on="game_id")
    rows = []
    for _, d in m.groupby("game_id"):
        ko = d.start_utc.iloc[0]
        cs = kalshi.candles(d.ticker.tolist(), ko - pd.Timedelta(days=5), ko)
        cutoff = (ko - pd.Timedelta(hours=HOURS_BEFORE)).timestamp()
        for t, c in cs.items():
            c = [x for x in c if x["end_period_ts"] <= cutoff]
            if not c:
                continue
            last = c[-1]
            bid = float(last["yes_bid"]["close_dollars"])
            ask = float(last["yes_ask"]["close_dollars"])
            rows.append({"ticker": t, "yes_bid": bid, "yes_ask": ask,
                         "no_bid": 1 - ask, "no_ask": 1 - bid})  # fmt: skip
    prices = pd.DataFrame(rows)
    out = m.drop(columns=["yes_bid", "yes_ask", "no_bid", "no_ask"]).merge(prices, on="ticker")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out


def main() -> None:
    hist = pd.read_csv(ROOT / "paper_trading" / "2026_predictions.csv", parse_dates=["start_utc"])
    games = pd.read_parquet(RAW / "games.parquet")
    done = games[games.completed & ~games.shortened].game_id
    hist = hist[hist.game_id.isin(done)]
    cal = json.loads((ROOT / "models" / "period_calibration.json").read_text())
    m = settled_with_prices(hist, games)
    m = m[m.result.isin(["yes", "no"])]
    d = kalshi.price(m, hist, cal)
    p = kalshi.picks(d)
    p["won"] = p.result == p.side
    p["profit"] = np.where(p.won, 1 - p.cost, -p.cost)  # per contract, after fees

    def summary(x: pd.DataFrame) -> dict:
        staked = x.cost.sum()
        return {"bets": int(len(x)), "won": int(x.won.sum()),
                "win %": round(100 * float(x.won.mean()), 1) if len(x) else None,
                "avg price+fee": round(float(x.cost.mean()), 3) if len(x) else None,
                "avg model chance": round(float(x.p_side.mean()), 3) if len(x) else None,
                "profit per contract": round(float(x.profit.sum()), 2),
                "ROI %": round(100 * float(x.profit.sum() / staked), 1) if staked
                else None}  # fmt: skip

    main_lines = p[p.tier != "info"]
    y = (main_lines.result == "yes").astype(float)
    out = {
        "weeks": sorted(int(w) for w in hist.week.unique()),
        "snapshot": f"{HOURS_BEFORE} hours before kickoff",
        "main_lines": int(len(main_lines)),
        "brier_model": round(float(((main_lines.p_yes - y) ** 2).mean()), 4),
        "brier_kalshi_mid": round(float(((main_lines.mid - y) ** 2).mean()), 4),
        "tiers": {t: summary(p[p.tier == t]) for t in ("best", "light", "pass", "check_news")},
        "picks (best + lean)": summary(p[p.tier.isin(["best", "light"])]),
        "by market": {
            f"{a} {b}": summary(x)
            for (a, b), x in p[p.tier.isin(["best", "light"])].groupby(["period", "stat"])
        },  # fmt: skip
    }
    (ROOT / "models" / "kalshi_backtest.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
