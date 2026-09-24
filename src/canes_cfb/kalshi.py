"""Kalshi college football prices for the lines sportsbooks don't post in our data:
halves, quarters and team totals (plus full-game ladders for reference).

Public market data, no key: https://api.elections.kalshi.com/trade-api/v2. Each market is
a yes/no contract paying $1; its price in dollars is the market's probability. Kalshi
keeps only recent history, so prices are snapshotted forward (data/raw/kalshi/).
"""

from __future__ import annotations

import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests

API = "https://api.elections.kalshi.com/trade-api/v2"

# series → (period key, stat)
SERIES = {
    "KXNCAAF1HTOTAL": ("1H", "total"),
    "KXNCAAF2HTOTAL": ("2H", "total"),
    "KXNCAAF1QTOTAL": ("Q1", "total"),
    "KXNCAAF2QTOTAL": ("Q2", "total"),
    "KXNCAAF3QTOTAL": ("Q3", "total"),
    "KXNCAAF4QTOTAL": ("Q4", "total"),
    "KXNCAAF1HSPREAD": ("1H", "margin"),
    "KXNCAAF2HSPREAD": ("2H", "margin"),
    "KXNCAAF1QSPREAD": ("Q1", "margin"),
    "KXNCAAF2QSPREAD": ("Q2", "margin"),
    "KXNCAAF3QSPREAD": ("Q3", "margin"),
    "KXNCAAF4QSPREAD": ("Q4", "margin"),
    "KXNCAAFTEAMTOTAL": ("Game", "team"),
    "KXNCAAF1HTEAMTOTAL": ("1H", "team"),
    "KXNCAAFTOTAL": ("Game", "total"),
    "KXNCAAFSPREAD": ("Game", "margin"),
}


def fee(price: float | np.ndarray) -> np.ndarray:
    """Kalshi taker fee per contract: 7% × P × (1 − P), rounded up to the cent."""
    p = np.asarray(price, dtype=float)
    return np.ceil(np.round(0.07 * p * (1 - p) * 100, 6)) / 100


def _get(path: str, params: dict) -> dict:
    for attempt in range(4):
        r = requests.get(f"{API}{path}", params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1 + attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def open_markets(series: str, status: str = "open") -> list[dict]:
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": status, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        d = _get("/markets", params)
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            return out


def open_events(series: str, status: str = "open") -> dict[str, str]:
    """event_ticker → title ('Oregon vs USC: 1st Half Total')."""
    out, cursor = {}, None
    while True:
        params = {"series_ticker": series, "status": status, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        d = _get("/events", params)
        out.update({e["event_ticker"]: e.get("title", "") for e in d.get("events", [])})
        cursor = d.get("cursor")
        if not cursor or not d.get("events"):
            return out


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def snapshot(series: dict = SERIES, status: str = "open") -> pd.DataFrame:
    """One row per market: game title, period, stat, strike, subject team, prices (and the
    result, for settled markets)."""
    rows = []
    now = pd.Timestamp.now(tz="UTC")
    for s, (period, stat) in series.items():
        titles = open_events(s, status)
        for m in open_markets(s, status):
            code = m["ticker"].split("-")[-1]
            team_code = re.sub(r"[\d.]+$", "", code)
            rows.append(
                {
                    "series": s,
                    "event": m["event_ticker"],
                    "ticker": m["ticker"],
                    "title": titles.get(m["event_ticker"], ""),
                    "subtitle": m.get("yes_sub_title", ""),
                    "period": period,
                    "stat": stat,
                    "strike": _f(m.get("floor_strike")),
                    "team_code": team_code if stat in ("team", "margin") else None,
                    "yes_bid": _f(m.get("yes_bid_dollars")),
                    "yes_ask": _f(m.get("yes_ask_dollars")),
                    "no_bid": _f(m.get("no_bid_dollars")),
                    "no_ask": _f(m.get("no_ask_dollars")),
                    "yes_ask_size": _f(m.get("yes_ask_size_fp")),
                    "no_ask_size": _f(m.get("yes_bid_size_fp")),  # a yes bid is a no offer
                    "volume": _f(m.get("volume_fp")),
                    "open_interest": _f(m.get("open_interest_fp")),
                    "result": m.get("result") or None,
                    "snapshot_utc": now,
                }
            )
    return pd.DataFrame(rows)


ALIASES = {  # Kalshi name → ESPN short name, where the two differ
    "miami (fl)": "Miami",
    "appalachian st.": "App State",
    "louisiana-monroe": "UL Monroe",
    "umass": "Massachusetts",
    "louisiana-lafayette": "Louisiana",
    "southern miss": "Southern Miss",
    "hawaii": "Hawai'i",
    "san jose st.": "San José State",
}


def _norm(name: str) -> str:
    name = ALIASES.get(name.strip().lower(), name)
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", "and").replace("state", "st").replace("st.", "st")
    return re.sub(r"[^a-z0-9]", "", s)


def event_teams(title: str) -> tuple[str, str] | None:
    """'Oregon vs USC: 1st Half Total' → ('Oregon', 'USC') — Kalshi lists away first."""
    m = re.match(r"^(.*?)\s+(?:vs\.?|at|@)\s+(.*?):", title)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def match_games(snap: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach game_id (and which side each team code is) by event date and team names.

    ``games`` needs game_id, start_utc, home_team/away_team, home_abbr/away_abbr and the
    ESPN location names; matching tries full names, then abbreviations."""
    ev = snap.drop_duplicates("event")[["event", "title"]].copy()
    ev["date"] = pd.to_datetime(ev.event.str.split("-").str[1].str[:7], format="%y%b%d")
    ev["teams"] = ev.title.map(event_teams)
    g = games.copy()
    g["date"] = g.start_utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    names = {}
    for r in g.itertuples():
        for side in ("home", "away"):
            keys = {_norm(getattr(r, f"{side}_team")), _norm(getattr(r, f"{side}_abbr") or "")}
            for k in keys - {""}:
                names.setdefault((r.date, k), []).append((r.game_id, side))
    out = []
    for e in ev.itertuples():
        if not e.teams:
            continue
        hits = {}
        for kal_side, name in zip(("first", "second"), e.teams, strict=True):
            for d in (e.date, e.date - pd.Timedelta(days=1), e.date + pd.Timedelta(days=1)):
                for gid, side in names.get((d, _norm(name)), []):
                    hits.setdefault(gid, {})[kal_side] = side
        good = [gid for gid, sides in hits.items() if len(set(sides.values())) == 2]
        first_side = hits[good[0]]["first"] if len(good) == 1 else "away"
        if len(good) != 1:  # fallback: ticker code = away abbreviation + home abbreviation
            code = e.event.split("-")[1][7:]
            near = g.date.between(e.date - pd.Timedelta(days=1), e.date + pd.Timedelta(days=1))
            day = g[near]
            pair = day.away_abbr.fillna("") + day.home_abbr.fillna("")
            good = day[pair == code].game_id.tolist()
        if len(good) == 1:
            out.append(
                {
                    "event": e.event,
                    "game_id": good[0],
                    "first_side": first_side,
                    "first_name": e.teams[0],
                    "second_name": e.teams[1],
                }  # fmt: skip
            )
    cols = ["event", "game_id", "first_side", "first_name", "second_name"]
    m = snap.merge(pd.DataFrame(out, columns=cols), on="event", how="inner")
    # which team a team-total or spread market is about, from "Oregon wins 1H by over 7.5"
    first = m.apply(lambda r: r.subtitle.lower().startswith(r.first_name.lower()), axis=1)
    other = np.where(m.first_side == "home", "away", "home")
    m["subject"] = np.where(m.stat == "total", None, np.where(first, m.first_side, other))
    return m.drop(columns=["first_name", "second_name", "first_side"])


def candles(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, list]:
    """Hourly candlesticks (bid/ask), 25 markets per call."""
    out = {}
    for i in range(0, len(tickers), 25):
        d = _get(
            "/markets/candlesticks",
            {
                "market_tickers": ",".join(tickers[i : i + 25]),
                "period_interval": 60,
                "start_ts": int(start.timestamp()),
                "end_ts": int(end.timestamp()),
            },  # fmt: skip
        )
        for m in d.get("markets", []):
            out[m.get("market_ticker") or m.get("ticker")] = m.get("candlesticks", [])
    return out


# ------------------------------------------------------------------ pricing
# Pre-registered 2026-09-24, before any Kalshi pick was graded (docs/experiments/
# 2026-09-24_kalshi_periods.md). Stricter than the sportsbook tiers: unproven market.
LEAN_EDGE = 0.04  # model chance ≥ price + fee + 4 points → Lean (½ unit)
BEST_EDGE = 0.07  # ≥ 7 points → Best Bet (1 unit)
MAX_GAP = 0.15  # model vs Kalshi mid differs by more → likely news the model lacks: skip
MAX_SPREAD = 0.08  # bid-ask wider than 8¢ → too thin to trust the price
PICK_MARKETS = {("Game", "team")} | {
    (p, s) for p in ("1H", "2H", "Q1", "Q2", "Q3", "Q4") for s in ("total", "margin", "team")
}  # full-game total and spread are priced by the sportsbook rules instead


PERIODS = ["1H", "2H", "Q1", "Q2", "Q3", "Q4"]


def period_points(g: pd.DataFrame, period: str) -> tuple[pd.Series, pd.Series]:
    """Model team points (home, away) for a period, before anchoring."""
    if period == "Game":
        return g["pred"], g["pred_away"]
    t, m = g[f"pred_total_{period}"], g[f"pred_margin_{period}"]
    return (t + m) / 2, (t - m) / 2


def price(m: pd.DataFrame, g: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Model chance for every matched Kalshi market and the value of each side.

    ``m``: matched snapshot (match_games). ``g``: the week's game predictions with pred,
    pred_away, pred_total_<p>, pred_margin_<p> and the current sportsbook lines."""
    from canes_cfb import ladder

    g = g.copy()
    g["total_line"] = g.total_close.fillna(g.total_open)
    g["spread_line"] = g.spread_close.fillna(g.spread_open)
    mh, ma = ladder.market_team_points(g.total_line, g.spread_line)
    home_split = ladder.split_market({p: period_points(g, p)[0] for p in PERIODS}, mh)
    away_split = ladder.split_market({p: period_points(g, p)[1] for p in PERIODS}, ma)
    out = []
    for period in ladder.PERIOD_KEYS:
        proj = pd.DataFrame(
            {"game_id": g.game_id, "home_pts": home_split[period], "away_pts": away_split[period]}
        )
        d = m[m.period == period].merge(proj, on="game_id")
        if d.empty:
            continue
        q = cal["quantiles"][period]
        home = d.subject == "home"
        mu = np.select(
            [d.stat == "total", d.stat == "team", d.stat == "margin"],
            [
                d.home_pts + d.away_pts,
                np.where(home, d.home_pts, d.away_pts),
                np.where(home, d.home_pts - d.away_pts, d.away_pts - d.home_pts),
            ],  # fmt: skip
        )
        flip = [-x for x in reversed(q["margin"])]
        p = np.where(d.stat == "total", ladder.p_over(mu, d.strike, q["total"]),
            np.where(d.stat == "team", ladder.p_over(mu, d.strike, q["team"]),
            np.where(home, ladder.p_over(mu, d.strike, q["margin"]),
                     ladder.p_over(mu, d.strike, flip))))  # fmt: skip
        out.append(d.assign(model_mean=mu, p_yes=p))
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    d["mid"] = (d.yes_bid + d.yes_ask) / 2
    d["cost_yes"] = d.yes_ask + fee(d.yes_ask)
    d["cost_no"] = d.no_ask + fee(d.no_ask)
    d["edge_yes"] = d.p_yes - d.cost_yes
    d["edge_no"] = (1 - d.p_yes) - d.cost_no
    return d


def picks(d: pd.DataFrame) -> pd.DataFrame:
    """The main line of each market (price nearest 50¢, two-sided, tight) and its better
    side, tiered by the pre-registered edges."""
    if d.empty:
        return d
    ok = (
        (d.yes_bid > 0) & (d.yes_ask < 1) & (d.no_ask < 1)
        & ((d.yes_ask - d.yes_bid) <= MAX_SPREAD)
    )  # fmt: skip
    d = d[ok].copy()
    d["dist"] = (d.mid - 0.5).abs()
    key = ["game_id", "series", "subject"]
    main = d.sort_values("dist").groupby(key, dropna=False).head(1).copy()
    yes = main.edge_yes >= main.edge_no
    main["side"] = np.where(yes, "yes", "no")
    main["price"] = np.where(yes, main.yes_ask, main.no_ask)
    main["cost"] = np.where(yes, main.cost_yes, main.cost_no)
    main["p_side"] = np.where(yes, main.p_yes, 1 - main.p_yes)
    main["edge"] = main.p_side - main.cost
    main["gap"] = (main.p_yes - main.mid).abs()
    eligible = main.apply(lambda r: (r.period, r.stat) in PICK_MARKETS, axis=1)
    main["tier"] = np.select(
        [~eligible, main.gap > MAX_GAP, main.edge >= BEST_EDGE, main.edge >= LEAN_EDGE],
        ["info", "check_news", "best", "light"],
        default="pass",
    )
    return main.drop(columns="dist")


def describe(r) -> str:
    """Plain-English bet: 'Oregon 1H over 7.5 margin', 'Under 31.5 1H total', ..."""
    per = {"Game": "", "1H": "1st half ", "2H": "2nd half ", "Q1": "1st quarter ",
           "Q2": "2nd quarter ", "Q3": "3rd quarter ", "Q4": "4th quarter "}[r.period]  # fmt: skip
    k = f"{r.strike:g}"
    if r.stat == "total":
        return f"{per}total {'over' if r.side == 'yes' else 'under'} {k}".capitalize()
    team = r.subject_name
    if r.stat == "team":
        return f"{team} {per}points {'over' if r.side == 'yes' else 'under'} {k}"
    if r.side == "yes":
        return f"{team} wins {per}by {k}+".replace("  ", " ")
    return f"{team} doesn't win {per}by {k}+ ".replace("  ", " ").strip()
