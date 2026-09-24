"""Build the weekly card website: every prediction for the latest predicted slate, honest
probabilities, expected value, and a verification report.

    uv run python scripts/build_site.py      → site/index.html

Reads data/predictions/<latest>.parquet (from predict_week.py), models/calibration.json
(from calibrate.py), the paper-bet log and the raw lines/games. The page template is
site/template.html; this script only injects data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from canes_cfb import availability, betting, calibration, cfbd
from canes_cfb.paths import PREDICTIONS, RAW, ROOT
from canes_cfb.periods import PERIODS

SITE = ROOT / "site"
BREAK_EVEN = betting.BREAK_EVEN_110


def devig(home_ml: pd.Series, away_ml: pd.Series) -> pd.Series:
    """Market's home win probability from American moneylines, vig removed."""

    def implied(ml):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))

    h, a = implied(home_ml.to_numpy(float)), implied(away_ml.to_numpy(float))
    return pd.Series(h / (h + a), index=home_ml.index)


# Tiers by win chance, set on the whole-number % the page shows (break-even at -110: 52.4%)
BEST_P = 0.545  # shows 55%+ -> Best Bet (expected value +4% or more)
LEAN_P = 0.535  # shows 54%  -> Lean (+2% to +4%); 53% or less is too close to break-even


def verdict(row, conflict: bool) -> tuple[str, str | None, str]:
    """(tier, bet label, plain-English why) for a game's total. Spreads are never picked:
    the model's spread edge hasn't beaten the opening line historically."""
    per100 = round(100 * row.total_p_side)
    model = int(np.floor(round(row.pred_total, 1) + 0.5))
    line = f"{row.total_open:g}"
    if conflict:
        return (
            "pass",
            None,
            f"Two signals disagree. The model projects {model} points (over {line}), "
            f"but this looks like a shootout, and shootouts tend to go under. When that "
            f"happens, it's been a coin flip. Pass.",
        )
    if row.total_p_side >= LEAN_P:
        tier = "best" if row.total_p_side >= BEST_P else "light"
        label = f"{row.total_side.capitalize()} {line}"
        if row.shootout and row.total_side == "under":
            why = (
                f"Games that look like shootouts (ratings project {row.exp_total:.0f} points) "
                f"have gone under {per100}% of the time since 2021."
            )
        else:
            why = (
                f"The model projects {model} points against a line of {line}. "
                f"Bets like this have won {per100}% of the time since 2021."
            )
        return tier, label, why
    return (
        "pass",
        None,
        f"Not enough edge. To beat the sportsbook's cut you need to win 52.4%, "
        f"and this one sits around {per100}%.",
    )


def split_whole(total: int, weights: list[float]) -> list[int]:
    """Whole numbers proportional to `weights` that add up exactly to `total`
    (largest-remainder rounding)."""
    w = np.clip(np.asarray(weights, dtype=float), 0, None)
    raw = total * (w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w)))
    out = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - out))[: total - out.sum()]:
        out[i] += 1
    return out.tolist()


def period_points(row) -> dict[str, dict[str, int]]:
    """Projected points per team for the game, halves and quarters, as whole numbers that
    add up: Q1+Q2 = 1H, Q3+Q4 = 2H, 1H+2H = full game, home+away = the rounded model total.

    The period models are fitted separately, so their raw outputs don't sum exactly; they
    are used here only for how each team's points split across the game."""

    def pts(p: str, sign: int) -> float:
        t, m = getattr(row, f"pred_total_{p}"), getattr(row, f"pred_margin_{p}")
        return (t + sign * m) / 2

    total = int(np.floor(round(row.pred_total, 1) + 0.5))
    game = dict(zip(("home", "away"), split_whole(total, [row.pred, row.pred_away]), strict=True))
    out: dict[str, dict[str, int]] = {}
    for side, sign in (("home", 1), ("away", -1)):
        h1, h2 = split_whole(game[side], [pts("1H", sign), pts("2H", sign)])
        q1, q2 = split_whole(h1, [pts("Q1", sign), pts("Q2", sign)])
        q3, q4 = split_whole(h2, [pts("Q3", sign), pts("Q4", sign)])
        for p, v in (
            ("Game", game[side]),
            ("1H", h1),
            ("2H", h2),
            ("Q1", q1),
            ("Q2", q2),
            ("Q3", q3),
            ("Q4", q4),
        ):
            out.setdefault(p, {})[side] = v
    return out


def team_meta(season: int, week: int) -> dict[int, dict]:
    """Abbreviation, colors, AP rank and record per team, from the cached ESPN scoreboard."""
    path = RAW / "espn" / "scoreboard" / f"{season}_2_{week}.json"
    meta: dict[int, dict] = {}
    if not path.exists():
        return meta
    for event in json.loads(path.read_text()).get("events", []):
        for c in event["competitions"][0]["competitors"]:
            t = c["team"]
            rank = (c.get("curatedRank") or {}).get("current", 99)
            records = [r.get("summary") for r in c.get("records", []) if r.get("type") == "total"]
            meta[int(t["id"])] = {
                "abbr": t.get("abbreviation"),
                "color": f"#{t['color']}" if t.get("color") else None,
                "alt": f"#{t['alternateColor']}" if t.get("alternateColor") else None,
                "rank": rank if rank <= 25 else None,
                "record": records[0] if records else None,
            }
    return meta


def add_probabilities(g: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Win/cover/over chances (calibrated on out-of-sample 2021-2025), each market's side
    and its expected value at -110. Needs pred_margin, pred_total, spread/total_open,
    exp_total and, for the market win chance, home_ml/away_ml."""
    g = g.copy()
    g["spread_edge"] = betting.spread_edge(g, "spread_open")
    g["total_edge"] = betting.total_edge(g, "total_open")
    # Probabilities (calibrated on out-of-sample 2021-2025) and expected value at -110.
    g["p_home_win"] = calibration.win_probability(g.pred_margin, cal)
    if "home_ml" in g:
        g["p_home_win_market"] = devig(g.home_ml, g.away_ml)
    g["p_home_cover"] = calibration.probability(calibration.spread_inputs(g), cal["spread_vs_open"])
    g["p_over"] = calibration.probability(calibration.total_inputs(g), cal["total_vs_open"])
    g["shootout"] = g.exp_total > calibration.SHOOTOUT_EXP_TOTAL
    # Spread: the model's side (by edge). Its historical probability is ~50% either way,
    # because the spread edge vs the opener hasn't predicted covers (see calibration).
    likes_home = g.spread_edge > 0
    g["spread_side"] = np.where(likes_home, g.home, g.away)
    g["spread_p_side"] = np.where(likes_home, g.p_home_cover, 1 - g.p_home_cover)
    # Totals: the model's side by edge, and the side that has historically won more often
    # given this edge and the shootout flag (the two can differ: unders win more often).
    g["total_model_side"] = np.where(g.total_edge > 0, "over", "under")
    g["total_side"] = np.where(g.p_over >= 0.5, "over", "under")
    g["total_p_side"] = np.maximum(g.p_over, 1 - g.p_over)
    for kind in ("spread", "total"):
        g[f"{kind}_ev"] = calibration.expected_value(g[f"{kind}_p_side"])
    return g


def grade(pick_side: str | None, line: float, actual: float) -> str | None:
    """'win' / 'loss' / 'push' for a pick ('over'/'under' or 'home'/'away' as +1/-1 sides)."""
    if pick_side is None or pd.isna(line):
        return None
    if actual == line:
        return "push"
    return "win" if (actual > line) == (pick_side in ("over", "home")) else "loss"


def past_results(season: int, cal: dict, games: pd.DataFrame) -> list[dict]:
    """Every finished game with a saved prediction: what the model said vs what happened.

    Reads paper_trading/<season>_predictions.csv (live rows and backfilled weeks)."""
    path = ROOT / "paper_trading" / f"{season}_predictions.csv"
    if not path.exists():
        return []
    h = pd.read_csv(path, parse_dates=["start_utc"])
    cols = ["game_id", "home_id", "away_id", "completed", "shortened", "home_points",
            "away_points", *[f"{s}_{q}" for s in ("home", "away")
                             for q in ("q1", "q2", "q3", "q4", "ot")]]  # fmt: skip
    h = h.merge(games[cols], on="game_id", how="left")
    h = h[h.completed.fillna(False).astype(bool)]
    if h.empty:
        return []
    h = add_probabilities(h, cal)
    metas = {w: team_meta(season, int(w)) for w in h.week.unique()}

    out = []
    for row in h.sort_values(["week", "start_utc"]).itertuples():
        # the practice picks exactly as the weekly card would have shown them
        rule_a = row.total_model_side if abs(row.total_edge) >= 4 else None
        rule_b = "under" if row.shootout and not pd.isna(row.total_open) else None
        conflict = bool(rule_a and rule_b and rule_a != rule_b)
        tier, bet, _ = verdict(row, conflict)
        total, margin = row.home_points + row.away_points, row.home_points - row.away_points

        actual = {"Game": {"home": int(row.home_points), "away": int(row.away_points)}}
        if not row.shortened:
            for side in ("home", "away"):
                q = [int(getattr(row, f"{side}_q{i}")) for i in range(1, 5)]
                ot = int(getattr(row, f"{side}_ot") or 0)
                for k, v in (("1H", q[0] + q[1]), ("2H", q[2] + q[3] + ot), ("Q1", q[0]),
                             ("Q2", q[1]), ("Q3", q[2]), ("Q4", q[3])):  # fmt: skip
                    actual.setdefault(k, {})[side] = v
        book_margin = -row.spread_close if not pd.isna(row.spread_close) else None
        meta = metas[row.week]

        def fl(x, n=1):
            return None if x is None or pd.isna(x) else round(float(x), n)

        out.append(
            {
                "id": int(row.game_id),
                "week": int(row.week),
                "source": row.source,
                "kickoff": row.start_utc.isoformat(),
                "home": row.home,
                "away": row.away,
                "home_team": meta.get(int(row.home_id), {}),
                "away_team": meta.get(int(row.away_id), {}),
                "pred": period_points(row),
                "actual": actual,
                "pred_margin": fl(row.pred_margin),
                "pred_total": fl(row.pred_total),
                "book_margin": fl(book_margin),
                "book_total": fl(row.total_close),
                "spread_open": fl(row.spread_open, 2),
                "total_open": fl(row.total_open, 2),
                "winner_ok": None
                if row.pred_margin == 0 or margin == 0
                else bool(np.sign(row.pred_margin) == np.sign(margin)),
                "book_winner_ok": None
                if book_margin in (None, 0) or margin == 0
                else bool(np.sign(book_margin) == np.sign(margin)),
                "ats": grade(
                    "home" if row.spread_edge > 0 else "away", -row.spread_open, margin
                ),  # fmt: skip
                "ats_side": row.home if row.spread_edge > 0 else row.away,
                "ou": grade(row.total_model_side, row.total_open, total),
                "tier": tier,
                "bet": bet,
                "bet_result": grade(row.total_side, row.total_open, total) if bet else None,
            }
        )
    return out


def write_rosters(season: int, g: pd.DataFrame, teams: pd.DataFrame) -> None:
    """site/rosters.json: this week's teams, {team_id: [[pos, #, name, class, ht, wt], ...]}
    (loaded by the page only when a Roster tab is opened)."""
    path = RAW / f"roster_{season}.parquet"
    if not path.exists():
        return
    r = pd.read_parquet(path)
    ids = set(g.home_id.astype(int)) | set(g.away_id.astype(int))
    name_to_id = teams[teams.team_id.isin(ids)].drop_duplicates("team").set_index("team").team_id
    r = r[r.team.isin(name_to_id.index)].copy()
    r["team_id"] = r.team.map(name_to_id).astype(int)
    yr = {1: "FR", 2: "SO", 3: "JR", 4: "SR", 5: "GR", 6: "GR"}
    order = ["QB", "RB", "FB", "WR", "TE", "OL", "OT", "OG", "C", "DL", "DE", "DT", "NT",
             "EDGE", "LB", "ILB", "OLB", "DB", "CB", "S", "PK", "P", "LS", "ATH"]  # fmt: skip
    rank = {p: i for i, p in enumerate(order)}
    out = {}
    for tid, d in r.groupby("team_id"):
        d = d.assign(o=d.position.map(rank).fillna(99)).sort_values(["o", "jersey"])
        out[int(tid)] = [
            [x.position if isinstance(x.position, str) else "",
             None if pd.isna(x.jersey) else int(x.jersey),
             " ".join(n for n in (x.firstName, x.lastName) if isinstance(n, str)),
             yr.get(x.year, ""),
             None if pd.isna(x.height) else f"{int(x.height) // 12}-{int(x.height) % 12}",
             None if pd.isna(x.weight) else int(x.weight)]
            for x in d.itertuples()
        ]  # fmt: skip
    (SITE / "rosters.json").write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))


def check(name: str, ok: bool | None, detail: str, warn: bool = False) -> dict:
    status = "pass" if ok else ("warn" if warn or ok is None else "fail")
    return {"name": name, "status": status, "detail": detail}


def main() -> None:
    latest = sorted(p for p in PREDICTIONS.glob("*.parquet") if "_kalshi" not in p.name)[-1]
    g = pd.read_parquet(latest)
    cal = json.loads((ROOT / "models" / "calibration.json").read_text())
    games = pd.read_parquet(RAW / "games.parquet")
    lines = cfbd.clean_moneylines(pd.read_parquet(RAW / "lines.parquet"))[
        ["game_id", "home_ml", "away_ml"]
    ]
    g = g.merge(lines, on="game_id", how="left")
    season, week = int(g.season.iloc[0]), int(g.week.iloc[0])
    g = g.merge(games[["game_id", "home_id", "away_id"]], on="game_id", how="left")
    meta = team_meta(season, week)

    g = add_probabilities(g, cal)

    bets_path = ROOT / "paper_trading" / f"{season}_bets.csv"
    bets = pd.read_csv(bets_path) if bets_path.exists() else pd.DataFrame(columns=["game_id"])
    bets = bets[bets.get("week", pd.Series(dtype=int)) == week]
    rules = bets.groupby("game_id").apply(
        lambda d: [{"rule": r, "pick": p} for r, p in zip(d.rule, d.total_pick, strict=True)],
        include_groups=False,
    )

    # ---------------------------------------------------------------- verification
    slate = games[(games.season == season) & (games.season_type == 2) & (games.week == week)]
    fbs_slate = slate[slate.home_fbs & slate.away_fbs]
    now = pd.Timestamp.now(tz="UTC")
    corr = g.pred_margin.corr(-g.spread_open)
    tot_gap = (g.pred_total - g.total_open).abs().mean()
    same_fav = (np.sign(g.pred_margin) == np.sign(-g.spread_open))[g.spread_open != 0].mean()
    rel = pd.DataFrame(cal["win_reliability"])
    rel["gap"] = rel.predicted - rel.actual
    rel["z"] = rel.gap / np.sqrt(rel.actual * (1 - rel.actual) / rel.n)
    ece = float((rel.gap.abs() * rel.n).sum() / rel.n.sum())
    worst = rel.loc[rel.z.abs().idxmax()]
    arith = max(
        float((g.pred_margin - (g.pred - g.pred_away)).abs().max()),
        float((g.pred_total - (g.pred + g.pred_away)).abs().max()),
    )
    period_cols = [f"pred_total_{p.value}" for p in PERIODS] + [
        f"pred_margin_{p.value}" for p in PERIODS
    ]
    expect_a = set(g.loc[g.total_edge.abs() >= 4, "game_id"])
    expect_b = set(g.loc[g.shootout & g.total_open.notna(), "game_id"])
    logged_a = set(bets.loc[bets.rule == "edge4", "game_id"])
    logged_b = set(bets.loc[bets.rule == "shootout_under", "game_id"])
    n = len(g)
    n_spread, n_total = int(g.spread_open.notna().sum()), int(g.total_open.notna().sum())
    started = int((g.start_utc <= now).sum())
    checks = [
        check(
            "Every FBS-vs-FBS game this week has a prediction",
            n == len(fbs_slate) and set(g.game_id) == set(fbs_slate.game_id),
            f"{n} of {len(fbs_slate)} predicted for week {week}; "
            f"{len(slate) - len(fbs_slate)} games against FCS teams are out of scope",
        ),
        check(
            "No missing predictions",
            bool(g[["pred", "pred_away", *period_cols]].notna().all().all()),
            "team points and every half and quarter present",
        ),
        check(
            "Spread and total match the projected score",
            arith < 1e-9,
            f"largest arithmetic gap {arith:.2e} points",
        ),
        check(
            "Opening lines available",
            n_spread == n and n_total == n,
            f"spread {n_spread}/{n}, total {n_total}/{n}",
            warn=True,
        ),
        check(
            "Model agrees with the sportsbooks on who's better",
            corr > 0.85 and same_fav > 0.8,
            f"correlation between model margin and the spread {corr:.2f}; "
            f"same favorite in {100 * same_fav:.0f}% of games",
        ),
        check(
            "Model totals are close to the sportsbooks'",
            tot_gap < 6,
            f"average |model total − opening total| = {tot_gap:.1f} points",
        ),
        check(
            "Win probabilities are honest (2021–2025)",
            ece < 0.03 and abs(worst.z) < 2,
            f"average miss {100 * ece:.1f} pts; worst bin: said {100 * worst.predicted:.0f}%, "
            f"happened {100 * worst.actual:.0f}% ({int(worst.n)} games, "
            f"{abs(worst.z):.1f} standard errors)",
            warn=ece < 0.05 and abs(worst.z) < 3,
        ),
        check(
            "Practice-bet log matches the rules",
            expect_a == logged_a and expect_b == logged_b,
            f"rule A {len(logged_a)} (expected {len(expect_a)}), "
            f"rule B {len(logged_b)} (expected {len(expect_b)})",
        ),
        check(
            "Predictions were made before kickoff",
            started == 0,
            f"{started} games had already started when the page was built",
            warn=True,
        ),
    ]

    # ---------------------------------------------------------------- context
    wx = pd.read_parquet(RAW / "weather.parquet") if (RAW / "weather.parquet").exists() else None
    wx = wx.set_index("game_id") if wx is not None else None
    teams = pd.read_parquet(RAW / "teams.parquet")
    box_path = RAW / f"box_{season}.parquet"
    keys = (
        availability.key_players(pd.read_parquet(box_path), games[games.season == season], teams)
        if box_path.exists()
        else pd.DataFrame(columns=["team_id"])
    )
    key_by_team = {int(t): d.drop(columns="team_id").to_dict("records")
                   for t, d in keys.groupby("team_id")}  # fmt: skip
    write_rosters(season, g, teams)
    kal_path = sorted(PREDICTIONS.glob(f"{season}_*_w{week}_kalshi.parquet"))
    kal = pd.read_parquet(kal_path[-1]) if kal_path else pd.DataFrame(columns=["game_id"])
    kal_by_game = {
        int(gid): d.sort_values(["period", "stat"]).apply(
            lambda x: {"market": x.bet, "period": x.period, "stat": x.stat,
                       "kalshi": round(float(x.price), 2), "model": round(float(x.p_side), 3),
                       "edge": round(float(x.edge), 3), "tier": x.tier}, axis=1).tolist()
        for gid, d in kal[kal.tier != "info"].groupby("game_id")
    }  # fmt: skip

    def weather_of(gid: int) -> dict | None:
        if wx is None or gid not in wx.index or pd.isna(wx.at[gid, "wind_mph"]):
            return None
        w = wx.loc[gid]
        return {"wind": round(float(w.wind_mph)), "rain": round(float(w.precip_in), 2),
                "temp": round(float(w.temp_f)), "indoor": bool(w.indoor)}  # fmt: skip

    # ---------------------------------------------------------------- page data
    def r(x, n=1):
        return None if pd.isna(x) else round(float(x), n)

    records = []
    for row in g.sort_values("start_utc").itertuples():
        game_rules = rules.get(row.game_id, [])
        conflict = len({x["pick"] for x in game_rules}) > 1
        tier, bet_label, why = verdict(row, conflict)
        periods = period_points(row)
        records.append(
            {
                "id": int(row.game_id),
                "kickoff": row.start_utc.isoformat(),
                "home": row.home,
                "away": row.away,
                "home_id": int(row.home_id),
                "away_id": int(row.away_id),
                "home_team": meta.get(int(row.home_id), {}),
                "away_team": meta.get(int(row.away_id), {}),
                "pred_home": r(row.pred),
                "pred_away": r(row.pred_away),
                "margin": r(row.pred_margin),
                "total": r(row.pred_total),
                "exp_total": r(row.exp_total),
                "p_home_win": r(row.p_home_win, 3),
                "p_home_win_market": r(row.p_home_win_market, 3),
                "spread_open": r(row.spread_open, 2),
                "spread_now": r(row.spread_close, 2),
                "spread_edge": r(row.spread_edge),
                "spread_side": row.spread_side,
                "spread_p": r(row.spread_p_side, 3),
                "spread_ev": r(row.spread_ev, 3),
                "total_open": r(row.total_open, 2),
                "total_now": r(row.total_close, 2),
                "total_edge": r(row.total_edge),
                "total_side": row.total_side,
                "total_model_side": row.total_model_side,
                "total_p": r(row.total_p_side, 3),
                "total_ev": r(row.total_ev, 3),
                "shootout": bool(row.shootout),
                "rules": rules.get(row.game_id, []),
                "home_ml": r(row.home_ml, 0),
                "away_ml": r(row.away_ml, 0),
                "p_home_cover": r(row.p_home_cover, 3),
                "p_over": r(row.p_over, 3),
                "tier": tier,
                "bet": bet_label,
                "why": why,
                "periods": periods,
                "weather": weather_of(int(row.game_id)),
                "home_keys": key_by_team.get(int(row.home_id), []),
                "away_keys": key_by_team.get(int(row.away_id), []),
                "kalshi": kal_by_game.get(int(row.game_id), []),
            }
        )

    model_card = json.loads((ROOT / "models" / "team_points_final.json").read_text())
    payload = {
        "season": season,
        "week": week,
        "built_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "model": model_card["final"],
        "break_even": BREAK_EVEN,
        "calibration": {
            "seasons": cal["seasons"],
            "games": cal["games"],
            "mae_margin": cal["mae_margin"],
            "mae_total": cal["mae_total"],
            "sigma_margin": cal["sigma_margin"],
            "reliability": cal["win_reliability"],
        },
        "checks": checks,
        "scorecard": json.loads((ROOT / "models" / "scorecard.json").read_text())
        if (ROOT / "models" / "scorecard.json").exists()
        else None,
        "games": records,
        "results": past_results(season, cal, games),
        "clv_history": json.loads((ROOT / "models" / "clv_history.json").read_text())
        if (ROOT / "models" / "clv_history.json").exists()
        else None,
    }
    template = (SITE / "template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
    (SITE / "index.html").write_text(html)
    failed = [c["name"] for c in checks if c["status"] == "fail"]
    graded = payload["results"]
    print(f"site/index.html: week {week}, {len(records)} games, {len(graded)} past games graded, "
          f"checks failed: {failed or 'none'}")  # fmt: skip
    for c in checks:
        print(f"  [{c['status']}] {c['name']}: {c['detail']}")


if __name__ == "__main__":
    main()
