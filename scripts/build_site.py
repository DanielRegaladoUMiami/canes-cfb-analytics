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

from canes_cfb import betting, calibration
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


BEST_EV = 0.03  # "buen valor": at least +3% expected value per bet
LIGHT_EV = 0.01  # "valor ligero": +1% to +3%; below that it's noise, so pass


def verdict(row, conflict: bool) -> tuple[str, str | None, str]:
    """(tier, bet label, plain-English why) for a game's total. Spreads are never picked:
    the model's spread edge hasn't beaten the opening line historically."""
    per100 = round(100 * row.total_p_side)
    line = f"{row.total_open:g}"
    if conflict:
        return (
            "pass",
            None,
            f"Two signals disagree. The model projects {row.pred_total:.0f} points (over {line}), "
            f"but this looks like a shootout, and shootouts tend to go under. When that "
            f"happens, it's been a coin flip. Pass.",
        )
    if row.total_ev >= LIGHT_EV:
        tier = "best" if row.total_ev >= BEST_EV else "light"
        label = f"{row.total_side.capitalize()} {line}"
        if row.shootout and row.total_side == "under":
            why = (
                f"Games that look like shootouts (ratings project {row.exp_total:.0f} points) "
                f"have gone under about {per100} of every 100 times since 2021."
            )
        else:
            why = (
                f"The model projects {row.pred_total:.0f} points against a line of {line}. "
                f"Bets like this have won about {per100} of every 100 times since 2021."
            )
        return tier, label, why
    return (
        "pass",
        None,
        f"Not enough edge. To beat the sportsbook's cut you need to win 53 of every 100, "
        f"and this one sits around {per100}.",
    )


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


def check(name: str, ok: bool | None, detail: str, warn: bool = False) -> dict:
    status = "pass" if ok else ("warn" if warn or ok is None else "fail")
    return {"name": name, "status": status, "detail": detail}


def main() -> None:
    latest = sorted(PREDICTIONS.glob("*.parquet"))[-1]
    g = pd.read_parquet(latest)
    cal = json.loads((ROOT / "models" / "calibration.json").read_text())
    games = pd.read_parquet(RAW / "games.parquet")
    lines = pd.read_parquet(RAW / "lines.parquet")[["game_id", "home_ml", "away_ml"]]
    g = g.merge(lines, on="game_id", how="left")
    season, week = int(g.season.iloc[0]), int(g.week.iloc[0])
    g = g.merge(games[["game_id", "home_id", "away_id"]], on="game_id", how="left")
    meta = team_meta(season, week)

    # Probabilities (calibrated on out-of-sample 2021-2025) and expected value at -110.
    g["p_home_win"] = calibration.win_probability(g.pred_margin, cal)
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
    worst_gap = float((rel.predicted - rel.actual).abs().max())
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
            worst_gap < 0.05,
            f"largest gap between predicted and actual win rate: {100 * worst_gap:.1f} points",
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

    # ---------------------------------------------------------------- page data
    def r(x, n=1):
        return None if pd.isna(x) else round(float(x), n)

    records = []
    for row in g.sort_values("start_utc").itertuples():
        game_rules = rules.get(row.game_id, [])
        conflict = len({x["pick"] for x in game_rules}) > 1
        tier, bet_label, why = verdict(row, conflict)
        periods = {
            p.value: {
                "total": r(getattr(row, f"pred_total_{p.value}")),
                "margin": r(getattr(row, f"pred_margin_{p.value}")),
            }
            for p in PERIODS
        }
        records.append(
            {
                "id": int(row.game_id),
                "kickoff": row.start_utc.isoformat(),
                "home": row.home,
                "away": row.away,
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
    }
    template = (SITE / "template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
    (SITE / "index.html").write_text(html)
    failed = [c["name"] for c in checks if c["status"] == "fail"]
    print(f"site/index.html: week {week}, {len(records)} games, checks failed: {failed or 'none'}")
    for c in checks:
        print(f"  [{c['status']}] {c['name']}: {c['detail']}")


if __name__ == "__main__":
    main()
