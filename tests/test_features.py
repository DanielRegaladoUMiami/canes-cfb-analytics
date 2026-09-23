"""The as-of rule, tested directly: hiding the future must not change past features."""

import numpy as np
import pandas as pd
import pytest

from canes_cfb.features import build_features, cumulative_sets

N_TEAMS, WEEKS, SEASONS = 24, 12, (2020, 2021, 2022)


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    strength = rng.normal(0, 7, N_TEAMS)
    rows, gid = [], 0
    for season in SEASONS:
        for week in range(1, WEEKS + 1):
            order = rng.permutation(N_TEAMS)
            start = pd.Timestamp(f"{season}-09-01", tz="UTC") + pd.Timedelta(weeks=week - 1)
            for h, a in order.reshape(-1, 2):
                gid += 1
                hp = max(0, round(27 + strength[h] - strength[a] / 2 + 2 + rng.normal(0, 10)))
                ap = max(0, round(27 + strength[a] - strength[h] / 2 + rng.normal(0, 10)))
                q = {f"{s}_q{i}": np.nan for s in ("home", "away") for i in range(1, 5)}
                rows.append(
                    {
                        "game_id": gid, "season": season, "season_type": 2, "week": week,
                        "start_utc": start + pd.Timedelta(hours=int(rng.integers(0, 48))),
                        "completed": True, "shortened": False,
                        "neutral_site": bool(rng.random() < 0.05),
                        "conference_game": bool(rng.random() < 0.6), "indoor": False,
                        "home_id": int(h), "home_team": f"T{h}", "home_points": float(hp),
                        "away_id": int(a), "away_team": f"T{a}", "away_points": float(ap),
                        "home_fbs": True, "away_fbs": True, "home_ot": 0.0, "away_ot": 0.0,
                        **q,
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


FEATURES = cumulative_sets()["+context"]


def test_hiding_the_future_does_not_change_features(games):
    cutoff = pd.Timestamp("2021-10-01", tz="UTC")
    upcoming_week = games[games.start_utc >= cutoff].sort_values("start_utc").iloc[0]
    slate = (upcoming_week.season, upcoming_week.week)

    hidden = games.copy()
    future = hidden.start_utc >= cutoff
    hidden.loc[future, "completed"] = False
    hidden.loc[future, ["home_points", "away_points"]] = np.nan

    full = build_features(games).set_index(["game_id", "team_id"])
    asof = build_features(hidden).set_index(["game_id", "team_id"])

    keys = full[(full.season == slate[0]) & (full.week <= slate[1])].index
    keys = keys[(full.loc[keys, "start_utc"] < cutoff) | (full.loc[keys, "week"] == slate[1])]
    pd.testing.assert_frame_equal(
        full.loc[keys, FEATURES], asof.loc[keys, FEATURES], check_exact=False, rtol=1e-9
    )


def test_ratings_exist_after_first_season(games):
    f = build_features(games)
    later = f[f.season > SEASONS[0]]
    assert later["exp_points"].notna().all()
    assert later["elo"].notna().all()


def test_ratings_recover_strength_order(games):
    f = build_features(games)
    last = f[f.season == SEASONS[-1]].groupby("team_id")[["off", "elo"]].last()
    assert last["off"].corr(last["elo"]) > 0.5
