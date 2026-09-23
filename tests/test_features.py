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
FEATURES_V2 = cumulative_sets()["+priors"]


def cfbd_tables(games: pd.DataFrame) -> dict:
    """Synthetic CFBD tables keyed by school name, like the real API."""
    rng = np.random.default_rng(1)
    teams = pd.DataFrame({"team_id": range(N_TEAMS), "team": [f"T{i}" for i in range(N_TEAMS)]})
    adv = [
        {
            "game_id": g.game_id, "team": f"T{tid}", "o_plays": rng.normal(70, 8),
            "o_ppa": rng.normal(0.2, 0.15), "o_successRate": rng.normal(0.42, 0.06),
            "o_explosiveness": rng.normal(1.2, 0.2), "o_pass_ppa": rng.normal(0.25, 0.2),
            "o_rush_ppa": rng.normal(0.12, 0.15),
        }
        for g in games.itertuples()
        for tid in (g.home_id, g.away_id)
    ]  # fmt: skip
    seasons = [(s, f"T{i}") for s in SEASONS for i in range(N_TEAMS)]
    talent = pd.DataFrame(
        [{"season": s, "team": t, "talent": rng.normal(700, 100)} for s, t in seasons]
    )
    returning = pd.DataFrame(
        [
            {"season": s, "team": t, "ret_ppa": rng.random(), "ret_pass_ppa": rng.random(),
             "ret_rush_ppa": rng.random(), "ret_usage": rng.random()}
            for s, t in seasons
        ]
    )  # fmt: skip
    return {"advanced": pd.DataFrame(adv), "talent": talent, "returning": returning, "teams": teams}


def test_hiding_the_future_does_not_change_features(games):
    cutoff = pd.Timestamp("2021-10-01", tz="UTC")
    upcoming_week = games[games.start_utc >= cutoff].sort_values("start_utc").iloc[0]
    slate = (upcoming_week.season, upcoming_week.week)

    hidden = games.copy()
    future = hidden.start_utc >= cutoff
    hidden.loc[future, "completed"] = False
    hidden.loc[future, ["home_points", "away_points"]] = np.nan

    cfbd = cfbd_tables(games)
    adv = cfbd["advanced"]
    hidden_adv = adv[~adv.game_id.isin(hidden.loc[future, "game_id"])]
    full = build_features(games, **cfbd).set_index(["game_id", "team_id"])
    asof = build_features(hidden, **{**cfbd, "advanced": hidden_adv}).set_index(
        ["game_id", "team_id"]
    )

    keys = full[(full.season == slate[0]) & (full.week <= slate[1])].index
    keys = keys[(full.loc[keys, "start_utc"] < cutoff) | (full.loc[keys, "week"] == slate[1])]
    assert full.loc[keys, "exp_ppa"].notna().any()
    pd.testing.assert_frame_equal(
        full.loc[keys, FEATURES_V2], asof.loc[keys, FEATURES_V2], check_exact=False, rtol=1e-9
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
