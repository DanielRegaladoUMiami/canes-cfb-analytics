import numpy as np
import pandas as pd
import pytest

from canes_cfb.markets import Period
from canes_cfb.periods import add_period_shares, add_period_targets


@pytest.fixture
def tg():
    # Team 1 always scores everything in Q1 (share 1.0); team 2 spreads evenly.
    rows = []
    for i in range(6):
        start = pd.Timestamp("2024-09-01", tz="UTC") + pd.Timedelta(weeks=i)
        rows += [
            {"game_id": i, "team_id": 1, "opp_id": 2, "start_utc": start, "completed": i < 5,
             "shortened": False, "points": 28.0, "q1": 28.0, "q2": 0.0, "q3": 0.0, "q4": 0.0,
             "ot": 0.0, "exp_points": 28.0},
            {"game_id": i, "team_id": 2, "opp_id": 1, "start_utc": start, "completed": i < 5,
             "shortened": False, "points": 28.0, "q1": 7.0, "q2": 7.0, "q3": 7.0, "q4": 7.0,
             "ot": 0.0, "exp_points": 28.0},
        ]  # fmt: skip
    return pd.DataFrame(rows)


def test_targets_follow_period_rules(tg):
    t = add_period_targets(tg).query("team_id == 2 and completed").iloc[0]
    assert (t["pts_1H"], t["pts_2H"], t["pts_Q2"]) == (14, 14, 7)


def test_targets_missing_for_unplayed_games(tg):
    t = add_period_targets(tg)
    assert t.loc[~t.completed, "pts_Q1"].isna().all()


def test_shares_are_as_of_and_reach_upcoming_games(tg):
    s = add_period_shares(add_period_targets(tg)).set_index(["game_id", "team_id"])
    assert np.isnan(s.loc[(0, 1), "share_Q1"])  # first game: no history
    assert s.loc[(5, 1), "share_Q1"] == pytest.approx(1.0)  # scheduled game gets history
    # Team 2's defense allowed everything in Q1 -> team 1 expects all 28 in Q1.
    assert s.loc[(5, 1), "opp_allowed_share_Q1"] == pytest.approx(1.0)
    assert s.loc[(5, 1), "exp_pts_Q1"] == pytest.approx(28.0)
    assert s.loc[(5, 2), f"share_{Period.Q3.value}"] == pytest.approx(0.25)
