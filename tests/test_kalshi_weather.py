import numpy as np
import pandas as pd

from canes_cfb import availability, kalshi, ladder


def test_p_over_monotone_and_centered():
    q = ladder.quantiles(np.random.default_rng(0).normal(0, 5, 20000))
    p = ladder.p_over(20.0, np.array([10.5, 19.5, 20.0, 30.5]), q)
    assert np.all(np.diff(p) < 0)
    assert abs(p[2] - 0.5) < 0.02


def test_split_market_adds_up():
    preds = {"1H": 13.0, "2H": 11.0, "Q1": 5.0, "Q2": 7.0, "Q3": 5.0, "Q4": 5.5}
    s = ladder.split_market(preds, 30.0)
    assert np.isclose(s["1H"] + s["2H"], 30.0)
    assert np.isclose(s["Q1"] + s["Q2"] + s["Q3"] + s["Q4"], 30.0 * (1 - ladder.OT_SHARE))
    assert s["1H"] > s["2H"]


def test_market_team_points():
    home, away = ladder.market_team_points(50.0, -7.0)  # home favored by 7
    assert (home, away) == (28.5, 21.5)


def test_kalshi_fee_rounds_up_to_the_cent():
    assert kalshi.fee(0.5) == 0.02  # 0.07 * 0.25 = 0.0175 -> 2 cents
    assert kalshi.fee(0.99) == 0.01


def test_event_teams():
    assert kalshi.event_teams("Oregon vs USC: 1st Half Total") == ("Oregon", "USC")
    assert kalshi.event_teams("no colon here") is None


def test_missed_last_game_flag():
    games = pd.DataFrame({
        "game_id": [1, 2], "completed": True, "home_id": [10, 10], "away_id": [20, 30],
        "start_utc": pd.to_datetime(["2026-09-05", "2026-09-12"], utc=True),
    })  # fmt: skip
    teams = pd.DataFrame({"team": ["A", "B", "C"], "team_id": [10, 20, 30]})
    box = pd.DataFrame({
        "game_id": [1, 1, 2], "team": ["A", "A", "A"], "player_id": ["q", "w", "q2"],
        "player": ["Starter", "Receiver", "Backup"], "pass_att": [30, 0, 25],
        "pass_yds": [250, 0, 180], "rec_yds": [0, 90, 0],
    })  # fmt: skip
    k = availability.key_players(box, games, teams).set_index("player")
    assert bool(k.at["Receiver", "missed_last"])
    assert bool(k.at["Starter", "missed_last"])  # QB1 by attempts, sat out game 2
    assert "Backup" not in k.index  # only the top QB is a key player
