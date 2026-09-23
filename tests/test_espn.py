import json
from pathlib import Path

import pandas as pd
import pytest

from canes_cfb.espn import REGULAR, flag_fbs, parse_games, season_weeks

FIXTURE = Path(__file__).parent / "fixtures" / "espn_scoreboard_2024_w5.json"


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def games(payload):
    return parse_games(payload).set_index("game_id")


def test_one_row_per_game(games):
    assert len(games) == 3
    assert games.index.is_unique


def test_quarters_and_final_score(games):
    miami_vt = games.loc[401634767]
    assert (miami_vt.home_team, miami_vt.away_team) == ("Miami", "Virginia Tech")
    assert [miami_vt[f"home_q{i}"] for i in range(1, 5)] == [14, 3, 7, 14]
    assert (miami_vt.home_points, miami_vt.away_points) == (38, 34)
    assert miami_vt.home_ot == 0


def test_overtime_goes_to_ot_not_q4(games):
    ot_game = games.loc[401629043]  # Miami (OH) 23-20 UMass in OT
    assert ot_game.home_q4 == 3
    assert ot_game.home_ot == 3
    assert ot_game.home_n_periods == 5


def test_periods_sum_to_final(games):
    for side in ("home", "away"):
        total = sum(games[f"{side}_q{i}"] for i in range(1, 5)) + games[f"{side}_ot"]
        pd.testing.assert_series_equal(total, games[f"{side}_points"], check_names=False)


def test_calendar_has_regular_season_and_postseason(payload):
    weeks = season_weeks(payload)
    assert (REGULAR, 1) in weeks
    assert {season_type for season_type, _ in weeks} == {2, 3}


def test_flag_fbs_by_appearances():
    # Team 1 plays 6 games (FBS); team 99 appears once (FCS opponent).
    rows = [{"season": 2024, "home_id": 1, "away_id": 10 + i} for i in range(5)]
    rows.append({"season": 2024, "home_id": 1, "away_id": 99})
    flagged = flag_fbs(pd.DataFrame(rows))
    assert flagged["home_fbs"].all()
    assert not flagged.loc[flagged.away_id == 99, "away_fbs"].item()


def test_full_games_are_not_shortened(games):
    assert not games["shortened"].any()
