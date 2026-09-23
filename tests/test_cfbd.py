import json
from pathlib import Path

import pytest

from canes_cfb import cfbd

FIXTURES = Path(__file__).parent / "fixtures"


def test_consensus_lines_are_median_across_books():
    payload = json.loads((FIXTURES / "cfbd_lines_2024.json").read_text())
    lines = cfbd.parse_lines(payload).set_index("game_id")
    assert len(lines) == 1  # the game with no books is dropped
    pitt = lines.loc[401635575]  # Pittsburgh (home) -3.5 vs California
    assert pitt.spread_close == -3.5
    assert pitt.total_close == 57.5
    assert pitt.n_books == 3


def test_advanced_stats_flatten_offense_defense_and_splits():
    payload = json.loads((FIXTURES / "cfbd_advanced_2024.json").read_text())
    adv = cfbd.parse_advanced(payload)
    row = adv.iloc[0]
    assert row.team == payload[0]["team"]
    assert row.o_plays == payload[0]["offense"]["plays"]
    assert row.o_pass_successRate == payload[0]["offense"]["passingPlays"]["successRate"]
    assert "d_rush_ppa" in adv.columns


def test_missing_key_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    monkeypatch.setattr(cfbd, "CACHE_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="CFBD_API_KEY"):
        cfbd.get("/lines", year=1900)


def test_cache_path_never_contains_the_key(monkeypatch):
    monkeypatch.setenv("CFBD_API_KEY", "secret-value")
    assert "secret" not in str(cfbd._cache_path("/lines", {"year": 2024}))


def test_clean_moneylines_blanks_swapped_lines():
    import pandas as pd

    lines = pd.DataFrame(
        {"spread_close": [-12.0, -12.0], "home_ml": [-520.0, 375.0], "away_ml": [375.0, -520.0]}
    )
    clean = cfbd.clean_moneylines(lines)
    assert clean.home_ml.iloc[0] == -520  # home favored by 12 and priced as favorite: kept
    assert pd.isna(clean.home_ml.iloc[1])  # home favored by 12 but priced as underdog: swapped
