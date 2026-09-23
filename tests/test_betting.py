import numpy as np
import pandas as pd
import pytest

from canes_cfb import betting, market


@pytest.fixture
def game():
    # Home 30, away 20. Model: home 28, away 21 -> margin +7, total 49.
    team_rows = pd.DataFrame({"game_id": [1, 1], "team_id": [10, 20], "pred": [28.0, 21.0]})
    games = pd.DataFrame(
        {"game_id": [1], "home_id": [10], "home_points": [30.0], "away_points": [20.0]}
    )
    g = betting.to_games(team_rows, games)
    g["spread_open"], g["spread_close"] = -3.0, -4.5
    g["total_open"], g["total_close"] = 52.0, 51.0
    return g


def test_to_games_pairs_home_and_away(game):
    row = game.iloc[0]
    assert (row.pred_margin, row.pred_total, row.margin, row.total) == (7, 49, 10, 50)


def test_spread_edge_and_result_use_home_line(game):
    # Home -3: model margin 7 -> likes home by 4; actual 10 - 3 > 0 -> home covers.
    assert betting.spread_edge(game, "spread_open").item() == 4
    assert betting.spread_result(game, "spread_open").item() == 1


def test_clv_positive_when_line_moves_our_way(game):
    edge = betting.spread_edge(game, "spread_open")
    assert betting.spread_clv(edge, game).item() == 1.5  # took -3, closed -4.5
    t_edge = betting.total_edge(game, "total_open")  # 49 vs 52 -> under
    assert betting.total_clv(t_edge, game).item() == 1.0  # 52 -> 51, toward under


def test_grade_excludes_pushes_and_prices_at_minus_110():
    edge = pd.Series([1.0, 1.0, -1.0, 1.0])
    result = pd.Series([1, -1, -1, 0])  # win, loss, win, push
    table = betting.grade(edge, result, thresholds=(0,))
    assert table.loc[0, "bets"] == 3
    assert table.loc[0, "win %"] == pytest.approx(200 / 3)
    assert betting.BREAK_EVEN_110 == pytest.approx(0.5238, abs=1e-4)


def test_market_layer_learns_that_edges_win():
    rng = np.random.default_rng(0)
    n = 2000
    edge = rng.normal(0, 4, n)
    # Home covers more often when the model likes home: P = sigmoid(0.15 * edge).
    covers = rng.random(n) < 1 / (1 + np.exp(-0.15 * edge))
    margin = np.where(covers, 10.0, -10.0)
    g = pd.DataFrame(
        {
            "pred_margin": edge + 3.0, "spread_open": -3.0, "spread_close": -3.0,
            "margin": margin + 3.0, "season_type": 2, "week": 8,
        }
    )  # fmt: skip
    model = market.fit(g, "spread")
    p = market.side_probability(model, g, "spread")
    assert p[g.pred_margin > 8].mean() > 0.6 > 0.4 > p[g.pred_margin < -2].mean()
    table = market.grade_by_confidence(p, g, "spread", thresholds=(0.5, 0.6))
    assert table.loc[1, "win %"] > table.loc[0, "win %"] > 50


def test_grade_paper_totals():
    bets = pd.DataFrame(
        {"game_id": [1, 2, 3, 4], "total_open": [50.0, 50.0, 50.0, 50.0],
         "total_pick": ["over", "under", "over", "over"]}
    )  # fmt: skip
    games = pd.DataFrame(
        {"game_id": [1, 2, 3, 4], "completed": [True, True, True, False],
         "home_points": [30.0, 30.0, 25.0, np.nan], "away_points": [24.0, 24.0, 25.0, np.nan]}
    )  # fmt: skip
    lines = pd.DataFrame({"game_id": [1, 2, 3, 4], "total_close": [52.0, 52.0, 49.0, 50.0]})
    graded = betting.grade_paper_totals(bets, games, lines).set_index("game_id")
    assert list(graded.index) == [1, 2, 3]  # game 4 not final yet
    assert list(graded.result) == ["win", "loss", "push"]
    assert list(graded.clv) == [2.0, -2.0, -1.0]  # over 50 -> closed 52 is +2 for the over
    summary = betting.summarize(graded.reset_index())
    assert summary["record"] == "1-1-1"
    assert summary["units"] == round(100 / 110 - 1, 2)
