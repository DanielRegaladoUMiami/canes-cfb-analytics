import pandas as pd
import pytest

from canes_cfb import calibration


def test_expected_value_at_break_even_is_zero():
    assert calibration.expected_value(110 / 210) == pytest.approx(0.0, abs=1e-12)
    assert calibration.expected_value(0.55) == pytest.approx(0.55 * 100 / 110 - 0.45)


def test_win_probability_is_symmetric():
    cal = {"sigma_margin": 16.0}
    p = calibration.win_probability([-7.0, 0.0, 7.0], cal)
    assert p[1] == pytest.approx(0.5)
    assert p[0] == pytest.approx(1 - p[2])


def test_probability_uses_every_coefficient():
    params = {"intercept": 0.0, "coef": {"edge": 0.1, "shootout": -1.0}}
    x = pd.DataFrame({"edge": [0.0, 0.0], "shootout": [0.0, 1.0]})
    p = calibration.probability(x, params)
    assert p[0] == pytest.approx(0.5)
    assert p[1] < 0.5
