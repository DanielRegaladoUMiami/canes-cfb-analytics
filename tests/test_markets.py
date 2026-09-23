from canes_cfb.markets import MARKETS, PERIOD_COMPONENTS, MarketType, Period, iter_notebooks
from canes_cfb.paths import NOTEBOOKS


def test_halves_partition_the_full_game():
    first, second = PERIOD_COMPONENTS[Period.H1], PERIOD_COMPONENTS[Period.H2]
    assert set(first).isdisjoint(second)
    assert set(first) | set(second) == set(PERIOD_COMPONENTS[Period.FULL_GAME])


def test_quarters_exclude_overtime():
    for q in (Period.Q1, Period.Q2, Period.Q3, Period.Q4):
        assert "OT" not in PERIOD_COMPONENTS[q]


def test_registry_scope():
    assert MARKETS[MarketType.WINNER].periods == (Period.FULL_GAME,)
    assert len(MARKETS[MarketType.SPREAD].periods) == 7
    assert len(MARKETS[MarketType.TOTAL].periods) == 7
    assert len(MARKETS[MarketType.TEAM_TOTAL].periods) == 3
    assert len(iter_notebooks()) == 18


def test_every_registered_notebook_exists():
    missing = [
        f"{spec.folder}/{p.value}.ipynb"
        for spec, p in iter_notebooks()
        if not (NOTEBOOKS / spec.folder / f"{p.value}.ipynb").exists()
    ]
    assert not missing, f"run scripts/make_notebooks.py; missing: {missing}"
