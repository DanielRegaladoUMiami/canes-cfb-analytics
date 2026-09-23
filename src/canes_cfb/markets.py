"""Registry of every market this repo models.

One entry per (market type, period) pair. The notebook tree under ``notebooks/`` is
generated from this registry (``scripts/make_notebooks.py``), so adding a market means
adding it here first.

Conventions (see ``docs/markets.md`` for the full rules):
- Margin is always ``home - away``. A home spread line of -7 means home is favored by 7;
  home covers when ``margin + line > 0`` and pushes when it equals 0.
- Quarters are regulation only. ``1H`` = Q1 + Q2. ``2H`` = Q3 + Q4 + OT and
  ``full_game`` includes OT, matching standard US sportsbook rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MarketType(StrEnum):
    WINNER = "winner"
    SPREAD = "spread"
    TOTAL = "total"
    TEAM_TOTAL = "team_total"


class Period(StrEnum):
    FULL_GAME = "full_game"
    H1 = "1H"
    H2 = "2H"
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


# Which quarters each period sums over; "OT" is overtime points.
PERIOD_COMPONENTS: dict[Period, tuple[str, ...]] = {
    Period.FULL_GAME: ("Q1", "Q2", "Q3", "Q4", "OT"),
    Period.H1: ("Q1", "Q2"),
    Period.H2: ("Q3", "Q4", "OT"),
    Period.Q1: ("Q1",),
    Period.Q2: ("Q2",),
    Period.Q3: ("Q3",),
    Period.Q4: ("Q4",),
}

ALL_PERIODS = tuple(Period)
HALVES_AND_GAME = (Period.FULL_GAME, Period.H1, Period.H2)


@dataclass(frozen=True)
class MarketSpec:
    market: MarketType
    folder: str  # notebooks/<folder>/
    periods: tuple[Period, ...]
    target: str  # what the model predicts
    target_kind: str  # "binary" or "continuous"
    line: str  # the market line the prediction is priced against
    metrics: tuple[str, ...]


MARKETS: dict[MarketType, MarketSpec] = {
    MarketType.WINNER: MarketSpec(
        market=MarketType.WINNER,
        folder="01_winner",
        periods=(Period.FULL_GAME,),
        target="home_win: 1 if home team wins the game (incl. OT)",
        target_kind="binary",
        line="moneyline implied probability",
        metrics=("log_loss", "brier", "accuracy", "roi_vs_close"),
    ),
    MarketType.SPREAD: MarketSpec(
        market=MarketType.SPREAD,
        folder="02_spread",
        periods=ALL_PERIODS,
        target="home_margin: home points - away points in the period",
        target_kind="continuous",
        line="home spread (negative = home favored)",
        metrics=("mae", "rmse", "cover_rate", "clv", "roi_vs_close"),
    ),
    MarketType.TOTAL: MarketSpec(
        market=MarketType.TOTAL,
        folder="03_total",
        periods=ALL_PERIODS,
        target="total_points: home points + away points in the period",
        target_kind="continuous",
        line="over/under total",
        metrics=("mae", "rmse", "over_rate", "clv", "roi_vs_close"),
    ),
    MarketType.TEAM_TOTAL: MarketSpec(
        market=MarketType.TEAM_TOTAL,
        folder="04_team_total",
        periods=HALVES_AND_GAME,
        target="team_points: points scored by one team in the period (home and away rows)",
        target_kind="continuous",
        line="team over/under total",
        metrics=("mae", "rmse", "over_rate", "clv", "roi_vs_close"),
    ),
}


def notebook_name(period: Period) -> str:
    return f"{period.value}.ipynb"


def iter_notebooks() -> list[tuple[MarketSpec, Period]]:
    """Every (market, period) pair that gets its own notebook."""
    return [(spec, p) for spec in MARKETS.values() for p in spec.periods]
