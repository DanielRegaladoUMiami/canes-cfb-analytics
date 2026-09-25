# Kalshi halves, quarters and team totals: probabilities and a price backtest

**Question.** Kalshi lists what our sportsbook data doesn't: totals and spreads for each
half and quarter, and team totals. Can the model find value there?

## 1. Period projections: model vs "split the market"

Out of sample 2021–2025 (walk-forward), mean absolute error:

| Period | Total: model | Total: market split | Margin: model | Margin: market split |
|---|---:|---:|---:|---:|
| 1H | 8.63 | **8.52** | 9.56 | **9.41** |
| 2H | 8.84 | **8.79** | 8.63 | **8.47** |
| Q1 | 5.67 | **5.65** | 6.27 | **6.21** |
| Q2 | 6.42 | **6.38** | 7.05 | 7.06 |
| Q3 | 5.89 | **5.86** | 6.27 | **6.20** |
| Q4 | 6.44 | **6.43** | 6.26 | **6.23** |

"Market split" = each team's points implied by the opening sportsbook line, split across
halves and quarters by the model's shares (`ladder.split_market`). Two problems it fixes:
the period models predict *medians* (L1 loss), so their quarters summed to ~47 points
against ~53 for the full game; and their level is worse than the market's. Adopted for
pricing Kalshi.

## 2. Probabilities

P(stat > line) from the empirical distribution of out-of-sample errors per period and
stat (`models/period_calibration.json`, `scripts/period_calibration.py`). Fitted on
2021–2023 errors and scored on 2024–2025 lines near each projection: calibration error
0.4–3.2 points across periods and stats.

## 3. Price backtest (pre-registered rules, real Kalshi prices)

Rules fixed in `src/canes_cfb/kalshi.py` before any grading: main line only (price nearest
50¢, bid-ask ≤ 8¢), Lean at model chance ≥ price + fee + 4 points, Best at + 7, skip if the
model and Kalshi differ by more than 15 points. Prices: last hourly candle ≥ 4 hours before
kickoff. 2026 weeks 1–3 (all Kalshi still keeps). `scripts/kalshi_backtest.py` →
`models/kalshi_backtest.json`.

| | Bets | Win % | ROI (after fees) |
|---|---:|---:|---:|
| Best | 127 | 52.8% | +4.3% |
| Lean | 173 | 49.7% | −6.9% |
| **Best + Lean** | **300** | **51.0%** | **−2.3%** |
| Everything else (main lines) | 1,470 | 53.1% | −3.9% |

(Numbers after the weather model was adopted; before it: 299 picks, 50.8%, −2.6%.)

Brier score on all 1,836 main lines: **Kalshi mid 0.2240, model 0.2256** (lower is
better). Kalshi's own prices predicted these outcomes slightly better than we did. The
model said its picks would win 59%; they won 51%.

By market the picks range from +32% (Q4 totals, 13 bets) to −28% (Q3 totals, 32 bets):
with 10–45 bets each, that's noise. Choosing the "good" markets after the fact is exactly
the overfitting the rule forbids.

## Decision

**Not adopted for picks.** Kalshi period markets are priced at least as well as our
information allows. The site shows Kalshi's price next to the model's chance for every
half, quarter and team total (information, not advice), and the Best tier keeps being
logged on paper (`paper_trading/2026_kalshi.csv`) so the 2026 season can confirm or
overturn this with a real sample.

Side finding: quarter markets open on Thursday for Saturday games, so a Wednesday-only run
never sees them. The weekly workflow adds a Friday price snapshot.
