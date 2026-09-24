# Paper trading log

Pre-registered rules (fixed 2026-09-23), totals only:

- **A. `edge4`**: bet the side the model likes when |model total − opening total| ≥ 4
  (`notebooks/04_team_total/full_game.ipynb`).
- **B. `shootout_under`**: bet the under when the ratings expect a shootout
  (`exp_total` > 63.5; `notebooks/00_data/05_nonlinearity_insights.ipynb`).

The two can disagree on the same game; each is graded on its own.
Priced at −110 (break-even 52.38%). No real money until a full season of paper results
says otherwise.

`scripts/predict_week.py` appends each week's picks to `<season>_bets.csv` before
kickoff. A (game, rule) already logged keeps its first pick. Commit the file after each run, so
git timestamps prove the pick came before the result.

Grading (win %, ROI, CLV vs the closing total) happens after the games; see ROADMAP.

## Every game's prediction

`<season>_predictions.csv` stores the model's prediction for every FBS-vs-FBS game: each
team's points, spread and total for the game, halves and quarters, and the lines. The
website's **Results** section compares it with the final scores.

- `source = live`: saved by `scripts/predict_week.py` before kickoff (week 4 of 2026 on).
- `source = backfill`: weeks 1–3 of 2026, made afterwards by `scripts/backfill_season.py`
  with models refit only on games played before that week. Honest about the model, but
  not timestamped before kickoff. A backfill never overwrites a live prediction.

## Week 4 of 2026 re-logged (2026-09-24, before kickoff)

The weather model (round 4) was adopted before any week-4 game started, so week 4's
picks were logged again with it; the earlier week-4 rows stay in git history.

## Kalshi (information only)

`<season>_kalshi.csv` logs the Best and Lean tiers of Kalshi half/quarter/team-total
markets with the price at the snapshot (first price kept per market and side). They're
**not** recommended: the backtest on 2026 weeks 1–3 lost 2.3% after fees
(`docs/experiments/2026-09-24_kalshi_periods.md`). The log lets the season settle it.
