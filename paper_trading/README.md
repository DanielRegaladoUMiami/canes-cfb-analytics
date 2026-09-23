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
