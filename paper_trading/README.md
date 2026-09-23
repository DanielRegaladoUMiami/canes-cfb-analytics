# Paper trading log

Pre-registered rule (fixed 2026-09-23, from `notebooks/04_team_total/full_game.ipynb`):
**totals only; bet the side the model likes when |model total − opening total| ≥ 4 points.**
Priced at −110 (break-even 52.38%). No real money until a full season of paper results
says otherwise.

`scripts/predict_week.py` appends each week's picks to `<season>_bets.csv` before
kickoff. A game already logged keeps its first pick. Commit the file after each run, so
git timestamps prove the pick came before the result.

Grading (win %, ROI, CLV vs the closing total) happens after the games; see ROADMAP.
