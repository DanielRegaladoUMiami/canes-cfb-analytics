# Team Total

**Target:** team_points: points scored by one team in the period (home and away rows) (continuous)  
**Priced against:** team over/under total  
**Metrics:** mae, rmse, over_rate, clv, roi_vs_close

| Notebook | Period | Points counted | Status |
|---|---|---|---|
| [`full_game.ipynb`](full_game.ipynb) | full_game | Q1 + Q2 + Q3 + Q4 + OT | done: the main points model (tuning, ensemble, 2025 test) |
| [`1H.ipynb`](1H.ipynb) | 1H | Q1 + Q2 | not started |
| [`2H.ipynb`](2H.ipynb) | 2H | Q3 + Q4 + OT | not started |

Conventions: see [`docs/markets.md`](../../docs/markets.md).

| [`periods.ipynb`](periods.ipynb) | 1H, 2H, Q1–Q4 team points: baselines, model, game script in Q4 | done (accuracy; no period lines yet) |
