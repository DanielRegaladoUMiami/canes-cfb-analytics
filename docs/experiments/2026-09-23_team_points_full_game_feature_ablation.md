# team_total / full_game: feature ablation (score-only features)

- **Date:** 2026-09-23
- **Notebook:** notebooks/00_data/04_feature_engineering.ipynb
- **Target:** points per team, FBS vs FBS, full game (incl. OT); shortened games excluded
- **CV:** walk-forward, train 2016..N−1 → validate N, N ∈ 2019–2023
- **Features:** raw_form, ratings (ridge, alpha 1, half-life 120 d), elo, matchup, context
- **Models:** Ridge (median impute + missing flags, scaled); LightGBM untuned

## Results (CV MAE, points)
| Features | Ridge | LightGBM |
|---|---|---|
| league average (baseline) | 11.09 | |
| raw_form | 10.02 | 10.06 |
| +ratings | 9.45 | 9.62 |
| +elo | 9.44 | 9.59 |
| +matchup | 9.45 | 9.61 |
| +context | 9.45 | 9.61 |
| ratings formula alone | 9.44 | |
| LGBM additive / depth-2 / residual boosting | | 9.46 / 9.48 / **9.43** |

Residual diagnostics: regression to the mean at the extremes of `exp_points` (+1.5 below 17,
−1.2 above 37). Weeks 1–2 are worst (MAE 10.0 vs 9.2 in weeks 5–8).

## Verdict
Keep ridge ratings as the core feature, and use boosting on the ratings residual as the
modeling pattern. Elo, matchup and context: keep, but they don't earn much yet. Next gains
need CFBD play-level data (pace, unit efficiency, talent) and weather. Not a
model-complexity problem.
