# team_total / full_game: v2 CFBD features and first look vs the market

- **Date:** 2026-09-23
- **Notebooks:** 00_data/04_feature_engineering (§6–8), 99_evaluation/01_backtest_all_markets
- **Target:** points per team, FBS vs FBS; games with a closing line (all of them)
- **CV:** walk-forward 2019–2023 (train from 2016)
- **New features:** opponent-adjusted ridge ratings (per slate) for plays, EPA/play, success
  rate, explosiveness, pass EPA, rush EPA; `exp_pace_x_ppa`, `game_pace`,
  `pass_vs_rush_edge`; talent composite and returning production

## Accuracy (CV MAE, points per team)
| | MAE |
|---|---|
| Market: closing line implied points | **8.91** |
| Market: opening line implied points | 8.93 |
| v1 best (score-only) | 9.43 |
| + efficiency | 9.38 |
| + pace × efficiency | 9.38 |
| + priors | **9.30** (LightGBM on ratings residual) |
| LightGBM additive / depth 2 / 15 leaves | 9.30 / 9.30 / 9.42 |

## Betting (derived spread/total, out-of-sample 2019–2023, −110)
| | Win % | CLV |
|---|---|---|
| Spread vs close | 47.3–49.1 | |
| Total vs close | 47.5–50.5 | |
| Spread vs open | 48.4–50.2 | +0.23 → +1.08 pts, rising with edge |
| Total vs open | 50.7–51.7 (≤5-pt edge) | +0.09 → +0.48 pts |

Break-even: 52.4%.

## Verdict
Keep all v2 families except the explicit pace × efficiency term (zero gain; harmless).
Still no interactions to exploit. The model isn't profitable against either line.
Positive, edge-monotonic CLV against openers is the one real signal: the market moves
toward the model. Next: tune (Optuna) + ensemble, and model the market residual
(points vs opening-line implied points) directly.
