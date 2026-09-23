# team_total / full_game: Optuna, ensemble, market layer, and the 2025 test

- **Date:** 2026-09-23
- **Notebook:** notebooks/04_team_total/full_game.ipynb; tuning: scripts/tune_team_points.py
- **Features:** `+priors` (v1 + v2), 18612 team-game rows
- **Protocol:** Optuna on walk-forward CV 2019–2023 → ensemble on OOF → pick on 2024 →
  2025 test run **once**

## Tuning (per-season-mean CV MAE)
| Model | Defaults | Tuned | Trials |
|---|---|---|---|
| XGBoost (residual) | 9.301 | **9.256** | 40 |
| CatBoost (residual) | 9.274 | 9.257 | 25 |
| LightGBM (residual) | 9.280 | 9.260 | 60 |
| Ridge | 9.333 | 9.272 | 25 (alpha at the edge of the search range, 978 of 1000) |
| Random forest (residual) | 9.314 | 9.294 | 15 |
| Poisson GLM | 9.420 | 9.306 | 20 |

Error correlation between models ≥ 0.98. NNLS stacking weights: ridge 0.45, xgboost
0.33, poisson 0.10, lightgbm 0.06, random forest 0.04, catboost 0.

## Selection and test (MAE)
| | OOF 2019–23 (pooled) | Validation 2024 | Test 2025 |
|---|---|---|---|
| Random forest (**final**, best on 2024) | 9.275 | 9.081 | **8.940** |
| Stacking / average (all) | 9.200 / 9.212 | 9.097 / 9.085 | |
| Market open / close | | 8.869 / 8.861 | 8.696 / 8.594 |

## Betting (2025 test, −110)
| Rule | 2024 | 2025 |
|---|---|---|
| Market layer, spread, p ≥ 0.5 | 50.4% | 49.3% |
| Market layer, total, p ≥ 0.5 | 52.0% | 55.7%* |
| Raw edge vs open, spread ≥ 4 | 45.9% (220) | 55.1% (265) |
| Raw edge vs open, total ≥ 4 | **57.8% (213)** | **55.2% (201)** |
| Avg CLV, total edge ≥ 4 | +0.53 | +1.01 |

\*Market layer, totals: negative edge coefficient; mostly bets unders on high totals. 2025
unders hit 53.7% (57.4% on high totals) vs 49.6–51.1% in 2022–2023, so it's a
base-rate effect, not model skill.

## Verdict
The market is more accurate (by ~0.25 pts/team). The market layer is dropped. Spread
edges are noise. Totals with |edge| ≥ 4 vs the opener is the one rule above break-even
in both held-out years, but it's within ~1 SE and the threshold was chosen post hoc.
CLV is positive and grows with edge in every season. **Pre-registered for paper trading
in 2026; no real money.**
