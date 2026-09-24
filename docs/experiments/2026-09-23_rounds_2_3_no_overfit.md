# Rounds 2 and 3: trying to improve accuracy without overfitting

**Protocol (fixed before running):** a short, pre-registered list of candidates; adopt
only if margin MAE beats the current model in at least 2 of the 3 decision seasons
(2021–2023) **and** the paired bootstrap 95% interval of the gain excludes zero.
2024–2025 are a check only. The live 2026 season is the real test.
Scripts: `scripts/round2_experiments.py`, `scripts/round3_recency.py`
(results in `models/round2_experiments.json`, `models/round3_recency.json`).

## Round 2: variance reduction and dedicated targets
| Candidate | Margin MAE 21–23 | Total MAE 21–23 | O/U vs open 21–23 | Margin MAE 24–25 | O/U vs open 24–25 | Seasons better | Gain 95% CI | Adopted |
|---|---|---|---|---|---|---|---|---|
| A: current (random forest) | 12.82 | 13.03 | 50.9% | 12.53 | 50.3% | — | — | — |
| B: average of the 6 tuned models | 12.78 | 12.89 | 52.2% | 12.49 | 51.6% | 2 | [−0.010, +0.108] | No (interval touches 0) |
| C: dedicated margin and total models | 12.87 | 13.01 | 51.5% | 12.60 | 51.2% | 1 | [−0.155, +0.050] | No |
| D: B and C averaged | 12.79 | 12.91 | 51.4% | 12.50 | 51.6% | 2 | [−0.030, +0.100] | No |
| E: blowouts clipped at ±21 in training | 12.85 | 13.01 | 50.2% | 12.54 | 51.4% | 0 | [−0.055, −0.002] | No (worse) |

B is best in both periods but misses the rule by a hair. It now runs as a **shadow
model** in `scripts/predict_week.py` (stored as `shadow_margin` / `shadow_total`, not
used for picks) so the 2026 season can decide.

## Round 3: making recent information count more
| Candidate | Margin MAE 21–23 | O/U vs open 21–23 | O/U vs open 24–25 | Gain 95% CI | Adopted |
|---|---|---|---|---|---|
| A: current | 12.82 | 50.9% | 50.3% | — | — |
| R1: season recency weights (a season 2 years back counts half) | 12.83 | 49.9% | 52.5% | [−0.044, +0.028] | No |
| R2: 30-day "form" ratings + momentum | 12.83 | 50.4% | 52.2% | [−0.030, +0.026] | No |
| R3: R1 + R2 | 12.86 | 50.2% | 51.9% | [−0.079, +0.003] | No |

R1 and R2 look good on 2024–25 totals but are worse on 2021–23. An effect that flips
between periods is noise; adopting it from 2024–25 alone would have been overfitting.
Recency is already built in at its tuned optimum: the ratings' 120-day half-life beat
60 days, and last-3-games and residual-form features already exist. With about 12 games
per team, short-term form mostly regresses to the mean.

## Conclusion
The remaining gap to the market is **information, not modeling**: starting-QB changes
and injuries, weather for totals, opt-outs for bowls. Next work targets those sources.
