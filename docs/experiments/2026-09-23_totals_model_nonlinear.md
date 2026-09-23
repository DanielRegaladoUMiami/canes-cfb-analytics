# total / full_game: totals model on the nonlinear insights (#10)

- **Date:** 2026-09-23
- **Notebook:** notebooks/03_total/full_game.ipynb; code: src/canes_cfb/totals.py
- **Target:** under vs the closing total (training 2016–N−1); bets graded vs the opening total
- **Caveat:** 2024–25 were seen in the insights analysis, so it's a partially clean check

## Results
| Model | Log loss (close, 19–23) | 2021–23 win % (open) | 2024–25 win % (open) |
|---|---|---|---|
| Logistic, base | 0.693 | 55.3 (870) | 50.5 (475) |
| Logistic, base + nonlinear | 0.694 | 52.7 (1025) | 50.0 (608) |
| LGBM additive, base | 0.693 | 53.1 (591) | 42.3 (222) |
| LGBM additive, base + nonlinear | 0.693 | 53.5 (609) | 44.6 (233) |
| LGBM depth 3, base + nonlinear | 0.696 | 53.5 (1106) | 50.7 (671) |
| **Rule B (shootout → under)** | — | **56.6 (355)** | **56.4 (156)** |
| Always under | 0.693 | 51.8 | 51.7 |

Coin-flip log loss 0.6931; break-even 52.38%. Totals drift down open→close on average
(always-under CLV +0.36/+0.49), and rule B's CLV (−0.14/+0.06) is at or below that drift.

## Rule A vs B conflict (out-of-sample RF predictions, 2021–2025)
- Rule A overall 53.3% (1068); by season 49.5 / 52.0 / 52.0 / 57.7 / 55.2
- Rule A in shootouts: overs 48.5% (99), unders 57.6% (66)
- Conflict games (A over, B under): under won 51.5% (99), so neither side pays

## Verdict
Totals model not adopted: the nonlinear features overfit; rule B's single threshold
captures the effect. Conflicts are skipped; the weekly summary adds a "portfolio (A+B,
no conflicts)" line. Rules A and B stay graded as registered.
