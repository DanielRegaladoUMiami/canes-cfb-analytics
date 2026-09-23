# Preseason features (issue #11, part 1): transfer portal, recruiting, new coach, preseason AP

- **Date:** 2026-09-23
- **Why:** the report card showed the model trails the market most in weeks 0–2 (+1.7 pts).
- **Features** (`features.preseason_table`, all known before kickoff of week 1):
  portal in/out and net stars (2021+), recruiting class points and 4-year average, new
  head coach (starter hired after Sep 1 of the prior year), preseason AP poll points
  (0 if unranked); team, opponent and differences. Portal entries dated after the season
  start: ≤0.2%.
- **Protocol:** adopt/reject decided on 2021–23 out-of-sample seasons; 2024–25 used only
  as a check (2025 was the one-shot test of the earlier model).

## Margin error, gap vs closing line (points), random forest
| | Before (+priors) | After (+preseason) |
|---|---|---|
| 2021–23 overall gap | 0.628 | **0.590** |
| 2021–23 weeks 0–2 gap | 1.960 | **1.837** |
| 2024–25 overall gap (check) | 0.595 | **0.573** |
| 2024–25 weeks 0–2 gap (check) | 1.321 | **1.110** |
| Winner accuracy, weeks 0–2 (2021–25) | 72.7% | 73.3% |

LightGBM moves the same way (2024–25 weeks 0–2 gap 1.32 → 0.98).

## Verdict
Adopted (`modeling.FEATURES = +preseason`); tuned hyperparameters kept. The early-season
gap narrows but doesn't close: the market still knows more in weeks 0–2 (depth charts,
QB battles, injuries).

## Knock-on effects
- Calibration refit: the 45%-win bin is off by 5.1 pts (2.3 SE). The site check now uses
  a sample-size-aware rule (average miss and z-score) and reports it as "Review".
- Week 4 paper picks were re-logged with the new model before any kickoff (the previous
  log is in git history). Notre Dame @ Purdue became a rule A/B conflict (pass).
