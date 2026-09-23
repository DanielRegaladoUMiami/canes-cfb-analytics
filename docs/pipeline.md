# ML pipeline

The same pipeline runs for every market. The first target is **points per team, full
game**. Spread (difference), total (sum) and winner (who scores more) all derive from it.
Dedicated spread and total models are built later and must beat the derived version to
earn their place.

```
00_data/01_ingest_games ──┐
00_data/02_ingest_lines ──┼─► 03_eda ─► 04_feature_engineering ─► data/processed/team_games.parquet
                          │                                              │
                          │      each market notebook (04_team_total/full_game first):
                          │      split ─► CV + Optuna per model ─► compare on validation
                          │      ─► ensemble / stacking ─► test once ─► predict this week
                          └──────────────────────────────────────────────► 99_evaluation/
```

## 1. Understand the data (`03_eda`)

Findings (2015–2025, FBS vs FBS):

| | |
|---|---|
| Games | ~8,300 → ~16,600 team-game rows |
| Team points | mean 27.7, SD 13.9 |
| Scoring trend | Game totals down from ~57 (2015) to ~52 (2025); the 2023 clock rule cut plays |
| Home field | +4.3 points at home, +1.0 at neutral sites |
| FCS games | ~1,180, average margin ~31. Excluded from training targets. |
| Quarters | Q1 12.2, Q2 16.3, Q3 12.4, Q4 13.9 points per game. OT in 4.4% of games. |

## 2. Feature engineering (`04_feature_engineering`)

**As-of rule:** a game's features use only games played before kickoff. No exceptions.

| Family | Features | Why |
|---|---|---|
| Raw form | Points scored/allowed, last 3 / last 5 / season to date | Baseline the adjusted features must beat |
| **Opponent-adjusted ratings** | Offense and defense ratings from a ridge regression refit each week: `points = avg + off[team] − def[opp] + home` | Scoring 40 against a bad defense ≠ 40 against a good one. This is the core feature. |
| Matchup | `off[team] − def[opp]`, expected points from the ratings | What the model mostly needs: this offense against this defense |
| Recency | Exponentially weighted versions of the above | Teams change during the season |
| Early-season priors | Last season's final ratings, shrunk toward the mean | Weeks 1–4 have little current data |
| Era | Features relative to the season's league average | Scoring changed over time (see EDA) |
| Context | Home/away/neutral, rest days, travel distance, conference game, week, dome | Known effects |
| Period shares | Historical share of points by quarter/half (as-of) | For half and quarter models |
| Efficiency (with CFBD key) | EPA/PPA per play, success rate, plays per game (tempo), all opponent-adjusted | Tempo and efficiency separate "scores a lot" from "plays fast" |

Every family is added in steps, and each step must improve CV error to stay. That's how
we'll see which features actually made the model better.

## 3. Split

Random 80/20 splits leak the future: the model would train on November and "predict"
September. So the split is **by time**, which is roughly an 80/10/10:

| Set | Seasons | Use |
|---|---|---|
| Train | 2015–2023 | Fit models; CV and Optuna run inside this window only |
| Validation | 2024 | Compare tuned models, choose ensemble weights |
| Test | 2025 | Touched **once**, at the end. The honest final number. |
| Production | 2026 | Weekly predictions |

2020 (COVID, 530 FBS games, empty stadiums) stays in train but is a candidate for
down-weighting. Test that in CV.

**Cross-validation = walk-forward by season** (expanding window):

```
fold 1: train 2015–2018 → validate 2019
fold 2: train 2015–2019 → validate 2020
fold 3: train 2015–2020 → validate 2021
fold 4: train 2015–2021 → validate 2022
fold 5: train 2015–2022 → validate 2023
```

## 4. Models

| Model | Why it's in the lineup |
|---|---|
| Naive: season-average points | Floor. Any model must beat it (~11-point MAE). |
| Ratings only (the ridge ratings as prediction) | Strong, simple baseline, like a power rating |
| Linear regression / Ridge / ElasticNet | Interpretable, hard to overfit with ~16k rows |
| Poisson / Tweedie GLM | Points are non-negative counts; right-skewed |
| Random Forest | Nonlinear, low-tuning reference |
| XGBoost | Gradient boosting, the usual winner on tabular data |
| LightGBM | Faster boosting; different regularization |
| CatBoost | Handles categorical features (conference, venue) natively |

**Tuning:** Optuna (TPE sampler) per model, minimizing mean walk-forward CV MAE (the
metric that maps directly to points). ~100 trials per model.

**Ensemble:** after tuning, compare on validation (2024):
1. Simple average of the top models.
2. Weighted average (weights fit on out-of-fold predictions).
3. Stacking: Ridge meta-model on out-of-fold predictions from the walk-forward CV.

Keep whichever wins on validation; if the ensemble doesn't beat the best single model,
use the single model.

## 5. Metrics

| Metric | For |
|---|---|
| MAE, RMSE | Points prediction accuracy (primary: MAE) |
| MAE vs the closing line's implied points | Does the model know something the market doesn't? (needs CFBD lines) |
| Cover % / over % against the line, ROI at −110 | What matters for betting (break-even at −110 is 52.4%) |
| Calibration | When the model says 60%, does it happen 60% of the time? |

## 6. Weekly prediction

After the final test: refit the chosen model on 2015–2025 (plus 2026 to date), compute
this week's features as-of kickoff, predict points for both teams in each game, derive
spread/total/winner, and write to `data/predictions/`. `99_evaluation/03_weekly_card`
ranks the picks by edge against the current line.
