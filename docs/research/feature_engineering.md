# Research: feature engineering for football betting models

Goal: features that let models capture interactions and nonlinear effects. Researched
2026-09-23. Each idea is marked with where it stands in this repo.

## 1. Opponent adjustment via ridge regression (**built**)

Instead of iteratively adjusting offense and defense, solve for both at once:
`stat ~ offense[team] + defense[opponent] + home_field`, with one dummy per team per role
and ridge (L2) regularization to shrink teams with little data. CFBD applies it to EPA
per play, with alpha ~150–200 on play-level data (higher with less data), home = +1,
away = −1, neutral = 0, and FCS games excluded.
([CFBD / RAD blog](https://radsportsanalytics.com/blog/opponent-adjusted-stats-ridge-regression/))

Here: applied to points per game, refit before every slate with recency weights.
Tuned `alpha = 1`, `half_life = 120 days` (game-level data needs far less shrinkage
than play-level). **Result: 11.1 → 9.45 MAE, the biggest single gain.**

## 2. Elo with a nonlinear margin-of-victory multiplier (**built**)

FiveThirtyEight's multiplier is `ln(|MOV| + 1) × 2.2 / (winner_elo_diff × 0.001 + 2.2)`.
The log gives diminishing credit for blowouts (a 40-point win ≈ a 30-point win), and the
second term corrects autocorrelation: favorites win by more, so their Elo would inflate.
K ≈ 20–25, with offseason regression toward the mean.
([538 NFL methodology](https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work),
[538 code](https://github.com/fivethirtyeight/nfl-elo-game),
[autocorrelation in Elo](https://stmorse.github.io/journal/Elo-2.html))

**Result: adds < 0.01 MAE on top of ridge ratings.** It's redundant with them.

## 3. Pace × efficiency is multiplicative (**needs CFBD**)

Points = plays × points per play. Two fast, efficient offenses inflate totals by more than
the sum of their individual effects, and additive models miss it. Models should project
plays (both teams' tempo) and efficiency separately and multiply them, or give the model
both so trees can find the product.
([Boyd's Bets: pace first](https://www.boydsbets.com/handicapping-totals/),
[Establish The Run: snaps and pace](https://establishtherun.com/thorman-why-snaps-and-pace-matter/),
[multivariate GLMM for joint outcomes](https://arxiv.org/pdf/1710.05284))

Plays per game isn't in ESPN's scoreboard. CFBD `/stats/game/advanced` has it.

## 4. Unit-vs-unit matchups (**needs CFBD**)

The real "offense vs. defense" interaction is at the unit level: pass offense vs. pass
defense, rush offense vs. rush defense, explosiveness vs. explosiveness allowed, havoc.
Each is opponent-adjusted with the same ridge method (§1) on EPA/success rate. The
interaction terms are products and differences of the matched units.
([CFBD GBDT spread model](https://radsportsanalytics.com/blog/predicting-spreads-gbdt/),
[modeling tips](https://radsportsanalytics.com/blog/college-football-modeling-tips/))

## 5. Weather thresholds (**needs a weather source**)

Wind hurts passing and kicking with a threshold, not linearly: effects grow above ~10 mph
and are large above 15 mph. In CFB, ~55% of games above 10 mph go under, ~58% above
15 mph and ~60% above 17 mph. That's a step function trees capture naturally.
([Football Study Hall: wind and CFB totals](https://www.footballstudyhall.com/2018/6/25/17500384/football-betting-windy-conditions-effect),
[Advanced Football Analytics: weather and passing](http://www.advancedfootballanalytics.com/2012/01/weather-effects-on-passing.html))

## 6. Talent and returning production as priors (**needs CFBD**)

Recruiting composites and returning production are "sticky" and explain variance that
stats-only models miss, especially early in the season. Our weakest weeks are 1–2
(MAE 10.0 vs 9.2 mid-season).
([modeling tips](https://radsportsanalytics.com/blog/college-football-modeling-tips/))

## 7. Other takeaways

- **Model margin/points, not win/loss.** Regression on points yields spread, total and win
  probability from one model. That's our design.
  ([modeling tips](https://radsportsanalytics.com/blog/college-football-modeling-tips/))
- **Season-to-date aggregation of both teams' stats** is standard; the CFBD GBDT model used
  LightGBM with early stopping, L1/L2 and column sampling. Its spread RMSE was ~15.7.
  ([CFBD GBDT](https://radsportsanalytics.com/blog/predicting-spreads-gbdt/))
- **Where we disagree with sources:** one guide suggests shuffled k-fold CV. For betting
  that leaks the future, so we use walk-forward by season.
- Gradient boosting's edge on tabular sports data comes from interactions and
  nonlinearities. When the features don't contain any, it overfits, which is exactly
  what we measured.
  ([Turtle +EV Labs](https://turtleevlabs.com/blog/sports-betting-probability-models))

## What we measured (score-only data, walk-forward CV 2019–2023)

| | CV MAE |
|---|---|
| League average | 11.09 |
| Raw form (Ridge) | 10.02 |
| Ridge ratings formula | 9.44 |
| + Elo, matchup, context (Ridge) | 9.44–9.45 |
| LightGBM additive (no interactions) | 9.46 |
| LightGBM depth 2 / 15 leaves | 9.48 / 9.61 |
| LightGBM boosting on ratings residual | **9.43** |

With only final scores there are no interactions to exploit. The interactions in §3–§5
require play-level and weather data.
