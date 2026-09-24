# Canes CFB Analytics

Machine-learning predictions for every FBS college football game: **who wins, the
spread, the total, each team's points, and every half and quarter**. Every prediction
is measured against what the sportsbooks said, on seasons the model never saw.

**Weekly card (live):** https://danielregaladoumiami.github.io/canes-cfb-analytics/

> **Status (2026 season, week 4): practice mode.** The model predicts well, but it
> doesn't beat the sportsbooks yet. Picks are tracked on paper, never with real money,
> until a full season of results says otherwise.

## The numbers at a glance

Out of sample, 2021–2025 (each season predicted by a model trained only on earlier
seasons). Source: [`models/scorecard.json`](models/scorecard.json).

| What we predict | Measure | Model | Sportsbooks | To make money you need |
|---|---|---|---|---|
| **Winner** | Picks the right winner | **70.6%** | 71.9% | a better price than the books, not just the winner |
| | AUC (0.5 = coin flip, 1 = perfect) | **0.772** | 0.793 | |
| | Betting the moneyline where the model disagrees | **−4.5% ROI** | | positive ROI |
| **Spread** | Average miss on the final margin | 12.7 pts | 12.1 pts | |
| | Covers, betting the model's side vs the opening line | **50.1%** | | **52.4%** at −110 |
| **Game total** | Average miss on total points | 12.9 pts | 12.5 pts | |
| | Wins, betting the model's side vs the opening line | **51.0%** | | **52.4%** |
| | …when the model is 4+ points off the opening total | **54.2%** (1,064 bets) | | **52.4%** |
| **Team points** | Average miss on each team's points | 9.1 pts | 8.8 pts | |
| **Halves / quarters** | Improvement over guessing the average | 1H −11%, Q2 −8%, Q4 −1% | Kalshi prices (see below) | |

**Two signals have held up, both on totals:** games that look like shootouts (ratings
project more than 63.5 points) went **under 56.6% of the time**, and when the model is 4+
points off the opening total its side won **54.2%**, with the line moving toward it before
kickoff in all five seasons (closing-line value +0.45 points). The weekly card only
recommends totals with a win chance of 54%+ (Lean) or 55%+ (Best Bet); everything else
is marked *Pass*.

**What's in the model now (round 4, September 2026):** game-time **weather** (wind,
rain, temperature: forecast for upcoming games, station observations for the past)
improved total-points error in all three test seasons and was adopted. **Quarterback**
features (experience, efficiency, QB changes) were tested and added nothing. **Kalshi**
prices for halves, quarters and team totals are shown next to the model's chance, but a
backtest with real Kalshi prices (2026 weeks 1–3, 300 picks) lost 2.3% after fees, so
they're information, not picks. Details in [`docs/experiments/`](docs/experiments/).

## How it works, in plain English

1. **Collect the past.** ~10,600 games since 2015 from ESPN (scores by quarter) and
   CollegeFootballData (betting lines, efficiency stats, talent, transfer portal,
   recruiting, coaching changes, preseason AP poll).
2. **Turn each game into clues** known *before* kickoff (~130 features): each team's
   offense and defense adjusted for the quality of opponents, recent form, efficiency
   (EPA per play, success rate, pace), roster talent and turnover, and context (home
   field, rest, week). A test guarantees no clue ever uses information from the future.
3. **Learn.** One model predicts **each team's points**. Spread, total and winner come
   from the two predictions. Six algorithms were tuned with Optuna; each setting was
   scored by training on earlier seasons and predicting the next one (walk-forward
   cross-validation). A random forest won on 2024 and was tested once on 2025.
4. **Turn points into honest probabilities.** Win chances come from the model's
   out-of-sample error; cover/over chances come from how bets with the same edge
   actually did in 2021–2025.
5. **Pick only with value.** A bet is listed only if its historical win rate beats the
   52.4% break-even at −110. Most games are a *Pass*.

## Every week

A GitHub Action runs every Wednesday (9:00 ET):

1. **Grade** last week's paper picks → [`paper_trading/`](paper_trading/)
2. **Predict** the next slate → `data/predictions/`
3. **Log** the new paper picks and every game's prediction (before kickoff, timestamped in git) →
   [`paper_trading/2026_bets.csv`](paper_trading/2026_bets.csv), [`paper_trading/2026_predictions.csv`](paper_trading/2026_predictions.csv)
4. **Build** the weekly card website → [`site/index.html`](site/index.html), published to
   [GitHub Pages](https://danielregaladoumiami.github.io/canes-cfb-analytics/) by `.github/workflows/pages.yml`

The card shows Best Bets, every game with win chances, odds, the kickoff weather and a
score breakdown (game, halves, quarters: each team's points, total and spread). Each game
has **Team News** (key players and anyone who recorded no stats in his team's last game:
the closest thing to an injury report that free data allows), the full **Roster**, and
**Kalshi** prices vs the model. **Results** grades every finished game (model vs final
score, right/wrong, points off, line movement). Plus a *Your Call* tool, Betting 101, the
report card, and automated model checks. On Fridays the Action refreshes Kalshi prices
(quarter markets open on Thursday).

## Documentation

| Document | What's in it |
|---|---|
| [**`report/Canes-CFB-Analytics.pdf`**](report/Canes-CFB-Analytics.pdf) | The full book: data, features (with the math), nonlinearity insights, models, the one-shot test, the report card, improvement rounds, betting math, decisions and every correction |
| [`docs/pipeline.md`](docs/pipeline.md) | The ML pipeline step by step |
| [`docs/markets.md`](docs/markets.md) | Market definitions and sign conventions |
| [`docs/data_sources.md`](docs/data_sources.md) | Where each piece of data comes from |
| [`docs/experiments/`](docs/experiments/) | One log per experiment, including the ones that failed |
| [`ROADMAP.md`](ROADMAP.md) | What's done and what's next |

## Repository map

```
notebooks/
  00_data/            ingest → EDA → feature engineering → nonlinearity insights
  01_winner/ 02_spread/ 03_total/ 04_team_total/   one notebook per market and period
  99_evaluation/      backtests against the lines
src/canes_cfb/        shared, tested code (data clients, features, models, betting math)
scripts/              predict_week · grade_paper · build_site · calibrate · scorecard ·
                      tune_team_points · round2_experiments · round3_recency
models/               tuned parameters, final recipe, calibration, report card (JSON)
paper_trading/        pre-registered picks and their graded results
site/                 the weekly card website (template + built page)
report/               the Quarto book (PDF + HTML)
docs/                 pipeline, markets, data sources, research, experiment logs
tests/                39 tests, including leakage tests
```

## Run it yourself

```bash
mkdir .venv.nosync && ln -s .venv.nosync .venv   # only if the folder is iCloud-synced
uv sync
uv run pytest                                    # all tests, offline

export CFBD_API_KEY=...                          # put this in ~/.zshrc, never in the repo
uv run python scripts/predict_week.py            # next slate + paper picks
uv run python scripts/grade_paper.py             # grade finished picks
uv run python scripts/build_site.py              # rebuild the weekly card
uv run python scripts/scorecard.py               # rebuild the report card
```

The GitHub Action needs the key as a repository secret named `CFBD_API_KEY`.

## License

Apache 2.0
