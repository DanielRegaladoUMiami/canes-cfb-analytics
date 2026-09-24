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
| **Winner** | Picks the right winner | **70.8%** | 71.9% | a better price than the books, not just the winner |
| | AUC (0.5 = coin flip, 1 = perfect) | **0.773** | 0.793 | |
| | Betting the moneyline where the model disagrees | **−4.5% ROI** | | positive ROI |
| **Spread** | Average miss on the final margin | 12.7 pts | 12.1 pts | |
| | Covers, betting the model's side vs the opening line | **50.6%** | | **52.4%** at −110 |
| **Game total** | Average miss on total points | 12.9 pts | 12.5 pts | |
| | Wins, betting the model's side vs the opening line | **50.7%** | | **52.4%** |
| **Team points** | Average miss on each team's points | 9.1 pts | 8.8 pts | |
| **Halves / quarters** | Improvement over guessing the average | 1H −11%, Q2 −8%, Q4 ≈ 0% | (no lines yet) | |

**One signal has held up:** games that look like shootouts (ratings project more than
63.5 points) went **under about 56% of the time in every season 2021–2025**. That's
what the weekly card recommends; everything else is marked *Pass*.

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
3. **Log** the new paper picks (before kickoff, timestamped in git) → [`paper_trading/2026_bets.csv`](paper_trading/2026_bets.csv)
4. **Build** the weekly card website → [`site/index.html`](site/index.html), published to
   [GitHub Pages](https://danielregaladoumiami.github.io/canes-cfb-analytics/) by `.github/workflows/pages.yml`

The card shows Best Bets, every game with win chances and odds, a *Your Call* tool to
test your own hunch against the price, Betting 101, the report card, and automated
model checks.

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
