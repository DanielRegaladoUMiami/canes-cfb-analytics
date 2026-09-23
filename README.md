# Canes CFB Analytics

Notebook-first college football prediction models. Each market we predict has its own
folder of Jupyter notebooks, one per game period:

| Market | Folder | Full game | 1H | 2H | Q1–Q4 |
|---|---|:-:|:-:|:-:|:-:|
| Winner (moneyline) | [`notebooks/01_winner`](notebooks/01_winner) | ✓ | | | |
| Spread | [`notebooks/02_spread`](notebooks/02_spread) | ✓ | ✓ | ✓ | ✓ |
| Game total (O/U) | [`notebooks/03_total`](notebooks/03_total) | ✓ | ✓ | ✓ | ✓ |
| Team totals | [`notebooks/04_team_total`](notebooks/04_team_total) | ✓ | ✓ | ✓ | |

That makes 18 model notebooks. They are fed by the data notebooks and judged by the
evaluation notebooks.

## Documentation

**[`report/Canes-CFB-Analytics.pdf`](report/Canes-CFB-Analytics.pdf)**: the full write-up
(data, features, nonlinearity insights, models, the one-shot test, betting rules,
decisions and corrections). Built with Quarto + LaTeX from `report/`; every table and
figure is computed from the data. Rebuild with `cd report && quarto render`.

## Weekly card (website)

`site/index.html` is the weekly card: predicted score for every FBS game, win
probability (model vs market), spread and total with **historically calibrated**
probabilities and expected value at −110, every half and quarter, the paper bets, an
explainer of probability/ROI/risk with a risk calculator, and automated verification
checks. Built by `scripts/build_site.py` from the latest predictions and
`models/calibration.json` (`scripts/calibrate.py`, out-of-sample 2021–2025). The GitHub
Action rebuilds it every Wednesday.

## Repo layout

```
canes-cfb-analytics/
├── notebooks/
│   ├── 00_data/            # ingest (ESPN, CFBD) → EDA → feature engineering
│   ├── 01_winner/          # full_game.ipynb
│   ├── 02_spread/          # full_game, 1H, 2H, Q1, Q2, Q3, Q4
│   ├── 03_total/           # full_game, 1H, 2H, Q1, Q2, Q3, Q4
│   ├── 04_team_total/      # full_game, 1H, 2H
│   └── 99_evaluation/      # backtest, calibration, weekly card
├── src/canes_cfb/          # shared code imported by notebooks
│   ├── markets.py          # market × period registry: the single source of truth
│   ├── espn.py             # ESPN games + quarter scores client (cached)
│   ├── cfbd.py             # CFBD lines, advanced stats, talent, returning (cached)
│   ├── features.py         # as-of features: ridge ratings, Elo, matchup, context
│   ├── validation.py       # time split + walk-forward CV
│   └── paths.py            # data/notebook paths
├── scripts/
│   ├── make_notebooks.py   # generates notebook skeletons from the registry
│   ├── tune_team_points.py # Optuna for every model → models/team_points_params.json
│   └── predict_week.py     # next slate's predictions + paper bets
├── models/                 # tuned params + final recipe (JSON, committed)
├── paper_trading/          # pre-registered picks, logged before kickoff
├── data/                   # raw → interim → processed → predictions (git-ignored)
├── report/                 # Quarto book (PDF + HTML) documenting everything
├── docs/
│   ├── pipeline.md         # EDA findings, features, split, models, tuning, ensemble
│   ├── research/           # feature-engineering research with sources
│   ├── markets.md          # target definitions, sign conventions, OT rules
│   ├── data_sources.md     # where each piece of data comes from
│   └── experiments/        # one log entry per model version
└── tests/
```

## Workflow

Full detail in [`docs/pipeline.md`](docs/pipeline.md).

1. **Data**: `00_data/`: ingest → EDA (is there enough data?) → feature engineering.
2. **Model**: each market notebook runs split → walk-forward CV + Optuna for each model
   (linear, GLM, RF, XGBoost, LightGBM, CatBoost) → compare → ensemble/stacking →
   test once → predict this week. The first model is points per team (`04_team_total/full_game`).
3. **Evaluate**: `99_evaluation/` compares every market and builds the weekly card.

Rules that apply to every notebook:
- **As-of features only.** Nothing that wasn't known before kickoff.
- **Split by time**, never random: train 2015–2023, validation 2024, test 2025 (once).
- **The closing line is the benchmark.** A model that can't beat it has no edge.

## Adding a market or period

Add it to `src/canes_cfb/markets.py`, then run:

```bash
uv run python scripts/make_notebooks.py   # creates only what's missing
```

## How to run

```bash
mkdir .venv.nosync && ln -s .venv.nosync .venv   # macOS + iCloud Desktop only
uv sync
uv run pre-commit install
uv run jupyter lab
uv run pytest
uv run python scripts/predict_week.py   # this week's predictions and card
```

Data access needs a free CollegeFootballData API key in your shell profile
(`export CFBD_API_KEY=...` in `~/.zshrc`). Never commit it. See
[`docs/data_sources.md`](docs/data_sources.md).

## License

Apache 2.0
