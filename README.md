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

## Repo layout

```
canes-cfb-analytics/
├── notebooks/
│   ├── 00_data/            # ingest games, quarter scores, lines → shared feature table
│   ├── 01_winner/          # full_game.ipynb
│   ├── 02_spread/          # full_game, 1H, 2H, Q1, Q2, Q3, Q4
│   ├── 03_total/           # full_game, 1H, 2H, Q1, Q2, Q3, Q4
│   ├── 04_team_total/      # full_game, 1H, 2H
│   └── 99_evaluation/      # backtest, calibration, weekly card
├── src/canes_cfb/          # shared code imported by notebooks
│   ├── markets.py          # market × period registry: the single source of truth
│   └── paths.py            # data/notebook paths
├── scripts/
│   └── make_notebooks.py   # generates notebook skeletons from the registry
├── data/                   # raw → interim → processed → predictions (git-ignored)
├── docs/
│   ├── markets.md          # target definitions, sign conventions, OT rules
│   ├── data_sources.md     # where each piece of data comes from
│   └── experiments/        # one log entry per model version
└── tests/
```

## Workflow

1. **Data**: run `notebooks/00_data/` in order. The last notebook writes one feature table
   with every period target, so every model starts from the same rows.
2. **Model**: each notebook follows the same sections: load, target and features,
   baseline (the line), model, evaluation against the line, and this week's predictions.
3. **Evaluate**: `99_evaluation/` compares every market in one table and builds the
   weekly card.

Rules that apply to every notebook:
- **As-of features only.** Nothing that wasn't known before kickoff.
- **Walk-forward splits** by season and week. No random train/test splits.
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
```

Data access needs a free CollegeFootballData API key in your shell profile
(`export CFBD_API_KEY=...` in `~/.zshrc`). Never commit it. See
[`docs/data_sources.md`](docs/data_sources.md).

## License

Apache 2.0
