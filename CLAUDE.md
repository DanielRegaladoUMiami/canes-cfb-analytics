# Canes CFB Analytics

## Goal
Notebook-first college football prediction models: winner, spread, game total and team
totals for full game, halves and quarters. One folder per market type, one notebook per
period, all judged against the closing line.

## Stack
- Python 3.11+ (pinned 3.12 in `.python-version`)
- uv (build/dep management)
- ruff (lint + format, pre-commit), nbstripout (notebook outputs never committed)
- Jupyter notebooks for models; shared code in `src/canes_cfb/`

## Current milestone
2026 season in practice mode: weekly predictions, paper picks and the weekly card run from
`scripts/predict_week.py` → `grade_paper.py` → `kalshi_snapshot.py` → `build_site.py`
(GitHub Action, Wednesdays; Fridays re-snapshot Kalshi); `pages.yml` then deploys `site/`
to GitHub Pages. Weather is in the model (round 4); QB features and Kalshi period picks
were tested and not adopted (see `docs/experiments/2026-09-24_*`).
Next model work: an injury proxy from box scores (key player missed last game), bowls.

## Local rules
- Conventional Commits (feat:, fix:, docs:, refactor:, chore:, test:)
- No `Co-Authored-By` in commits. Sole author is Daniel.
- Use `uv` not pip; `uv add <pkg>` to add deps
- Pre-commit hooks (ruff, nbstripout) run on every commit
- README/docs in English; conversation can be Spanish
- `src/canes_cfb/markets.py` is the source of truth for markets and periods. Add there
  first, then run `scripts/make_notebooks.py`. Never hand-create model notebooks.
- Home-perspective conventions everywhere (see `docs/markets.md`). 2H includes OT,
  quarters don't.
- As-of features only. Split by time: train 2015–2023, validation 2024, test 2025
  (touched once). CV is walk-forward by season. Never random splits. See `docs/pipeline.md`.
- First model: points per team (full game); spread/total/winner derive from it.
- Logic reused by 2+ notebooks moves into `src/canes_cfb/` with a test.
- `CFBD_API_KEY` lives in `~/.zshrc`. Never in notebooks, never pasted in chat.

## How to run
```bash
uv sync
uv run jupyter lab
uv run pytest
uv run python scripts/make_notebooks.py   # after editing markets.py
uv run python scripts/tune_team_points.py  # Optuna, ~30-60 min
uv run python scripts/predict_week.py      # next slate + paper picks (needs CFBD_API_KEY)
uv run python scripts/grade_paper.py       # grade finished paper picks
uv run python scripts/calibrate.py         # probabilities from out-of-sample 2021-2025
uv run python scripts/scorecard.py         # report card for every target
uv run python scripts/build_site.py        # weekly card website (site/index.html)
uv run python scripts/backfill_season.py   # re-predict finished weeks with no live prediction
uv run python scripts/kalshi_snapshot.py   # Kalshi prices vs the model (information only)
uv run python scripts/kalshi_backtest.py   # Kalshi period picks on past weeks (real prices)
uv run python scripts/period_calibration.py  # probabilities for any half/quarter line
uv run python scripts/clv_history.py       # closing-line value of the practice rules
```

- The 2025 test in `04_team_total/full_game` is run once. Don't re-tune or re-select
  models after looking at it; any change must be justified on CV/2024 and logged.

## Where things live
- Notebooks: `notebooks/{00_data,01_winner,02_spread,03_total,04_team_total,99_evaluation}/`
- Shared code: `src/canes_cfb/`
- Data (git-ignored): `data/{raw,interim,processed,predictions}/`
- Pipeline plan (features, split, models): `docs/pipeline.md`
- Market rules: `docs/markets.md`; data sources: `docs/data_sources.md`
- Experiments: `docs/experiments/`
- Roadmap: `ROADMAP.md`

## iCloud gotcha
The repo sits under iCloud-synced `~/Desktop`, and iCloud hides the venv's `.pth` files,
which breaks `import canes_cfb`. So `.venv` is a symlink to `.venv.nosync/`, which iCloud
never syncs. On a fresh clone:
`mkdir .venv.nosync && ln -s .venv.nosync .venv && uv sync`.

## Anti-overfitting rule (use for every model change)
Pre-register the candidates. Adopt only if margin MAE beats the current model in at least
2 of the 3 decision seasons (2021–2023) and the paired bootstrap 95% interval of the gain
excludes zero. 2024–2025 are a check only. Log every attempt in `docs/experiments/`,
including failures. The website, README and book are in English.
