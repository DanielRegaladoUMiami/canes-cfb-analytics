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
v0.1: data foundation (games, quarter scores, lines → shared feature table)

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
- As-of features only; walk-forward splits by season/week; closing line is the baseline.
- Logic reused by 2+ notebooks moves into `src/canes_cfb/` with a test.
- `CFBD_API_KEY` lives in `~/.zshrc`. Never in notebooks, never pasted in chat.

## How to run
```bash
uv sync
uv run jupyter lab
uv run pytest
uv run python scripts/make_notebooks.py   # after editing markets.py
```

## Where things live
- Notebooks: `notebooks/{00_data,01_winner,02_spread,03_total,04_team_total,99_evaluation}/`
- Shared code: `src/canes_cfb/`
- Data (git-ignored): `data/{raw,interim,processed,predictions}/`
- Market rules: `docs/markets.md`; data sources: `docs/data_sources.md`
- Experiments: `docs/experiments/`
- Roadmap: `ROADMAP.md`

## iCloud gotcha
The repo sits under iCloud-synced `~/Desktop`, and iCloud hides the venv's `.pth` files,
which breaks `import canes_cfb`. So `.venv` is a symlink to `.venv.nosync/`, which iCloud
never syncs. On a fresh clone:
`mkdir .venv.nosync && ln -s .venv.nosync .venv && uv sync`.
