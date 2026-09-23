# Data

Shared pipeline stages: ingest, EDA, feature engineering. Run in order before any model
notebook. See [`docs/pipeline.md`](../../docs/pipeline.md).

| Notebook | What it does | Status |
|---|---|---|
| [`01_ingest_games.ipynb`](01_ingest_games.ipynb) | ESPN games + quarter scores 2015–2026 → `data/raw/games.parquet` | done |
| [`02_ingest_lines.ipynb`](02_ingest_lines.ipynb) | CFBD lines, advanced stats, talent, returning production | done |
| [`03_eda.ipynb`](03_eda.ipynb) | Enough data? Target, trends, home field, quarters | done |
| [`04_feature_engineering.ipynb`](04_feature_engineering.ipynb) | As-of features v1 (scores) + v2 (CFBD), ablation, interaction tests, market benchmark → `data/processed/team_games.parquet` | v2 done |
