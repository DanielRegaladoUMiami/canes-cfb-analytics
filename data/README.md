# Data

Nothing in here is committed except this file and the `.gitkeep` placeholders.
Everything is rebuilt by the notebooks in `notebooks/00_data/`.

| Folder | Contents | Written by |
|---|---|---|
| `raw/` | API pulls as received (games, line scores, betting lines) | `00_data/01`–`03` |
| `interim/` | Cleaned and joined, not yet model-ready | `00_data/*` |
| `processed/` | The shared feature table (one row per game, all period targets) | `00_data/04_build_feature_table` |
| `predictions/` | One parquet per market and period (`<market>_<period>.parquet`) | each model notebook |
