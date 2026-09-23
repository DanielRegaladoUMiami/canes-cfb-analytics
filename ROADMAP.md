# Roadmap — Canes CFB Analytics

Pipeline detail: [`docs/pipeline.md`](docs/pipeline.md).

## Current milestone: v0.1 — data foundation
- [x] Repo scaffold: market registry, model notebooks, data and evaluation notebooks
- [x] Ingest games + quarter scores 2015–2026 from ESPN (10,630 games, validated)
- [x] EDA: ~16,600 team-game rows; scoring trend, home field, quarters
- [ ] Get `CFBD_API_KEY` into `~/.zshrc` (#1)
- [ ] Ingest full-game lines from CFBD (#3)
- [ ] Find a source for half, quarter and team-total lines (#4)
- [ ] Feature engineering: opponent-adjusted ratings, recency, priors, context (#5)

## v0.2 — points per team model (full game)
- [ ] Walk-forward CV + Optuna for linear, GLM, RF, XGBoost, LightGBM, CatBoost
- [ ] Ensemble / stacking on validation (2024)
- [ ] Test once on 2025
- [ ] Derive spread, total and winner from team points; compare vs closing lines
- [ ] Weekly predictions for 2026

## v0.3 — dedicated markets
- [ ] Dedicated spread, total and winner models: do they beat the derived version?
- [ ] 1H / 2H spread, total, team total
- [ ] Q1–Q4 spread and total

## v0.4 — weekly operation
- [ ] Weekly card: every model's picks ranked by edge vs current line
- [ ] Track results and CLV through the 2026 season

## Done
- Repo created (2026-09-23)
- ESPN ingestion + EDA (2026-09-23)
