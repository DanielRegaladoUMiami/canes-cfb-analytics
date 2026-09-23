# Roadmap — Canes CFB Analytics

Pipeline detail: [`docs/pipeline.md`](docs/pipeline.md).

## Current milestone: v0.1 — data foundation
- [x] Repo scaffold: market registry, model notebooks, data and evaluation notebooks
- [x] Ingest games + quarter scores 2015–2026 from ESPN (10,630 games, validated)
- [x] EDA: ~16,600 team-game rows; scoring trend, home field, quarters
- [x] Get `CFBD_API_KEY` into `~/.zshrc` (#1)
- [x] Ingest CFBD lines, advanced stats, talent, returning production (#3)
- [ ] Find a source for half, quarter and team-total lines (#4)
- [x] Feature engineering v1 (score-only): ridge ratings, Elo, matchup, context + ablation (#5)
- [x] Feature engineering v2 (CFBD): efficiency, pace × efficiency, unit matchups, priors (#6), 9.30 vs market 8.91
- [x] First backtest vs lines: not profitable yet; positive CLV vs opening lines
- [x] Preseason features (portal, recruiting, new coach, preseason AP): early-season gap 1.96 → 1.84 (2021–23), 1.32 → 1.11 (2024–25)
- [ ] Bowls and blowouts gaps (#11)
- [ ] Weather features (needs a source)

## v0.2 — points per team model (full game)
- [x] Market layer vs opening line (#7): tested; dropped (spreads noise, totals = base-rate artifact)
- [x] Walk-forward CV + Optuna for linear, GLM, RF, XGBoost, LightGBM, CatBoost
- [x] Ensemble / stacking; final picked on 2024 (random forest)
- [x] Test once on 2025: MAE 8.94 vs market 8.70 open / 8.59 close
- [x] Derive spread, total from team points; graded vs open and close
- [x] Weekly predictions for 2026 (`scripts/predict_week.py`) + paper-trading log
- [x] Weekly grading script + GitHub Action (Wednesdays)
- [x] Nonlinearity insights (SHAP + market lens): shootout-under effect; paper rule B
- [x] Half/quarter team-points models (accuracy; no period lines yet)
- [x] Totals model on the nonlinear insights (#10): not adopted (overfit); rule B kept; A-vs-B conflicts skipped
- [ ] Winner (moneyline) probabilities from predicted margin

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
- Feature engineering v1: ratings 11.1 → 9.44 CV MAE (2026-09-23)
- CFBD ingestion + features v2: 9.30 CV MAE; first backtest vs lines (2026-09-23)
- Tuned zoo + ensemble + 2025 test; paper trading starts 2026 week 4 (2026-09-23)
