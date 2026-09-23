# Roadmap — Canes CFB Analytics

## Current milestone: v0.1 — data foundation
- [x] Repo scaffold: market registry, 18 model notebooks, data and evaluation notebooks
- [ ] Get `CFBD_API_KEY` (free) into `~/.zshrc`
- [ ] Ingest games + quarter line scores (2015–2026), assert 1H + 2H = full game
- [ ] Ingest full-game lines (spread, total, moneyline; open + close)
- [ ] Find a source for half, quarter, and team-total lines
- [ ] Shared feature table with all period targets

## v0.2 — full-game models
- [ ] Winner / full_game
- [ ] Spread / full_game
- [ ] Total / full_game
- [ ] Team total / full_game
- [ ] Backtest + calibration notebooks working for full game

## v0.3 — halves and quarters
- [ ] 1H / 2H spread, total, team total
- [ ] Q1–Q4 spread and total (try deriving from full-game models first; see if
      dedicated models beat that)

## v0.4 — weekly operation
- [ ] Weekly card: every model's picks ranked by edge vs current line
- [ ] Track CLV on picks through the 2026 season

## Done
- Repo created (2026-09-23)
