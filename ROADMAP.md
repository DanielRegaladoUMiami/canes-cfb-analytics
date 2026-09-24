# Roadmap: Canes CFB Analytics

Last updated: 2026-09-23 (2026 season, week 4). Details for every finished item are in
[`docs/experiments/`](docs/experiments/) and the book ([`report/`](report/)).

## Needs the owner (Daniel)
- [ ] Add the `CFBD_API_KEY` repository secret so the Wednesday GitHub Action can run
- [ ] Rotate the CFBD key (it was pasted in a chat) and update `~/.zshrc` and the secret
- [ ] Decide whether to publish the weekly card publicly on GitHub Pages (auto-updates)

## Next
- [ ] **Starting-QB changes and injuries**: the biggest information gap vs the market (#11)
- [ ] **Weather** (wind) for totals, via Open-Meteo (#8)
- [ ] **Bowls**: opt-outs, rest, bowl tier (#11)
- [ ] **Blowouts**: game-script features (#11)
- [ ] Grade week 4 and every week after; track results and CLV through 2026
- [ ] End of 2026: compare the shadow model (average of 6) with the current model;
      decide real money only if a full season beats 52.4% with positive CLV
- [ ] Sportsbook lines for halves, quarters and team totals (#4): The Odds API
      ($119 once for 2023–2025 history, or $30/month going forward)

## Done

### Data
- [x] 10,630 games 2015–2026 with scores by quarter (ESPN), validated
- [x] Lines, efficiency, talent, returning production (CFBD); closing lines for 100% of
      FBS games, opening lines from 2021
- [x] Preseason data: transfer portal, recruiting, coaching changes, preseason AP poll
- [x] Swapped moneylines (2.1% of games) detected and filtered

### Features
- [x] Opponent-adjusted ridge ratings refit every week (the biggest single gain: 11.1 → 9.44)
- [x] Elo, matchup terms, context, CFBD efficiency, talent and returning production
- [x] Preseason family, adopted: early-season gap vs market 1.96 → 1.84 (2021–23),
      1.32 → 1.11 (2024–25)
- [x] Nonlinearity search (SHAP + market lens): curves and thresholds matter, crosses don't

### Models and evaluation
- [x] Six models tuned with Optuna inside walk-forward CV; random forest chosen on 2024;
      tested once on 2025
- [x] Halves and quarters models
- [x] Report card for every target (accuracy, AUC, MAE vs sportsbooks, by segment)
- [x] Improvement rounds 2–3 under a pre-registered anti-overfitting rule: nothing
      adopted; average-of-6 kept as a shadow model

### Betting and operation
- [x] Calibrated probabilities and expected value at −110
- [x] Two pre-registered paper rules (A: model edge ≥ 4 vs opening total; B: shootout
      under); conflicts are skipped
- [x] Weekly GitHub Action: grade, predict, log picks, build the site
- [x] Weekly card website in a sports-app layout (Best Bets, game cards, Your Call,
      Betting 101, Report Card, Model Check)
- [x] Quarto + LaTeX book documenting everything
