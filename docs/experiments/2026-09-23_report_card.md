# Report card: every target, out of sample 2021–2025

- **Date:** 2026-09-23 · **Script:** scripts/scorecard.py → models/scorecard.json
- Walk-forward: each season predicted by models trained only on earlier seasons.
  Market = closing lines (moneyline de-vigged for win probability).

| Target | Metric | Model | Sportsbooks |
|---|---|---|---|
| Winner (3,770 games with a moneyline) | Accuracy | 70.1% | 71.1% |
| | AUC | 0.769 | 0.783 |
| | Brier / log loss | 0.192 / 0.564 | 0.188 / 0.555 |
| Margin (spread) | MAE | 12.74 | 12.12 |
| | ATS vs close / open | 49.0% / 49.8% | — |
| Game total | MAE | 12.94 | 12.54 |
| | O/U vs close / open | 50.2% / 50.5% | — |
| Team points | MAE | 9.15 | 8.77 |
| 1H / 2H / Q1 / Q2 / Q3 / Q4 | MAE (vs constant) | 6.40 / 6.22 / 4.18 / 4.75 / 4.23 / 4.39 | 7.23 / 6.70 / 4.47 / 5.15 / 4.49 / 4.41 |

## Where the model trails the market most (margin MAE gap)
- Weeks 0–2: +1.7 pts (accuracy 72.6% vs 76.0%); weeks 3–4: +1.0; weeks 9+: +0.2
- Bowls/CFP: +1.2 (accuracy 60.8% vs 65.4%)
- Spreads of 21.5+: +1.6 (blowouts); other buckets +0.3 to +0.6

## What to improve next
1. Early-season priors: transfer portal, recruiting classes, coaching changes, preseason
   ratings (CFBD endpoints).
2. Bowls: opt-outs, transfers, days of rest, bowl tier/motivation.
3. Blowouts: game-script features (starters pulled, garbage time) and a heavier-tailed
   margin model.

## Update (same day): swapped moneylines and moneyline betting
- 81 of 3,770 FBS games (2.1%, mostly 2021–2023, older providers) had home/away
  moneylines swapped relative to the spread. They faked a +12% to +42% moneyline ROI.
  `cfbd.clean_moneylines` now blanks any moneyline whose de-vigged probability is more
  than 15 points from what the closing spread implies.
- Clean numbers, 2021–2025: winner accuracy model 70.8% vs books 71.9%; AUC 0.773 vs
  0.793. Betting the model's side on the moneyline whenever it's more confident than
  the books: 45.0% wins, **−4.5% ROI** (−5% to −10% at larger disagreements). No edge.
