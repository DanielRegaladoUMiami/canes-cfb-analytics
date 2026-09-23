# Nonlinearity and interaction insights (model lens + market lens)

- **Date:** 2026-09-23
- **Notebook:** notebooks/00_data/05_nonlinearity_insights.ipynb

## Model lens (SHAP, residual points model, 2019–2023 sample)
| Feature | Shape | Feature idea |
|---|---|---|
| exp_total | +2.2 → −2.2 pts, steepest at the ends | shrink extremes; shootout flag |
| talent_diff | −2.0 / flat middle / +2.5 | tail/threshold on big talent gaps |
| ret_ppa | −1.1 lowest decile, flat above median | low returning production threshold |
| exp_sr | convex, +1.3 top decile | elite-efficiency tail |

Strongest pairwise interaction (talent gap × scoring environment): ~0.14 pts vs ~10 pts
of main effects. Curves and thresholds matter; crosses don't.

## Market lens (closing line, discovery 2016–2023, holdout 2024–2025)
59 segments tested (~3 expected at |z| ≥ 2 by chance); 8 found. Every totals segment
leans under in both periods (high exp_total, high closing total, very high pace, weeks 1–3).
Every spread segment flipped in the holdout, so those are noise.

**Shootout under** (exp_total > 63.5, the top 20% of 2016–2023): under 55–60% in every
season 2021–2025 (2019 59.9%, 2020 51.7%, 2015–2018 ~50%). Vs the opening total:
56.4% over 2021–2025 (n = 509), above 52.4% in each season.

## Actions
- Paper rule B `shootout_under` pre-registered (predict_week.py, paper_trading/).
- Correction: the 2025 market-layer totals result isn't a pure base-rate artifact; part
  of it is this stable effect.
- Conflict on day 1 (Ole Miss @ Florida: rule A over, rule B under): the points model
  under-shrinks shootouts → build a dedicated totals model on these nonlinear features.
