# Closing-line value of the practice rules, 2021–2025

**Question.** Does the market move toward our totals picks between the open (where we'd
bet) and the close? Positive CLV is the fastest honest sign of a real edge: it doesn't
need game results.

**Setup.** Out-of-sample random-forest predictions 2021–2025 (walk-forward, same as the
report card). CLV = side × (closing total − opening total), in points.
`scripts/clv_history.py` → `models/clv_history.json`.

| Rule | Bets | Avg CLV | CLV > 0 | Won vs open | Seasons with CLV > 0 |
|---|---:|---:|---:|---:|---:|
| edge4: model side when \|model − open\| ≥ 4 | 1,053 | **+0.46** | 53.4% | 53.7% | 5 of 5 |
| shootout_under | 514 | −0.08 | 46.5% | **56.6%** | 3 of 5 |
| every game, model side | 3,941 | +0.16 | 49.1% | 50.7% | 3 of 5 |

By season, edge4 CLV: 2021 +0.44, 2022 +0.10, 2023 +0.30, 2024 +0.54, 2025 +0.96.

**Reading.**

- **edge4 has real information.** The market moves toward it every season, by about half
  a point. Its win rate (53.7%) is modest because betting at the open only captures part
  of that; it points to betting **early** (at the open) as the way to use it.
- **shootout_under is a different kind of edge.** It wins (56.6%) but the market does
  *not* move toward it: a bias the market doesn't correct during the week. That also
  means there's no rush to bet it early.
- Both are now shown on the site: per pick (line moved our way / against us) in Results,
  and this table in the Report Card.

No model change; nothing adopted or rejected. Pure measurement.
