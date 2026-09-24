# Round 4 results: weather and quarterbacks

Pre-registered in `2026-09-24_weather_qb_preregistration.md` (committed before running).
`scripts/round4_weather_qb.py` → `models/round4_weather_qb.json`.

Deviations, both forced by the data and decided before looking at results: gust speed was
dropped (station observations lack gusts in 95% of games) and `indoor` wasn't added (the
model already has it). Weather source changed from Open-Meteo reanalysis to Meteostat
station observations (Open-Meteo's rate limits made 10 seasons of history impractical);
97% of played FBS games have weather.

| Candidate | Total MAE 21–23 | Margin MAE 21–23 | O/U open 21–23 | Total MAE 24–25 | O/U open 24–25 |
|---|---:|---:|---:|---:|---:|
| A current | 13.03 | 12.82 | 50.9% | 12.82 | 50.3% |
| W1 weather (raw) | 12.99 | 12.83 | 50.1% | 12.80 | 52.3% |
| W2 weather (thresholds) | 13.00 | 12.83 | 50.8% | 12.84 | 51.6% |
| Q1 quarterbacks | 13.00 | 12.84 | 50.5% | 12.81 | 51.2% |

| Candidate | Metric | Seasons better (of 3) | Gain, 95% CI | Adopt |
|---|---|---:|---|---|
| W1 weather (raw) | total MAE | 3 | [+0.008, +0.064] | **yes** |
| W2 weather (thresholds) | total MAE | 3 | [+0.000, +0.054] | **yes** |
| Q1 quarterbacks | margin MAE | 0 | [-0.052, +0.009] | no |

## Decision

- **W1 adopted** (wind speed, precipitation, temperature at kickoff). Total error improves
  in all three decision seasons; the gain is small (about 0.04 points per game) but its
  interval excludes zero, and the 2024–25 check agrees (12.82 → 12.80; O/U vs the opener
  50.3% → 52.3%). W2 passed too, with a weaker interval touching zero; W1 was the stronger
  of two versions of the same idea, so only W1 goes in.
- **Q1 not adopted.** QB experience, efficiency and changes add nothing the team ratings
  don't already carry. What would matter (the starter is out *this* week) isn't in any
  free data source.
- Honest read on betting: W1 lowers the *error*, but the O/U record vs the opener moves
  both ways (2021–23 50.9% → 50.1%, 2024–25 50.3% → 52.3%). It's a slightly better
  estimate, not yet a proven edge against the line.
- Caveat: history uses observed weather, live weeks use a 3–4 day forecast, so the live
  gain will be somewhat smaller than the backtest.
- Period (half/quarter) models keep the pre-weather features: weather was only tested on
  full-game team points.
