# Pre-registration: weather and quarterback features (round 4)

Written and committed **before** running anything. Results go in
`2026-09-24_weather_qb_results.md`.

## Data

- **Weather** (`src/canes_cfb/weather.py`): Open-Meteo at each venue (CFBD venue
  coordinates, else the city), averaged over the 3.5 hours from kickoff: wind speed (mph),
  highest gust (mph), precipitation (inches), temperature (°F); indoor games = calm, dry,
  70°F. Past seasons use ERA5 reanalysis (what actually happened); live weeks use the
  forecast. That makes the backtest slightly optimistic for weather, noted in the results.
- **Quarterbacks** (`cfbd.load_passing`): passing box score per player-game (CFBD
  `/games/players`). Starter = most pass attempts. All QB features use **previous games
  only** (as of the slate): the team's most recent starter, his experience and efficiency.
  No injury data exists in a free source; the QB change is the proxy.

## Candidates (fixed)

All on the current random-forest recipe (residual of exp_points), walk-forward 2021–2025.

| Id | Adds | Primary metric |
|---|---|---|
| A | nothing (current model) | — |
| W1 | wind_mph, gust_mph, precip_in, temp_f, indoor | total MAE |
| W2 | thresholds: wind ≥ 15 mph, wind above 10 mph (mph over), precip ≥ 0.1 in, temp < 40°F, indoor | total MAE |
| Q1 | qb_starts (log of prior starts in the data, from 2016), qb_new (last game's starter wasn't the starter the game before), qb_ypa (prior yards per attempt, shrunk to 6.8 with 150 attempts), qb_vs_team (qb_ypa minus the team's prior yards per attempt), and the same four for the opponent's QB | margin MAE |

## Rule (unchanged)

Adopt a candidate only if its primary metric beats A in **at least 2 of the 3 decision
seasons (2021–2023)** and the paired bootstrap 95% interval of the gain **excludes zero**.
2024–2025 are a check only. If both a W and Q1 pass, the combination is run once as a
confirmation with the same rule. Also reported, not used to decide: O/U and ATS win % vs
the opening line.
