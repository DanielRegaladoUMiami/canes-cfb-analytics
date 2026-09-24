# Pre-registration: key-player availability (round 5)

Written and committed **before** running anything. Results go in
`2026-09-24_availability_results.md`.

## Idea

No free injury data exists. The closest signal: a key player who had stats earlier in the
season but **none in his team's previous game** was probably hurt, suspended or benched,
and may still be out. The market partly knows this (injury news); the question is whether
the model gains from it.

## Data

CFBD `/games/players` (full box score) for 2016–2025, one call per season-week. A player
"played" a game if he recorded any stat in it (passing, rushing, receiving, defense, ...).

## Features (as of each game, previous games of the same season only)

Key players = season-to-date leaders before the game: QB1 by pass attempts, top 2 by
carries, top 3 by receiving yards, top 3 by tackles (one role per player, in that order).

| Feature | Meaning |
|---|---|
| `miss_qb` | QB1 had no stats in the team's previous game |
| `miss_skill` | how many of the top 2 rushers + top 3 receivers had no stats in the previous game |
| `miss_def` | how many of the top 3 tacklers had no stats in the previous game |
| `opp_miss_qb`, `opp_miss_skill`, `opp_miss_def` | the same for the opponent |

A team's first game of the season gets 0 (no previous game this season).

## Candidate and rule

- **A**: current model (with weather).
- **I1**: A + the six features above.

Primary metric: **margin MAE**. Adopt only if I1 beats A in at least 2 of the 3 decision
seasons (2021–2023) and the paired bootstrap 95% interval of the gain excludes zero.
2024–2025 are a check only. Also reported, not used to decide: total MAE, ATS and O/U vs
the opening line.
