# Round 5 results: key-player availability (injury proxy)

Pre-registered in `2026-09-24_availability_preregistration.md` (committed before running).
`scripts/round5_availability.py` → `models/round5_availability.json`. Box scores: CFBD
`/games/players` 2016–2026 (697,143 player-game rows). The season's QB1 had no stats in
his team's previous game in 7.8% of team-games; at least one key skill player in 29%.

| Candidate | Margin MAE 21–23 | Total MAE 21–23 | O/U open 21–23 | Margin MAE 24–25 | O/U open 24–25 |
|---|---:|---:|---:|---:|---:|
| A current (with weather) | 12.83 | 12.99 | 50.1% | 12.53 | 52.3% |
| I1 availability | 12.83 | 12.98 | 51.6% | 12.57 | 51.7% |

| Metric | Seasons better (of 3) | Gain, 95% CI | Adopt |
|---|---:|---|---|
| margin MAE (primary) | 1 | [−0.021, +0.031] | **no** |
| total MAE (info only) | 2 | [−0.020, +0.030] | no |

## Decision

**Not adopted.** Who missed last week doesn't predict this week well enough: many of
those players are back, and when they aren't, the injury news reaches the market (and the
line) before kickoff. The model can't see that news, so the proxy adds noise, not signal.

The flag stays on the site (Team News) as information for the reader, who can check the
official availability report. A real edge here would need the report itself (game-week
availability), which no free source provides as data.
