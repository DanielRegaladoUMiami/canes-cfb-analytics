# Data sources

| Need | Source | Status |
|---|---|---|
| Games, venue, neutral site, conference game, final score | ESPN scoreboard (`groups=80`) | **Done.** 2015–2026 in `data/raw/games.parquet` |
| Points by quarter (Q1–Q4, OT) | ESPN scoreboard `linescores` | **Done.** Every completed game since 2015 |
| Full-game spread, total, moneyline (open + close) | CollegeFootballData (CFBD) `/lines` | Needs `CFBD_API_KEY` |
| Team efficiency: EPA/PPA, success rate, plays (tempo) | CFBD `/ppa/*`, `/stats/*` | Needs `CFBD_API_KEY` |
| Half, quarter and team-total lines | **Open question** (issue #4). CFBD mostly covers full-game lines. | |

## ESPN notes
- Free, no key. Client: `src/canes_cfb/espn.py`. Raw JSON is cached in `data/raw/espn/`.
- FCS opponents appear in the FBS scoreboard. A team is flagged FBS in a season if it has
  6+ games in that season's FBS scoreboard (`home_fbs` / `away_fbs`).
- 3 games were shortened by weather (`shortened = True`); they have no Q4.
- No historical betting lines.

## CFBD API key

Put the key in `~/.zshrc`, never in a notebook or in chat:

```bash
export CFBD_API_KEY=...
```

Notebooks read it with `os.environ["CFBD_API_KEY"]`.
