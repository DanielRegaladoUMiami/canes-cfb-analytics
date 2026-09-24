# Data sources

| Need | Source | Status |
|---|---|---|
| Games, venue, neutral site, conference game, final score | ESPN scoreboard (`groups=80`) | **Done.** 2015–2026 in `data/raw/games.parquet` |
| Points by quarter (Q1–Q4, OT) | ESPN scoreboard `linescores` | **Done.** Every completed game since 2015 |
| Full-game spread, total, moneyline (open + close) | CollegeFootballData (CFBD) `/lines` | **Done.** Close: 100% of FBS games; open: ~64% |
| Team efficiency: EPA/PPA, success rate, plays (tempo), pass/rush splits | CFBD `/stats/game/advanced` | **Done.** 99.6% of FBS games |
| Talent composite, returning production | CFBD `/talent`, `/player/returning` | **Done.** 2015–2026 |
| Transfer portal (2021+), recruiting classes, coaches (hire dates), preseason AP poll | CFBD `/player/portal`, `/recruiting/teams`, `/coaches`, `/rankings` (week 1) | **Done.** All known before week 1 |
| Half, quarter and team-total lines | **Open question** (issue #4). CFBD mostly covers full-game lines. | |

## ESPN notes
- Free, no key. Client: `src/canes_cfb/espn.py`. Raw JSON is cached in `data/raw/espn/`.
- FCS opponents appear in the FBS scoreboard. A team is flagged FBS in a season if it has
  6+ games in that season's FBS scoreboard (`home_fbs` / `away_fbs`).
- 3 games were shortened by weather (`shortened = True`); they have no Q4.
- No historical betting lines.

## CFBD notes
- Client: `src/canes_cfb/cfbd.py`. One call per season per endpoint; responses cached in
  `data/raw/cfbd/`. Full rebuild ≈ 60 calls (free tier: 3,000/month).
- Game ids match ESPN's. Name-keyed tables (advanced, talent, returning) map to ids
  through FBS teams only, because some school names repeat across divisions.

## CFBD API key

Put the key in `~/.zshrc`, never in a notebook or in chat:

```bash
export CFBD_API_KEY=...
```

Notebooks read it with `os.environ["CFBD_API_KEY"]`.

## Data quality fixes
- **Swapped moneylines:** about 2.1% of FBS games in 2021–2025 (mostly older providers)
  have home and away moneylines swapped relative to the spread. `cfbd.clean_moneylines`
  blanks any moneyline whose de-vigged probability is more than 15 points from what the
  closing spread implies. Without this filter a fake +12% to +42% moneyline ROI appears.
- **Weather-shortened games** (3) are flagged `shortened` and excluded from period targets.
