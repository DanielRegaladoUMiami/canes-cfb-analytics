# Data sources

| Need | Source | Notes |
|---|---|---|
| Games, venue, neutral site, final score | CollegeFootballData (CFBD) `/games` | Needs `CFBD_API_KEY` (free) |
| Quarter-by-quarter scores (Q1–Q4, OT) | CFBD `/games` (`homeLineScores` / `awayLineScores`) | Required for every half and quarter target |
| Full-game spread, total, moneyline | CFBD `/lines` | Several books per game; open and close |
| Team ratings, EPA/PPA, tempo | CFBD `/ratings/*`, `/ppa/*`, `/stats/*` | Use as-of values only |
| Schedule and live scores (fallback) | ESPN scoreboard (`groups=80`) | Free, no key |
| Half, quarter, and team-total lines | **Open question.** CFBD mostly covers full-game lines. | See issue on period lines |

## API key

Put the key in `~/.zshrc`, never in a notebook or in chat:

```bash
export CFBD_API_KEY=...
```

Notebooks read it with `os.environ["CFBD_API_KEY"]`.

## Related repo

[`cfb-canes-analytics`](https://github.com/DanielRegaladoUMiami/cfb-canes-analytics)
already has working Kalshi, Polymarket, and ESPN clients for full-game totals. Reuse its
code instead of rewriting it if exchange prices are needed here.
