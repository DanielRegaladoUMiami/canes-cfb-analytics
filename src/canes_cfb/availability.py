"""Key players and who missed their team's last game: an injury proxy from box scores.

No free source publishes college football injury reports as data. What we can see is who
recorded a stat in each game. For each team's key players this season (QB by pass
attempts, top 2 rushers by carries, top 3 receivers by receiving yards, top 3 tacklers),
a player who had stats earlier but **none in the team's most recent game** is flagged:
likely hurt, suspended or benched. It's a hint to check the official availability report,
not a confirmed injury.
"""

from __future__ import annotations

import pandas as pd

from canes_cfb.quarterbacks import team_ids

ROLES = [  # (role, stat to rank by, how many)
    ("QB", "pass_att", 1),
    ("RB", "carries", 2),
    ("WR/TE", "rec_yds", 3),
    ("Defense", "tackles", 3),
]
LINE = {"QB": ("pass_yds", "pass yds"), "RB": ("rush_yds", "rush yds"),
        "WR/TE": ("rec_yds", "rec yds"), "Defense": ("tackles", "tackles")}  # fmt: skip


def key_players(box: pd.DataFrame, games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per (team_id, key player): role, games with stats, per-game production,
    and whether he recorded a stat in the team's most recent game."""
    done = games[games.completed]
    b = team_ids(box[box.game_id.isin(done.game_id)], done, teams)
    kick = done.set_index("game_id").start_utc
    b["kickoff"] = b.game_id.map(kick)
    last_game = b.groupby("team_id").kickoff.transform("max")
    team_games = b.groupby("team_id").game_id.transform("nunique")
    b["in_last"] = b.kickoff == last_game
    rows = []
    for (tid, pid), d in b.groupby(["team_id", "player_id"]):
        s = d.sum(numeric_only=True)
        rows.append({"team_id": tid, "player_id": pid, "player": d.player.iloc[0],
                     "games": d.game_id.nunique(), "team_games": int(team_games.loc[d.index[0]]),
                     "played_last": bool(d.in_last.any()),
                     **{c: s.get(c, 0.0) for c in ("pass_att", "pass_yds", "carries",
                                                    "rush_yds", "rec_yds",
                                                    "tackles")}})  # fmt: skip
    p = pd.DataFrame(rows)
    out = []
    for role, stat, n in ROLES:
        top = p[p[stat] > 0].sort_values(stat, ascending=False).groupby("team_id").head(n)
        yard, label = LINE[role]
        out.append(top.assign(role=role, per_game=(top[yard] / top.games).round(1), unit=label))
    k = pd.concat(out, ignore_index=True).drop_duplicates(["team_id", "player"])  # keep top role
    k["missed_last"] = ~k.played_last & (k.team_games > 1)
    return k[["team_id", "role", "player", "games", "team_games", "per_game", "unit",
              "played_last", "missed_last"]]  # fmt: skip
