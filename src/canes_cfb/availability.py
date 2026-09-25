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


MISS_COLS = ["miss_qb", "miss_skill", "miss_def"]


def _leaders(prior, n: int, used: set[int]) -> list[int]:
    """Indexes of the top ``n`` players by a season-to-date stat, skipping players who
    already have a role; adds them to ``used``."""
    import numpy as np

    top = [int(i) for i in np.argsort(-prior) if prior[i] > 0 and int(i) not in used][:n]
    used.update(top)
    return top


def as_of_features(box: pd.DataFrame, games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team_id): key players (season-to-date leaders before the game) who
    recorded no stats in the team's previous game this season. Round 5 candidate
    (docs/experiments/2026-09-24_availability_preregistration.md)."""
    import numpy as np

    done = games[games.completed]
    b = team_ids(box[box.game_id.isin(done.game_id)], done, teams)
    stats = ["pass_att", "carries", "rec_yds", "tackles"]
    for c in stats:
        b[c] = b[c].fillna(0.0) if c in b else 0.0
    b = b.groupby(["game_id", "team_id", "player_id"], as_index=False)[stats].sum()

    tg = pd.concat([
        games[["game_id", "season", "start_utc", "home_id"]].rename(columns={"home_id": "team_id"}),
        games[["game_id", "season", "start_utc", "away_id"]].rename(columns={"away_id": "team_id"}),
    ]).sort_values("start_utc")  # fmt: skip
    by_team = {k: d for k, d in b.groupby("team_id")}
    rows = []
    for (tid, _season), d in tg.groupby(["team_id", "season"]):
        gids = d.game_id.tolist()
        bt = by_team.get(tid)
        bt = bt[bt.game_id.isin(gids)] if bt is not None else pd.DataFrame(columns=b.columns)
        players = bt.player_id.unique()
        pidx = {p: i for i, p in enumerate(players)}
        gidx = {g: j for j, g in enumerate(gids)}
        mats = {c: np.zeros((len(players), len(gids))) for c in stats}
        played = np.zeros((len(players), len(gids)), dtype=bool)
        for r in bt.itertuples():
            i, j = pidx[r.player_id], gidx[r.game_id]
            played[i, j] = True
            for c in stats:
                mats[c][i, j] = getattr(r, c)
        cum = {c: np.cumsum(m, axis=1) for c, m in mats.items()}
        for j, gid in enumerate(gids):
            feat = {"game_id": gid, "team_id": tid, "miss_qb": 0.0, "miss_skill": 0.0,
                    "miss_def": 0.0}  # fmt: skip
            if j > 0 and len(players) and played[:, j - 1].any():
                used: set[int] = set()
                prior = {c: cum[c][:, j - 1] for c in stats}
                out_last = ~played[:, j - 1]
                qb = _leaders(prior["pass_att"], 1, used)
                skill = _leaders(prior["carries"], 2, used) + _leaders(prior["rec_yds"], 3, used)
                dfn = _leaders(prior["tackles"], 3, used)
                feat["miss_qb"] = float(sum(out_last[i] for i in qb))
                feat["miss_skill"] = float(sum(out_last[i] for i in skill))
                feat["miss_def"] = float(sum(out_last[i] for i in dfn))
            rows.append(feat)
    return pd.DataFrame(rows)
