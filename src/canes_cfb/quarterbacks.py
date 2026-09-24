"""As-of quarterback features from the passing box score (CFBD /games/players).

Starter = the passer with the most attempts. For a game, everything describes the
team's **most recent starter before that game** (the likely starter; no free injury data
exists, so a QB change is the proxy) using only earlier games:

- qb_starts: log(1 + his earlier starts, any team, since 2016)
- qb_new: 1 if the most recent starter wasn't the starter the game before
- qb_ypa: his earlier yards per attempt, shrunk to 6.8 with 150 attempts
- qb_vs_team: qb_ypa minus the team's own yards per attempt over its previous 12 games
  (shrunk the same way): how much better or worse he is than the team's usual passing
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR_YPA, PRIOR_ATT = 6.8, 150.0
QB_COLS = ["qb_starts", "qb_new", "qb_ypa", "qb_vs_team"]


def team_ids(passing: pd.DataFrame, games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Attach team_id to passing rows: CFBD name → id among the game's two teams."""
    ids = teams.groupby("team").team_id.apply(set).to_dict()
    g = games.set_index("game_id")[["home_id", "away_id"]]
    p = passing[passing.game_id.isin(g.index)].copy()
    out = []
    for gid, d in p.groupby("game_id"):
        pair = {g.at[gid, "home_id"], g.at[gid, "away_id"]}
        names = d.team.unique()
        known = {n: (ids.get(n, set()) & pair) for n in names}
        mapped = {n: next(iter(s)) for n, s in known.items() if len(s) == 1}
        left = pair - set(mapped.values())
        for n in names:
            if n not in mapped and len(left) == 1 and len(names) == 2:
                mapped[n] = next(iter(left))
        out.append(d.assign(team_id=d.team.map(mapped)))
    return pd.concat(out).dropna(subset=["team_id"]).astype({"team_id": int})


def features(passing: pd.DataFrame, games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team_id) for every game in ``games`` (played or not)."""
    p = team_ids(passing, games, teams)
    starters = (
        p.sort_values("attempts", ascending=False).drop_duplicates(["game_id", "team_id"])
        [["game_id", "team_id", "player_id"]]
    )  # fmt: skip
    team_pass = p.groupby(["game_id", "team_id"])[["attempts", "pass_yards"]].sum()
    qb_games = p.groupby(["game_id", "player_id"])[["attempts", "pass_yards"]].sum()

    tg = pd.concat([
        games[["game_id", "start_utc", "home_id"]].rename(columns={"home_id": "team_id"}),
        games[["game_id", "start_utc", "away_id"]].rename(columns={"away_id": "team_id"}),
    ]).sort_values("start_utc")  # fmt: skip
    start = starters.set_index(["game_id", "team_id"]).player_id.to_dict()
    kick = games.set_index("game_id").start_utc.to_dict()

    # every player's games in time order, to count starts/efficiency before a date
    qb_hist: dict[str, list] = {}
    for (gid, pid), r in qb_games.iterrows():
        qb_hist.setdefault(pid, []).append((kick.get(gid), gid, r.attempts, r.pass_yards))
    starts_at: dict[str, list] = {}
    for (gid, _team), pid in start.items():
        if kick.get(gid) is not None:
            starts_at.setdefault(pid, []).append(kick[gid])

    rows: list[dict] = []
    last: dict[int, list] = {}
    for r in tg.itertuples():
        prev = last.get(r.team_id, [])  # this team's earlier games: (gid, starter)
        feat = {"game_id": r.game_id, "team_id": r.team_id}
        if prev:
            cur = prev[-1][1]
            feat["qb_new"] = float(len(prev) > 1 and prev[-2][1] != cur)
            hist = [h for h in qb_hist.get(cur, []) if h[0] is not None and h[0] < r.start_utc]
            n_starts = sum(1 for t in starts_at.get(cur, []) if t < r.start_utc)
            att = sum(h[2] for h in hist)
            yds = sum(h[3] for h in hist)
            feat["qb_starts"] = np.log1p(n_starts)
            feat["qb_ypa"] = (yds + PRIOR_YPA * PRIOR_ATT) / (att + PRIOR_ATT)
            recent = [team_pass.loc[(g, r.team_id)] for g, _ in prev[-12:]
                      if (g, r.team_id) in team_pass.index]  # fmt: skip
            t_att = sum(x.attempts for x in recent)
            t_yds = sum(x.pass_yards for x in recent)
            feat["qb_vs_team"] = feat["qb_ypa"] - (t_yds + PRIOR_YPA * PRIOR_ATT) / (
                t_att + PRIOR_ATT
            )
        rows.append(feat)
        s = start.get((r.game_id, r.team_id))
        if s is not None and r.start_utc < pd.Timestamp.now(tz="UTC"):
            last.setdefault(r.team_id, []).append((r.game_id, s))
    return pd.DataFrame(rows)
