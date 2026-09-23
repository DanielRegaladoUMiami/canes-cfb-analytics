"""As-of team features for the points-per-team model.

Every feature for a game is computed from games in **earlier slates only**. A slate is one
(season, season_type, week). Features for a Saturday game never see Thursday's result from
the same week, which also matches what's known when lines are bet midweek.

Only FBS-vs-FBS games feed ratings and form. FCS blowouts would distort them.

Feature families (see docs/pipeline.md §2):
- raw form: points for/against, season to date and last 3, relative to league average
- ridge ratings: opponent-adjusted offense/defense, refit every slate, recency-weighted
- elo: 538-style with log margin-of-victory multiplier and preseason regression
- matchup: interactions and nonlinear transforms of the above
- context: home/neutral, rest, week, bowl, indoor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

SLATE = ["season", "season_type", "week"]


# ---------------------------------------------------------------- team-game table


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game. ``hfa`` is +1 home, -1 away, 0 neutral site."""
    base = [
        "game_id", "season", "season_type", "week", "start_utc", "completed", "shortened",
        "neutral_site", "conference_game", "indoor",
    ]  # fmt: skip
    sides = []
    for side, opp in (("home", "away"), ("away", "home")):
        cols = {
            f"{side}_id": "team_id",
            f"{side}_team": "team",
            f"{opp}_id": "opp_id",
            f"{opp}_team": "opp",
            f"{side}_points": "points",
            f"{opp}_points": "points_allowed",
            f"{side}_fbs": "fbs",
            f"{opp}_fbs": "opp_fbs",
            **{f"{side}_q{i}": f"q{i}" for i in range(1, 5)},
            f"{side}_ot": "ot",
        }
        df = games[base + list(cols)].rename(columns=cols)
        sign = 1 if side == "home" else -1
        df["hfa"] = np.where(games["neutral_site"], 0, sign)
        sides.append(df)
    tg = pd.concat(sides, ignore_index=True)
    tg["fbs_game"] = tg["fbs"] & tg["opp_fbs"]
    slate_start = tg.groupby(SLATE)["start_utc"].transform("min")
    tg["slate_start"] = slate_start
    return tg.sort_values(["start_utc", "game_id", "hfa"], ascending=[True, True, False])


def slates(tg: pd.DataFrame) -> pd.DataFrame:
    return (
        tg.groupby(SLATE, as_index=False)["slate_start"].min().sort_values("slate_start")
    ).reset_index(drop=True)


def _history(tg: pd.DataFrame) -> pd.DataFrame:
    """Completed FBS-vs-FBS team-games: the only rows features learn from."""
    return tg[tg["completed"] & tg["fbs_game"]]


def _fbs_rows(tg: pd.DataFrame) -> pd.DataFrame:
    """FBS-vs-FBS rows, played or scheduled (unplayed rows have NaN points)."""
    rows = tg[tg["fbs_game"]].copy()
    rows.loc[~rows["completed"], ["points", "points_allowed"]] = np.nan
    return rows.sort_values("start_utc")


def _window(hist: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    """Rows from slates strictly before ``start`` and within ``days`` of it."""
    age = start - hist["slate_start"]
    return hist[(age > pd.Timedelta(0)) & (age <= pd.Timedelta(days=days))]


# ---------------------------------------------------------------- raw form


def league_average(tg: pd.DataFrame, window_days: int = 365) -> pd.Series:
    """Average team points over the ``window_days`` before each slate, indexed like slates."""
    hist = _history(tg)[["slate_start", "points"]]
    out = {}
    for start in slates(tg)["slate_start"]:
        recent = _window(hist, start, window_days)
        out[start] = recent["points"].mean() if len(recent) else np.nan
    return pd.Series(out, name="league_avg")


def raw_form(tg: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: prior points for/against this season and over the last 3 games.

    Computed on played and scheduled rows alike: unplayed games have NaN points, which the
    shifted means skip, so upcoming games get the latest form.
    """
    hist = _fbs_rows(tg)
    g = hist.groupby(["team_id", "season"])
    prior = pd.DataFrame(
        {
            "game_id": hist["game_id"],
            "team_id": hist["team_id"],
            "std_pf": g["points"].transform(lambda s: s.shift().expanding().mean()),
            "std_pa": g["points_allowed"].transform(lambda s: s.shift().expanding().mean()),
            "l3_pf": g["points"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean()),
            "l3_pa": g["points_allowed"].transform(
                lambda s: s.shift().rolling(3, min_periods=1).mean()
            ),
            "n_games": g["points"].transform(lambda s: s.shift().notna().cumsum()),
        }
    )
    return prior


# ---------------------------------------------------------------- ridge ratings


def ridge_ratings(
    tg: pd.DataFrame,
    alpha: float = 1.0,
    half_life_days: float = 120.0,
    window_days: int = 600,
    target: str = "points",
) -> pd.DataFrame:
    """Opponent-adjusted offense/defense ratings, refit before every slate.

    ``points = intercept + off[team] - def[opp] + hfa_coef * hfa``

    Fitted on prior FBS-vs-FBS games within ``window_days``, weighted by exponential
    recency decay, so early-season ratings lean on last season (the prior) and move toward
    current form as games accumulate. Ridge shrinks teams with little data toward average.
    Returns one row per (slate, team) with ``off``, ``def`` (higher = better defense),
    ``hfa_pts`` and ``intercept``.

    ``target`` can be any per-team-game offensive stat (plays, EPA per play, success
    rate...). The target is standardized before fitting so ``alpha`` means the same
    shrinkage whatever the stat's scale; coefficients are returned in the stat's units.
    """
    hist = _history(tg)
    hist = hist[hist[target].notna()]
    teams = pd.Index(sorted(tg["team_id"].unique()))
    n = len(teams)
    rows = []
    for _, slate in slates(tg).iterrows():
        start = slate.slate_start
        train = _window(hist, start, window_days)
        if len(train) < 200:
            continue
        t_idx = teams.get_indexer(train["team_id"])
        o_idx = teams.get_indexer(train["opp_id"])
        m = len(train)
        r = np.arange(m)
        x = sparse.hstack(
            [
                sparse.csr_matrix((np.ones(m), (r, t_idx)), shape=(m, n)),
                sparse.csr_matrix((np.ones(m), (r, o_idx)), shape=(m, n)),
                sparse.csr_matrix(train["hfa"].to_numpy(float)[:, None]),
            ]
        ).tocsr()
        age = (start - train["slate_start"]).dt.days.to_numpy()
        w = 0.5 ** (age / half_life_days)
        y = train[target].to_numpy(float)
        mu, sd = y.mean(), y.std() or 1.0
        model = Ridge(alpha=alpha).fit(x, (y - mu) / sd, sample_weight=w)
        coef = model.coef_ * sd
        rows.append(
            pd.DataFrame(
                {
                    **{k: slate[k] for k in SLATE},
                    "team_id": teams,
                    "off": coef[:n],
                    "def": -coef[n : 2 * n],
                    "hfa_pts": coef[-1],
                    "intercept": mu + model.intercept_ * sd,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- elo


def elo_ratings(
    tg: pd.DataFrame,
    k: float = 25.0,
    hfa: float = 55.0,
    revert: float = 1 / 3,
    mean: float = 1500.0,
) -> pd.DataFrame:
    """Pre-game Elo per team-game (538 style).

    Log margin-of-victory multiplier with autocorrelation correction: a 40-point win
    counts little more than a 30-point win, and favorites get less credit for expected
    blowouts. Ratings regress ``revert`` of the way to the mean each offseason.
    """
    games = _fbs_rows(tg)
    games = games[games["hfa"] >= 0].drop_duplicates("game_id")
    elo: dict[int, float] = {}
    season_of: dict[int, int] = {}
    out = []

    def rating(team: int, season: int) -> float:
        r = elo.get(team, mean)
        if team in season_of and season_of[team] != season:
            r = mean + (r - mean) * (1 - revert)
        season_of[team] = season
        return r

    for g in games.itertuples():
        home, away = g.team_id, g.opp_id
        # Neutral site rows have hfa 0; the first one listed is treated as "home" with no edge.
        rh, ra = rating(home, g.season), rating(away, g.season)
        out += [(g.game_id, home, rh, ra), (g.game_id, away, ra, rh)]
        if not g.completed:
            continue  # scheduled: record the pre-game rating, don't update
        diff = rh + hfa * g.hfa - ra
        expected = 1 / (1 + 10 ** (-diff / 400))
        margin = g.points - g.points_allowed
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_diff = diff if margin > 0 else -diff
        mov = np.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        shift = k * mov * (result - expected)
        elo[home], elo[away] = rh + shift, ra - shift
    return pd.DataFrame(out, columns=["game_id", "team_id", "elo", "opp_elo"])


# ---------------------------------------------------------------- v2: CFBD efficiency

# Offensive stats adjusted for opponent (ratings per slate, like points).
ADJUSTED_STATS = {
    "plays": "o_plays",  # tempo
    "ppa": "o_ppa",  # EPA per play
    "sr": "o_successRate",
    "expl": "o_explosiveness",
    "pass_ppa": "o_pass_ppa",
    "rush_ppa": "o_rush_ppa",
}


def attach_advanced(tg: pd.DataFrame, advanced: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Add CFBD offensive stats to team-game rows (joined on game_id + team_id).

    CFBD keys advanced stats by school name. Names are mapped to ids using FBS teams
    only (some names repeat across divisions), and a row only counts if that team
    actually played in that game.
    """
    fbs_ids = set(tg.loc[tg["fbs"], "team_id"])
    name_to_id = teams[teams["team_id"].isin(fbs_ids)].drop_duplicates("team")
    adv = advanced.merge(name_to_id, on="team", how="inner").drop(columns="team")
    cols = ["game_id", "team_id", *ADJUSTED_STATS.values()]
    out = tg.merge(adv[cols], on=["game_id", "team_id"], how="left")
    return out.rename(columns={v: k for k, v in ADJUSTED_STATS.items()})


def adjusted_stats(tg: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    """Opponent-adjusted offense/defense rating for every stat in ADJUSTED_STATS.

    One row per (slate, team): ``{stat}_off``, ``{stat}_def`` (higher = better defense)
    and ``{stat}_base`` (league intercept, for building expectations).
    """
    merged = None
    for stat in ADJUSTED_STATS:
        rr = ridge_ratings(tg, alpha=alpha, target=stat).rename(
            columns={"off": f"{stat}_off", "def": f"{stat}_def", "intercept": f"{stat}_base"}
        )
        rr = rr.drop(columns="hfa_pts")
        merged = rr if merged is None else merged.merge(rr, on=[*SLATE, "team_id"])
    return merged


def add_efficiency_features(tg: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Team offense vs. opponent defense for each stat, plus pace × efficiency."""
    opp_cols = [f"{s}_{side}" for s in ADJUSTED_STATS for side in ("off", "def")]
    opp = adj[[*SLATE, "team_id", *opp_cols]].rename(
        columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in opp_cols}}
    )
    tg = tg.merge(adj, on=[*SLATE, "team_id"], how="left")
    tg = tg.merge(opp, on=[*SLATE, "opp_id"], how="left")
    for s in ADJUSTED_STATS:
        # This offense against this defense, in the stat's own units.
        tg[f"exp_{s}"] = tg[f"{s}_base"] + tg[f"{s}_off"] - tg[f"opp_{s}_def"]
    # Pace × efficiency: expected plays times expected EPA per play (multiplicative).
    tg["exp_pace_x_ppa"] = tg["exp_plays"] * tg["exp_ppa"]
    # Game pace: both offenses' tempo (fast opponents give you more possessions too).
    tg["game_pace"] = tg["plays_off"] + tg["opp_plays_off"]
    # Unit matchups: which way this offense should attack this defense.
    tg["pass_vs_rush_edge"] = tg["exp_pass_ppa"] - tg["exp_rush_ppa"]
    return tg


def add_priors(
    tg: pd.DataFrame, talent: pd.DataFrame, returning: pd.DataFrame, teams: pd.DataFrame
) -> pd.DataFrame:
    """Season-level priors known before kickoff: roster talent and returning production."""
    fbs_ids = set(tg["team_id"])
    name_to_id = teams[teams["team_id"].isin(fbs_ids)].drop_duplicates("team")
    season_prior = (
        talent.merge(returning, on=["season", "team"], how="outer")
        .merge(name_to_id, on="team", how="inner")
        .drop(columns="team")
    )
    cols = ["talent", "ret_ppa", "ret_pass_ppa", "ret_rush_ppa", "ret_usage"]
    tg = tg.merge(season_prior[["season", "team_id", *cols]], on=["season", "team_id"], how="left")
    opp = season_prior[["season", "team_id", "talent", "ret_ppa"]].rename(
        columns={"team_id": "opp_id", "talent": "opp_talent", "ret_ppa": "opp_ret_ppa"}
    )
    tg = tg.merge(opp, on=["season", "opp_id"], how="left")
    tg["talent_diff"] = tg["talent"] - tg["opp_talent"]
    return tg


# ---------------------------------------------------------------- v3: preseason information


def preseason_table(
    portal: pd.DataFrame,
    recruiting: pd.DataFrame,
    coaches: pd.DataFrame,
    teams: pd.DataFrame,
    fbs_ids: set,
    ap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (season, team_id), all known before the season starts.

    - portal: players in/out and net stars (a player without stars counts as 2).
    - recruiting: this season's class points and the 4-year average (classes that make up
      most of the roster).
    - new_coach: the coach who starts the season was hired within the 12 months before it
      (hired after Sep 1 of the previous year). Interims with no games are ignored.
    - ap_points: preseason AP poll points (the week-1 poll, released in August); unranked
      teams get 0 in seasons that have a poll.
    """
    name_to_id = teams[teams["team_id"].isin(fbs_ids)].drop_duplicates("team")
    ids = dict(zip(name_to_id.team, name_to_id.team_id, strict=True))

    moves = portal.assign(stars=portal["stars"].fillna(2))
    incoming = moves.assign(team_id=moves.destination.map(ids)).dropna(subset=["team_id"])
    outgoing = moves.assign(team_id=moves.origin.map(ids)).dropna(subset=["team_id"])
    p_in = incoming.groupby(["season", "team_id"]).agg(
        portal_in=("stars", "size"), portal_in_stars=("stars", "sum")
    )
    p_out = outgoing.groupby(["season", "team_id"]).agg(
        portal_out=("stars", "size"), portal_out_stars=("stars", "sum")
    )
    port = p_in.join(p_out, how="outer").fillna(0)
    port["portal_net_stars"] = port.portal_in_stars - port.portal_out_stars

    rec = recruiting.assign(team_id=recruiting.team.map(ids)).dropna(subset=["team_id"])
    rec = rec.pivot_table(index="team_id", columns="season", values="recruit_points")
    rec_rows = []
    for season in rec.columns:
        window = [c for c in rec.columns if season - 3 <= c <= season]
        rec_rows.append(
            pd.DataFrame(
                {
                    "season": season,
                    "team_id": rec.index,
                    "recruit_points": rec[season].to_numpy(),
                    "recruit_4yr": rec[window].mean(axis=1).to_numpy(),
                }
            )
        )
    rec_long = pd.concat(rec_rows).set_index(["season", "team_id"])

    c = coaches.copy()
    c["hire_date"] = pd.to_datetime(c["hire_date"], utc=True, errors="coerce")
    c = c[c.team_id.isin(fbs_ids)]
    played = c[c.games > 0]
    starters = pd.concat(
        [
            played,
            c[
                ~c.set_index(["season", "team_id"]).index.isin(
                    played.set_index(["season", "team_id"]).index
                )
            ],
        ]
    )
    starters = starters.sort_values("hire_date").drop_duplicates(["season", "team_id"])
    cutoff = pd.to_datetime((starters.season - 1).astype(str) + "-09-01", utc=True)
    starters["new_coach"] = (starters.hire_date >= cutoff).astype(float)
    coach = starters.set_index(["season", "team_id"])[["new_coach"]]

    out = port.join(rec_long, how="outer").join(coach, how="outer").reset_index()
    out["team_id"] = out.team_id.astype(int)
    if ap is not None:
        out = out.merge(
            ap[["season", "team_id", "ap_points"]], on=["season", "team_id"], how="left"
        )
        polled = out.season.isin(set(ap.season))
        out.loc[polled, "ap_points"] = out.loc[polled, "ap_points"].fillna(0)
    else:
        out["ap_points"] = np.nan
    return out


def add_preseason(tg: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    cols = ["portal_in", "portal_out", "portal_net_stars", "recruit_points", "recruit_4yr",
            "new_coach", "ap_points"]  # fmt: skip
    tg = tg.merge(pre[["season", "team_id", *cols]], on=["season", "team_id"], how="left")
    opp_cols = ["portal_net_stars", "recruit_4yr", "new_coach", "ap_points"]
    opp = pre[["season", "team_id", *opp_cols]].rename(
        columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in opp_cols}}
    )
    tg = tg.merge(opp, on=["season", "opp_id"], how="left")
    tg["recruit_4yr_diff"] = tg.recruit_4yr - tg.opp_recruit_4yr
    tg["ap_points_diff"] = tg.ap_points - tg.opp_ap_points
    return tg


# ---------------------------------------------------------------- assemble


def build_features(
    games: pd.DataFrame,
    advanced: pd.DataFrame | None = None,
    talent: pd.DataFrame | None = None,
    returning: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
    preseason: pd.DataFrame | None = None,
    **ridge_kwargs,
) -> pd.DataFrame:
    """Feature table: one row per FBS-vs-FBS team-game with target ``points``.

    Pass the CFBD tables to add the v2 families (efficiency, pace, unit matchups, priors).
    """
    all_tg = team_games(games)
    tg = _fbs_rows(all_tg)

    # League average as of the slate (era normalization).
    lg = league_average(all_tg)
    tg["league_avg"] = tg["slate_start"].map(lg)

    # Raw form, relative to league average.
    form = raw_form(all_tg)
    tg = tg.merge(form, on=["game_id", "team_id"], how="left")
    opp_form = form.rename(
        columns={c: f"opp_{c}" for c in ("std_pf", "std_pa", "l3_pf", "l3_pa", "n_games")}
    ).rename(columns={"team_id": "opp_id"})
    tg = tg.merge(opp_form, on=["game_id", "opp_id"], how="left")
    for c in ("std_pf", "std_pa", "l3_pf", "l3_pa"):
        tg[f"{c}_rel"] = tg[c] - tg["league_avg"]
        tg[f"opp_{c}_rel"] = tg[f"opp_{c}"] - tg["league_avg"]

    # Ridge ratings for team and opponent.
    rr = ridge_ratings(all_tg, **ridge_kwargs)
    tg = tg.merge(rr, on=[*SLATE, "team_id"], how="left")
    opp_rr = rr[[*SLATE, "team_id", "off", "def"]].rename(
        columns={"team_id": "opp_id", "off": "opp_off", "def": "opp_def"}
    )
    tg = tg.merge(opp_rr, on=[*SLATE, "opp_id"], how="left")

    # Elo.
    tg = tg.merge(elo_ratings(all_tg), on=["game_id", "team_id"], how="left")
    tg["elo_diff"] = tg["elo"] - tg["opp_elo"]

    # Matchup: the ratings' own prediction, plus interactions and nonlinear transforms.
    tg["exp_points"] = tg["intercept"] + tg["off"] - tg["opp_def"] + tg["hfa_pts"] * tg["hfa"]
    tg["exp_points_allowed"] = (
        tg["intercept"] + tg["opp_off"] - tg["def"] - tg["hfa_pts"] * tg["hfa"]
    )
    tg["exp_total"] = tg["exp_points"] + tg["exp_points_allowed"]
    tg["exp_margin"] = tg["exp_points"] - tg["exp_points_allowed"]
    tg["off_x_opp_def"] = tg["off"] * tg["opp_def"]  # strength-on-strength vs mismatch
    tg["abs_exp_margin"] = tg["exp_margin"].abs()  # blowout / garbage-time regime
    tg["exp_margin_sq"] = tg["exp_margin"] ** 2
    tg["off_share_of_total"] = tg["exp_points"] / tg["exp_total"]

    # Form after adjustment: were recent games above what the ratings expected?
    tg = tg.sort_values("start_utc")
    tg["resid"] = tg["points"] - tg["exp_points"]  # NaN for unplayed games
    by_team = tg.groupby(["team_id", "season"])["resid"]
    tg["resid_l3"] = by_team.transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    tg["resid_std"] = by_team.transform(lambda s: s.shift().expanding(min_periods=2).std())

    # Context.
    # Rest counts every prior game, FCS opponents included.
    ordered = all_tg.sort_values("start_utc")
    prev = ordered[["game_id", "team_id"]].assign(
        prev_start=ordered.groupby("team_id")["start_utc"].shift()
    )
    tg = tg.merge(prev, on=["game_id", "team_id"], how="left")
    tg["rest_days"] = (tg["start_utc"] - tg["prev_start"]).dt.days.clip(upper=28)
    opp_rest = tg[["game_id", "team_id", "rest_days"]].rename(
        columns={"team_id": "opp_id", "rest_days": "opp_rest_days"}
    )
    tg = tg.merge(opp_rest, on=["game_id", "opp_id"], how="left")
    tg["rest_diff"] = tg["rest_days"] - tg["opp_rest_days"]
    tg["is_bowl"] = (tg["season_type"] == 3).astype(int)
    tg["early_season"] = ((tg["season_type"] == 2) & (tg["week"] <= 4)).astype(int)
    tg["indoor"] = tg["indoor"].fillna(False).astype(int)
    tg["conference_game"] = tg["conference_game"].astype(int)
    tg["neutral_site"] = tg["neutral_site"].astype(int)
    tg = tg.drop(columns=["prev_start", "resid"])

    if advanced is not None and teams is not None:
        with_stats = attach_advanced(all_tg, advanced, teams)
        with_stats.loc[~with_stats["completed"], list(ADJUSTED_STATS)] = np.nan
        tg = add_efficiency_features(tg, adjusted_stats(with_stats))
    if talent is not None and returning is not None and teams is not None:
        tg = add_priors(tg, talent, returning, teams)
    if preseason is not None:
        tg = add_preseason(tg, preseason)
    return tg.reset_index(drop=True)


FEATURE_SETS: dict[str, list[str]] = {
    "raw_form": [
        "league_avg", "hfa", "std_pf_rel", "std_pa_rel", "l3_pf_rel", "l3_pa_rel",
        "opp_std_pf_rel", "opp_std_pa_rel", "opp_l3_pf_rel", "opp_l3_pa_rel", "n_games",
    ],
    "ratings": ["off", "def", "opp_off", "opp_def", "hfa_pts", "intercept", "exp_points"],
    "elo": ["elo", "opp_elo", "elo_diff"],
    "matchup": [
        "exp_points_allowed", "exp_total", "exp_margin", "off_x_opp_def", "abs_exp_margin",
        "exp_margin_sq", "off_share_of_total", "resid_l3", "resid_std",
    ],
    "context": [
        "neutral_site", "conference_game", "rest_days", "opp_rest_days", "rest_diff",
        "is_bowl", "early_season", "indoor", "week",
    ],
}  # fmt: skip

FEATURE_SETS_V2: dict[str, list[str]] = {
    "efficiency": [
        *[f"{s}_{side}" for s in ADJUSTED_STATS for side in ("off", "def")],
        *[f"opp_{s}_{side}" for s in ADJUSTED_STATS for side in ("off", "def")],
        *[f"exp_{s}" for s in ADJUSTED_STATS],
    ],
    "pace_x_efficiency": ["exp_pace_x_ppa", "game_pace", "pass_vs_rush_edge"],
    "priors": [
        "talent", "opp_talent", "talent_diff", "ret_ppa", "opp_ret_ppa", "ret_pass_ppa",
        "ret_rush_ppa", "ret_usage",
    ],
    "preseason": [
        "portal_in", "portal_out", "portal_net_stars", "recruit_points", "recruit_4yr",
        "new_coach", "opp_portal_net_stars", "opp_recruit_4yr", "opp_new_coach",
        "recruit_4yr_diff", "ap_points", "opp_ap_points", "ap_points_diff",
    ],
}  # fmt: skip


def cumulative_sets(include_v2: bool = True) -> dict[str, list[str]]:
    """raw_form, +ratings, ... +priors: each adds one family (v1 then, optionally, v2)."""
    families = {**FEATURE_SETS, **FEATURE_SETS_V2} if include_v2 else FEATURE_SETS
    out, cols = {}, []
    for name, feats in families.items():
        cols = cols + feats
        out[("+" if out else "") + name] = list(cols)
    return out


def add_market_lines(
    features: pd.DataFrame, lines: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Attach consensus lines and each team's market-implied points (benchmark, not a feature).

    Home implied = total/2 - spread/2; away implied = total/2 + spread/2, using ESPN's
    home team (also at neutral sites, where CFBD lists the same home team).
    """
    cols = ["game_id", "spread_close", "total_close", "spread_open", "total_open"]
    out = features.merge(lines[cols], on="game_id", how="left").merge(
        games[["game_id", "home_id"]], on="game_id", how="left"
    )
    is_home = out["team_id"] == out["home_id"]
    for kind, name in (("close", "market_points"), ("open", "market_points_open")):
        total, spread = out[f"total_{kind}"], out[f"spread_{kind}"]
        out[name] = (total / 2 - spread / 2).where(is_home, total / 2 + spread / 2)
    return out.drop(columns="home_id")


def build_all(raw_dir) -> pd.DataFrame:
    """Full feature table (v1 + v2 + market lines) from the parquet files in ``raw_dir``."""

    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(raw_dir / f"{name}.parquet")

    games = read("games")
    teams = read("teams")
    preseason = None
    if all((raw_dir / f"{n}.parquet").exists() for n in ("portal", "recruiting", "coaches")):
        fbs_ids = set(games.loc[games.home_fbs, "home_id"]) | set(
            games.loc[games.away_fbs, "away_id"]
        )
        ap = read("preseason_ap") if (raw_dir / "preseason_ap.parquet").exists() else None
        preseason = preseason_table(
            read("portal"), read("recruiting"), read("coaches"), teams, fbs_ids, ap
        )
    feats = build_features(
        games,
        advanced=read("advanced"),
        talent=read("talent"),
        returning=read("returning"),
        teams=teams,
        preseason=preseason,
    )
    return add_market_lines(feats, read("lines"), games)
