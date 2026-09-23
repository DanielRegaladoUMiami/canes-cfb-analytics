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
) -> pd.DataFrame:
    """Opponent-adjusted offense/defense ratings, refit before every slate.

    ``points = intercept + off[team] - def[opp] + hfa_coef * hfa``

    Fitted on prior FBS-vs-FBS games within ``window_days``, weighted by exponential
    recency decay, so early-season ratings lean on last season (the prior) and move toward
    current form as games accumulate. Ridge shrinks teams with little data toward average.
    Returns one row per (slate, team) with ``off``, ``def`` (higher = better defense),
    ``hfa_pts`` and ``intercept``.
    """
    hist = _history(tg)
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
        model = Ridge(alpha=alpha).fit(x, train["points"].to_numpy(), sample_weight=w)
        coef = model.coef_
        rows.append(
            pd.DataFrame(
                {
                    **{k: slate[k] for k in SLATE},
                    "team_id": teams,
                    "off": coef[:n],
                    "def": -coef[n : 2 * n],
                    "hfa_pts": coef[-1],
                    "intercept": model.intercept_,
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


# ---------------------------------------------------------------- assemble


def build_features(games: pd.DataFrame, **ridge_kwargs) -> pd.DataFrame:
    """Feature table: one row per FBS-vs-FBS team-game with target ``points``."""
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
    return tg.drop(columns=["prev_start", "resid"]).reset_index(drop=True)


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


def cumulative_sets() -> dict[str, list[str]]:
    """raw_form, +ratings, +elo, +matchup, +context: each adds one family."""
    out, cols = {}, []
    for name, feats in FEATURE_SETS.items():
        cols = cols + feats
        out[("+" if out else "") + name] = list(cols)
    return out
