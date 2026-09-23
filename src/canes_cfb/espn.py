"""ESPN college football scoreboard: games, venues and quarter-by-quarter scores.

Free and keyless. Historical seasons carry line scores by quarter back to at least 2015,
but no betting lines (those come from CFBD).

Every response is cached as JSON under ``data/raw/espn/``. Weeks where every game is final
are never re-downloaded; weeks with unfinished games are refreshed on the next pull.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd

from canes_cfb.paths import RAW

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
)
FBS_GROUP = 80
REGULAR, POSTSEASON = 2, 3
CACHE_DIR = RAW / "espn" / "scoreboard"

# A team with at least this many games in a season's FBS scoreboard is FBS that season.
# FCS teams show up only in their one or two games against FBS opponents.
FBS_MIN_GAMES = 6


def _cache_path(season: int, season_type: int, week: int) -> Path:
    return CACHE_DIR / f"{season}_{season_type}_{week}.json"


def _all_final(payload: dict) -> bool:
    events = payload.get("events", [])
    return bool(events) and all(
        e["competitions"][0]["status"]["type"].get("completed", False) for e in events
    )


def fetch_scoreboard(
    season: int,
    season_type: int,
    week: int,
    client: httpx.Client | None = None,
    pause: float = 0.25,
) -> dict:
    """One week of FBS games, from cache when that week is already complete."""
    path = _cache_path(season, season_type, week)
    if path.exists():
        cached = json.loads(path.read_text())
        if _all_final(cached):
            return cached

    params = {
        "groups": FBS_GROUP,
        "dates": season,
        "seasontype": season_type,
        "week": week,
        "limit": 500,
    }
    own_client = client is None
    client = client or httpx.Client(timeout=30)
    try:
        resp = client.get(SCOREBOARD_URL, params=params)
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if own_client:
            client.close()
    time.sleep(pause)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def season_weeks(payload: dict) -> list[tuple[int, int]]:
    """(season_type, week) pairs for regular season and postseason, from ESPN's calendar."""
    weeks = []
    for block in payload["leagues"][0]["calendar"]:
        season_type = int(block["value"])
        if season_type not in (REGULAR, POSTSEASON):
            continue
        weeks += [(season_type, int(entry["value"])) for entry in block["entries"]]
    return weeks


def _side(competitor: dict, prefix: str) -> dict:
    team = competitor["team"]
    periods = [float(ls.get("value", 0)) for ls in competitor.get("linescores", [])]
    quarters = periods[:4] + [None] * (4 - len(periods[:4]))
    score = competitor.get("score")
    return {
        f"{prefix}_id": int(team["id"]),
        f"{prefix}_team": team.get("location") or team.get("displayName"),
        f"{prefix}_abbr": team.get("abbreviation"),
        f"{prefix}_conf_id": int(team["conferenceId"]) if team.get("conferenceId") else None,
        f"{prefix}_points": float(score) if score not in (None, "") else None,
        **{f"{prefix}_q{i}": q for i, q in enumerate(quarters, 1)},
        f"{prefix}_ot": sum(periods[4:]) if len(periods) > 4 else 0.0,
        f"{prefix}_n_periods": len(periods),
    }


def parse_games(payload: dict) -> pd.DataFrame:
    """One row per game, home/away columns. Pure function of the JSON payload."""
    rows = []
    for event in payload.get("events", []):
        comp = event["competitions"][0]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        if set(sides) != {"home", "away"}:
            continue
        venue = comp.get("venue", {})
        status = comp["status"]["type"]
        rows.append(
            {
                "game_id": int(event["id"]),
                "season": int(event["season"]["year"]),
                "season_type": int(event["season"]["type"]),
                "week": int(event.get("week", {}).get("number", 0)),
                "start_utc": pd.Timestamp(comp["date"]),
                "completed": bool(status.get("completed", False)),
                "status": status.get("name"),
                "neutral_site": bool(comp.get("neutralSite", False)),
                "conference_game": bool(comp.get("conferenceCompetition", False)),
                "venue": venue.get("fullName"),
                "venue_city": venue.get("address", {}).get("city"),
                "venue_state": venue.get("address", {}).get("state"),
                "indoor": venue.get("indoor"),
                "attendance": comp.get("attendance"),
                **_side(sides["home"], "home"),
                **_side(sides["away"], "away"),
            }
        )
    games = pd.DataFrame(rows)
    if not games.empty:
        # Weather-called games end before Q4; quarter and half targets don't exist for them.
        periods = games[["home_n_periods", "away_n_periods"]].min(axis=1)
        games["shortened"] = games["completed"] & (periods < 4)
    return games


def flag_fbs(games: pd.DataFrame, min_games: int = FBS_MIN_GAMES) -> pd.DataFrame:
    """Add home_fbs / away_fbs: whether each team is FBS in that season."""
    appearances = pd.concat(
        [
            games[["season", "home_id"]].rename(columns={"home_id": "team_id"}),
            games[["season", "away_id"]].rename(columns={"away_id": "team_id"}),
        ]
    ).value_counts()
    fbs = set(appearances[appearances >= min_games].index)
    out = games.copy()
    out["home_fbs"] = [(s, t) in fbs for s, t in zip(out["season"], out["home_id"], strict=True)]
    out["away_fbs"] = [(s, t) in fbs for s, t in zip(out["season"], out["away_id"], strict=True)]
    return out


def load_seasons(seasons: list[int], pause: float = 0.25) -> pd.DataFrame:
    """Every regular-season and postseason FBS game for the given seasons."""
    frames = []
    with httpx.Client(timeout=30) as client:
        for season in seasons:
            first = fetch_scoreboard(season, REGULAR, 1, client=client, pause=pause)
            for season_type, week in season_weeks(first):
                payload = fetch_scoreboard(season, season_type, week, client=client, pause=pause)
                frames.append(parse_games(payload))
    games = pd.concat(frames, ignore_index=True).drop_duplicates("game_id", keep="last")
    return flag_fbs(games).sort_values(["start_utc", "game_id"]).reset_index(drop=True)
