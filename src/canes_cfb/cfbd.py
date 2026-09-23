"""CollegeFootballData (CFBD) client: betting lines, advanced game stats, talent, returning
production.

The API key is read from the ``CFBD_API_KEY`` environment variable (set in ``~/.zshrc``).
It is never written to disk, logged, or included in cached files.

The free tier allows a limited number of calls per month, so every response is cached as
JSON under ``data/raw/cfbd/`` and past seasons are never requested twice. CFBD game ids
are the same as ESPN's, so tables join on ``game_id`` directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from canes_cfb.paths import RAW

BASE_URL = "https://api.collegefootballdata.com"
CACHE_DIR = RAW / "cfbd"


def _cache_path(endpoint: str, params: dict) -> Path:
    key = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    return CACHE_DIR / endpoint.strip("/").replace("/", "_") / f"{key}.json"


def get(endpoint: str, refresh: bool = False, **params) -> list[dict]:
    """GET an endpoint, from cache unless ``refresh``. Refresh only in-progress seasons."""
    path = _cache_path(endpoint, params)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("CFBD_API_KEY is not set. Add it to ~/.zshrc and restart the shell.")
    resp = httpx.get(
        BASE_URL + endpoint,
        params=params,
        headers={"Authorization": f"Bearer {key}"},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def _refresh(season: int, current_season: int | None) -> bool:
    return current_season is not None and season >= current_season


# ---------------------------------------------------------------- betting lines


def _median_over(books: list[dict]):
    def med(field: str) -> float:
        vals = [b[field] for b in books if b.get(field) is not None]
        return float(np.median(vals)) if vals else np.nan

    return med


def parse_lines(payload: list[dict]) -> pd.DataFrame:
    """One row per game: consensus (median across books) close and open lines.

    CFBD spreads are the home team's spread (negative = home favored), matching
    docs/markets.md.
    """
    rows = []
    for game in payload:
        books = game.get("lines") or []
        if not books:
            continue
        med = _median_over(books)
        rows.append(
            {
                "game_id": int(game["id"]),
                "spread_close": med("spread"),
                "spread_open": med("spreadOpen"),
                "total_close": med("overUnder"),
                "total_open": med("overUnderOpen"),
                "home_ml": med("homeMoneyline"),
                "away_ml": med("awayMoneyline"),
                "n_books": len(books),
                "providers": ",".join(sorted(b["provider"] for b in books)),
            }
        )
    return pd.DataFrame(rows)


def load_lines(seasons: list[int], current_season: int | None = None) -> pd.DataFrame:
    frames = []
    for season in seasons:
        for season_type in ("regular", "postseason"):
            payload = get(
                "/lines",
                refresh=_refresh(season, current_season),
                year=season,
                seasonType=season_type,
            )
            frames.append(parse_lines(payload))
    return pd.concat(frames, ignore_index=True).drop_duplicates("game_id", keep="last")


# ---------------------------------------------------------------- advanced game stats

_UNIT_FIELDS = ("ppa", "successRate", "explosiveness")


def _flatten(side: dict, prefix: str) -> dict:
    out = {
        f"{prefix}_plays": side.get("plays"),
        f"{prefix}_drives": side.get("drives"),
        f"{prefix}_line_yards": side.get("lineYards"),
        f"{prefix}_stuff_rate": side.get("stuffRate"),
        f"{prefix}_power_success": side.get("powerSuccess"),
    }
    for field in _UNIT_FIELDS:
        out[f"{prefix}_{field}"] = side.get(field)
        out[f"{prefix}_pass_{field}"] = (side.get("passingPlays") or {}).get(field)
        out[f"{prefix}_rush_{field}"] = (side.get("rushingPlays") or {}).get(field)
    return out


def parse_advanced(payload: list[dict]) -> pd.DataFrame:
    """One row per team-game: offense and defense efficiency, with pass/rush splits."""
    rows = [
        {
            "game_id": int(r["gameId"]),
            "team": r["team"],
            **_flatten(r.get("offense") or {}, "o"),
            **_flatten(r.get("defense") or {}, "d"),
        }
        for r in payload
    ]
    return pd.DataFrame(rows)


def load_advanced(seasons: list[int], current_season: int | None = None) -> pd.DataFrame:
    frames = [
        parse_advanced(
            get("/stats/game/advanced", refresh=_refresh(season, current_season), year=season)
        )
        for season in seasons
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates(["game_id", "team"])


# ---------------------------------------------------------------- season-level priors


def load_talent(seasons: list[int]) -> pd.DataFrame:
    rows = [r for s in seasons for r in get("/talent", year=s)]
    return pd.DataFrame(rows).rename(columns={"year": "season"})[["season", "team", "talent"]]


def load_returning(seasons: list[int]) -> pd.DataFrame:
    rows = [r for s in seasons for r in get("/player/returning", year=s)]
    df = pd.DataFrame(rows)
    return df[
        ["season", "team", "percentPPA", "percentPassingPPA", "percentRushingPPA", "usage"]
    ].rename(
        columns={
            "percentPPA": "ret_ppa",
            "percentPassingPPA": "ret_pass_ppa",
            "percentRushingPPA": "ret_rush_ppa",
            "usage": "ret_usage",
        }
    )


def load_teams() -> pd.DataFrame:
    """CFBD team ids (same as ESPN ids) and school names, for joining name-keyed tables."""
    df = pd.DataFrame(get("/teams"))
    return df[["id", "school"]].rename(columns={"id": "team_id", "school": "team"})


# ---------------------------------------------------------------- preseason information


def load_portal(seasons: list[int]) -> pd.DataFrame:
    """Transfer portal entries: one row per player move (origin → destination) per season."""
    rows = [r for s in seasons for r in get("/player/portal", year=s)]
    cols = ["season", "origin", "destination", "stars", "rating", "transferDate"]
    return pd.DataFrame(rows, columns=cols)


def load_recruiting(seasons: list[int]) -> pd.DataFrame:
    """Team recruiting class points per season (the class signed before that season)."""
    rows = [r for s in seasons for r in get("/recruiting/teams", year=s)]
    return pd.DataFrame(rows)[["year", "team", "points"]].rename(
        columns={"year": "season", "points": "recruit_points"}
    )


def load_coaches(seasons: list[int]) -> pd.DataFrame:
    """One row per (coach, team, season) with the coach's hire date and games coached."""
    rows = []
    for s in seasons:
        for coach in get("/coaches", year=s):
            for season in coach.get("seasons", []):
                rows.append(
                    {
                        "season": season["year"],
                        "team_id": season["teamId"],
                        "coach": f"{coach['firstName']} {coach['lastName']}",
                        "hire_date": coach.get("hireDate"),
                        "games": season.get("games", 0),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(["season", "team_id", "coach"])


def load_preseason_ap(seasons: list[int]) -> pd.DataFrame:
    """Preseason AP Top 25 (the week-1 poll, released in August): points per ranked team."""
    rows = []
    for s in seasons:
        for week in get("/rankings", year=s, seasonType="regular", week=1):
            for poll in week.get("polls", []):
                if poll.get("poll") != "AP Top 25":
                    continue
                for r in poll.get("ranks", []):
                    rows.append({"season": s, "team_id": r["teamId"], "ap_points": r["points"],
                                 "ap_rank": r["rank"]})  # fmt: skip
    return pd.DataFrame(rows)
