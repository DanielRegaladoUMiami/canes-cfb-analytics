"""Game-time weather at each venue (free sources, no key).

- Played games: hourly observations from the nearest Meteostat weather station with data
  (bulk.meteostat.net, one file per station for all years, cached in data/raw/weather/).
- Upcoming games (up to 16 days out): Open-Meteo forecast, one call per venue. That's a
  forecast, which is what a bettor has before kickoff.

Per game: average wind speed and highest gust (mph), total precipitation (inches) and
average temperature (°F) over the 3.5 hours from kickoff. Indoor games get calm, dry, 70°F
and indoor = 1.
"""

from __future__ import annotations

import json
import re
import time

import httpx
import numpy as np
import pandas as pd

from canes_cfb.paths import RAW

METEOSTAT = "https://bulk.meteostat.net/v2"
MAX_STATION_KM = 60
FORECAST = "https://api.open-meteo.com/v1/forecast"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
HOURLY = "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m"
CACHE = RAW / "weather"
GAME_HOURS = 3.5


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _get(url: str, params: dict) -> dict:
    for attempt in range(5):
        r = httpx.get(url, params=params, timeout=120)
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def venue_coords(games: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """venue → latitude, longitude: CFBD venue by name, else the city's CFBD venues,
    else Open-Meteo geocoding of the city."""
    u = games[["venue", "venue_city", "venue_state"]].dropna(subset=["venue"])
    u = u.drop_duplicates("venue").copy()
    v = venues.dropna(subset=["latitude", "longitude"]).copy()
    v["k"] = v.name.map(_norm)
    by_name = v.drop_duplicates("k").set_index("k")[["latitude", "longitude"]]
    by_city = v.groupby([v.city.map(_norm), v.state.map(_norm)])[["latitude", "longitude"]].mean()
    rows = []
    for r in u.itertuples():
        k, ck = _norm(r.venue), (_norm(r.venue_city), _norm(r.venue_state))
        if k in by_name.index:
            lat, lon = by_name.loc[k]
        elif ck in by_city.index:
            lat, lon = by_city.loc[ck]
        else:
            hits = _get(GEOCODE, {"name": r.venue_city, "count": 5}).get("results", [])
            hit = next((h for h in hits if _norm(h.get("admin1", "")).startswith(
                _norm(r.venue_state)[:2])), hits[0] if hits else None)  # fmt: skip
            if not hit:
                continue
            lat, lon = hit["latitude"], hit["longitude"]
        rows.append({"venue": r.venue, "lat": round(float(lat), 3), "lon": round(float(lon), 3)})
    return pd.DataFrame(rows)


def _hourly(url: str, lat: float, lon: float, extra: dict, cache_name: str) -> pd.DataFrame:
    path = CACHE / cache_name if cache_name else None
    if path is not None and path.exists():
        d = json.loads(path.read_text())
    else:
        d = _get(url, {"latitude": lat, "longitude": lon, "hourly": HOURLY,
                       "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
                       "precipitation_unit": "inch", "timezone": "UTC", **extra})  # fmt: skip
        if path is not None:  # forecasts change; only the archive is cached
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(d))
    h = pd.DataFrame(d["hourly"])
    h["time"] = pd.to_datetime(h["time"], utc=True)
    return h.set_index("time")


def _summarize(h: pd.DataFrame, kickoff: pd.Timestamp) -> dict:
    w = h.loc[kickoff.floor("h") : kickoff + pd.Timedelta(hours=GAME_HOURS)]
    if w.empty or w.wind_speed_10m.isna().all() or w.temperature_2m.isna().all():
        return {}
    return {
        "wind_mph": float(w.wind_speed_10m.mean()),
        "gust_mph": float(w.wind_gusts_10m.max())
        if w.wind_gusts_10m.notna().any()
        else float("nan"),
        "precip_in": float(w.precipitation.sum()),
        "temp_f": float(w.temperature_2m.mean()),
    }


def _stations() -> pd.DataFrame:
    path = CACHE / "stations_lite.json.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(httpx.get(f"{METEOSTAT}/stations/lite.json.gz", timeout=120).content)
    st = pd.read_json(path, compression="gzip")
    inv = st.inventory.map(lambda x: (x or {}).get("hourly") or {})
    return pd.DataFrame(
        {
            "station": st["id"],
            "lat": st.location.map(lambda x: x["latitude"]),
            "lon": st.location.map(lambda x: x["longitude"]),
            "start": pd.to_datetime(inv.map(lambda x: x.get("start")), errors="coerce"),
            "end": pd.to_datetime(inv.map(lambda x: x.get("end")), errors="coerce"),
        }  # fmt: skip
    ).dropna(subset=["start"])


def _km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _station_hourly(station: str, need_after: pd.Timestamp | None = None) -> pd.DataFrame:
    """A station's hourly history; re-downloaded if the cached file predates games we need."""
    path = CACHE / f"meteostat_{station}.csv.gz"
    stale = (
        need_after is not None
        and path.exists()
        and pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC") < need_after
    )
    if not path.exists() or stale:
        r = httpx.get(f"{METEOSTAT}/hourly/{station}.csv.gz", timeout=300)
        if r.status_code != 200:
            return pd.DataFrame()
        path.write_bytes(r.content)
    cols = ["date", "hour", "temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt",
            "pres", "tsun", "coco"]  # fmt: skip
    h = pd.read_csv(path, names=cols, compression="gzip")
    h = h[h.date >= "2015-08-01"]
    h.index = pd.to_datetime(h.date) + pd.to_timedelta(h.hour, unit="h")
    h.index = h.index.tz_localize("UTC")
    return pd.DataFrame(
        {
            "temperature_2m": h.temp * 9 / 5 + 32,
            "precipitation": h.prcp / 25.4,
            "wind_speed_10m": h.wspd * 0.621371,
            "wind_gusts_10m": h.wpgt * 0.621371,
        }  # fmt: skip
    )


def game_weather(games: pd.DataFrame, coords: pd.DataFrame, current_season: int) -> pd.DataFrame:
    """One row per game with wind_mph, gust_mph, precip_in, temp_f, indoor.

    Played games from the nearest stations with observations at kickoff (up to 3 tried,
    within 60 km); upcoming games from the Open-Meteo forecast."""
    g = games.merge(coords, on="venue", how="left").dropna(subset=["lat", "lon"])
    stations = _stations()
    hourly: dict[str, pd.DataFrame] = {}
    rows = []
    played = g[g.completed]
    for (lat, lon), d in played.groupby(["lat", "lon"]):
        near = stations.assign(km=_km(lat, lon, stations.lat, stations.lon))
        near = near[near.km <= MAX_STATION_KM].sort_values("km").head(3)
        for r in d.itertuples():
            got = {}
            for st in near.itertuples():
                if not (st.start <= r.start_utc.tz_localize(None) <= st.end + pd.Timedelta(days=1)):
                    continue
                if st.station not in hourly:
                    newest = d.start_utc.max() + pd.Timedelta(days=2)
                    hourly[st.station] = _station_hourly(st.station, newest)
                got = _summarize(hourly[st.station], r.start_utc) if len(hourly[st.station]) else {}
                if got:
                    break
            rows.append({"game_id": r.game_id, **got})
    upcoming = g[~g.completed & (g.season >= current_season)]
    upcoming = upcoming[upcoming.start_utc <= pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=15)]
    for (lat, lon), d in upcoming.groupby(["lat", "lon"]):
        try:
            h = _hourly(FORECAST, lat, lon, {"forecast_days": 16}, "")
        except httpx.HTTPError:  # forecast unavailable (rate limit): leave these blank
            continue
        for r in d.itertuples():
            rows.append({"game_id": r.game_id, **_summarize(h, r.start_utc)})
    out = pd.DataFrame(rows)
    out = games[["game_id", "indoor"]].merge(out, on="game_id", how="left")
    indoor = out.indoor.fillna(False).astype(bool)
    out.loc[indoor, ["wind_mph", "gust_mph", "precip_in"]] = 0.0
    out.loc[indoor, "temp_f"] = 70.0
    out["indoor"] = indoor.astype(int)
    return out


def load(games: pd.DataFrame, venues: pd.DataFrame, current_season: int) -> pd.DataFrame:
    fbs = games[games.home_fbs | games.away_fbs]
    return game_weather(fbs, venue_coords(fbs, venues), current_season)


__all__ = ["load", "game_weather", "venue_coords", "np"]
