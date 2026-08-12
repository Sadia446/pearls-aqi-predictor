"""The cities AeroSense forecasts for.

The list itself lives in `cities.json` at the project root — a single source of
truth shared with the web app, so the two can never drift. Adding a city is:

    1. append it to cities.json
    2. python -m ml.pipelines.backfill --cities <id>
    3. python -m ml.pipelines.training_pipeline

Because the model treats `city` as a *feature* (one model for all cities, not one
per city), a new city needs no architecture change — and every city's data makes
the shared model a little better.

`featured` cities lead the web app's home page; every city is forecastable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CITIES_FILE = Path(__file__).resolve().parents[2] / "cities.json"


@dataclass(frozen=True)
class City:
    id: str        # stable slug / primary key, e.g. "lahore"
    name: str      # display name, e.g. "Lahore"
    country: str
    lat: float
    lon: float
    timezone: str  # IANA tz, e.g. "Asia/Karachi"
    featured: bool = False


def _load() -> list[City]:
    payload = json.loads(CITIES_FILE.read_text(encoding="utf-8"))
    return [
        City(
            id=c["id"],
            name=c["name"],
            country=c["country"],
            lat=c["lat"],
            lon=c["lon"],
            timezone=c["timezone"],
            featured=c.get("featured", False),
        )
        for c in payload["cities"]
    ]


CITIES: list[City] = _load()


def get_city(city_id: str) -> City:
    for city in CITIES:
        if city.id == city_id:
            return city
    raise KeyError(f"Unknown city id: {city_id!r}")


def resolve(city_ids: list[str] | None) -> list[City]:
    """Cities named by id, or all of them when nothing is specified."""
    if not city_ids:
        return CITIES
    return [get_city(cid) for cid in city_ids]
