"""US EPA AQI categories.

Turns a raw AQI number into the health category people actually recognise —
the colour-coded bands that decide whether it's safe to go outside. These are
the official US EPA breakpoints (the same scale AQICN and Open-Meteo's us_aqi
use), so our forecasts speak the language users already know.
"""
from __future__ import annotations

# (upper_bound, label, hex_colour) — first band whose bound covers the value wins.
_BANDS: list[tuple[float, str, str]] = [
    (50, "Good", "#009966"),
    (100, "Moderate", "#ffde33"),
    (150, "Unhealthy for Sensitive Groups", "#ff9933"),
    (200, "Unhealthy", "#cc0033"),
    (300, "Very Unhealthy", "#660099"),
    (10_000, "Hazardous", "#7e0023"),
]


def aqi_category(aqi: float) -> str:
    """Health-category label for an AQI value."""
    for upper, label, _ in _BANDS:
        if aqi <= upper:
            return label
    return "Hazardous"


def aqi_color(aqi: float) -> str:
    """Official colour for an AQI value (for the dashboard)."""
    for upper, _, color in _BANDS:
        if aqi <= upper:
            return color
    return _BANDS[-1][2]


def is_hazardous_for_sensitive(aqi: float) -> bool:
    """True once air is risky for health-sensitive people (AQI > 100)."""
    return aqi > 100
