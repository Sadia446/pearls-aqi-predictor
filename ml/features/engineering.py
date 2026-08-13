"""Feature engineering — raw hourly readings -> model-ready features.

The model can't learn from a single reading in isolation; air quality is driven
by *momentum* (what it's been doing) and *context* (time of day, season, weather).
This module adds four families of features to the raw hourly data:

  1. Lag features     — the AQI/PM2.5 an hour/day ago. Air quality is highly
                        autocorrelated, so "recent past" is the strongest signal.
  2. Rolling features — rolling mean/std/max over recent windows. Captures the
                        local trend and volatility.
  3. Time features    — hour, day-of-week, month, weekend, plus cyclical
                        (sin/cos) encodings so the model knows 23:00 is next to
                        00:00. Uses *local* time so rush-hour patterns line up.
  4. Derived features — change rates, PM2.5/PM10 ratio, calm-wind flag: physical
                        signals that pollution is building up or dispersing.

IMPORTANT — no leakage: every feature uses only the current or *past* readings,
which are all known at prediction time. The future AQI *target* is created later
(in the training pipeline), not here.

All lag/rolling work is done *per city* (groupby) so one city's history never
bleeds into another's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.common.cities import get_city

# What to lag, and by how many hours.
LAG_COLUMNS = ["aqi", "pm25"]
LAG_HOURS = [1, 3, 6, 24]

# What to roll, and over which windows (hours).
ROLL_COLUMNS = ["aqi", "pm25"]
ROLL_WINDOWS = [3, 6, 24]


def _add_lag_features(g: pd.DataFrame) -> pd.DataFrame:
    """Value of each column N hours ago (rows are contiguous hourly, so shift == hours)."""
    for col in LAG_COLUMNS:
        for h in LAG_HOURS:
            g[f"{col}_lag_{h}h"] = g[col].shift(h)
    return g


def _add_rolling_features(g: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean/std/max — captures recent trend and volatility."""
    for col in ROLL_COLUMNS:
        for w in ROLL_WINDOWS:
            roll = g[col].rolling(window=w, min_periods=2)
            g[f"{col}_roll_mean_{w}h"] = roll.mean()
            g[f"{col}_roll_std_{w}h"] = roll.std()
            g[f"{col}_roll_max_{w}h"] = roll.max()
    return g


def _add_time_features(g: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Calendar features in the city's *local* time, plus cyclical encodings."""
    local = g["time"].dt.tz_convert(timezone)

    g["hour"] = local.dt.hour
    g["day_of_week"] = local.dt.dayofweek
    g["month"] = local.dt.month
    g["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)

    # Cyclical encodings: so the model sees 23:00 and 00:00 as neighbours,
    # and December and January as neighbours.
    g["hour_sin"] = np.sin(2 * np.pi * g["hour"] / 24)
    g["hour_cos"] = np.cos(2 * np.pi * g["hour"] / 24)
    g["month_sin"] = np.sin(2 * np.pi * g["month"] / 12)
    g["month_cos"] = np.cos(2 * np.pi * g["month"] / 12)

    # Wind direction is also circular (0° == 360°).
    g["wind_deg_sin"] = np.sin(2 * np.pi * g["wind_deg"] / 360)
    g["wind_deg_cos"] = np.cos(2 * np.pi * g["wind_deg"] / 360)
    return g


def _add_derived_features(g: pd.DataFrame) -> pd.DataFrame:
    """Physical signals that pollution is building up or clearing out."""
    g["aqi_change_1h"] = g["aqi"] - g["aqi_lag_1h"]
    g["aqi_change_24h"] = g["aqi"] - g["aqi_lag_24h"]
    # PM2.5 vs PM10 mix hints at the pollution source (combustion vs dust).
    g["pm_ratio"] = g["pm25"] / g["pm10"].replace(0, np.nan)
    # Calm, humid air traps pollutants; strong wind disperses them.
    g["is_calm_wind"] = (g["wind_speed"] < 1.5).astype(int)
    return g


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add all feature families to a raw hourly frame (one or many cities).

    Input columns (from OpenMeteoClient.get_history):
        city_id, time, aqi, pm25, pm10, o3, no2, so2, co,
        temp_c, humidity, wind_speed, wind_deg, pressure, precip, clouds

    Output: the same rows with ~40 engineered feature columns added. The first
    24 rows of each city will have NaNs in the 24h-lag columns — that's expected
    and handled downstream (dropped or imputed at training time).
    """
    if raw.empty:
        # An upstream gap (data lag, a bad window) must surface as a clear error,
        # not a confusing "No objects to concatenate" from pd.concat([]).
        raise ValueError("No raw rows to build features from — upstream returned nothing.")

    frames: list[pd.DataFrame] = []
    for city_id, group in raw.groupby("city_id", sort=False):
        g = group.sort_values("time").reset_index(drop=True)
        g = _add_lag_features(g)
        g = _add_rolling_features(g)
        g = _add_time_features(g, get_city(str(city_id)).timezone)
        g = _add_derived_features(g)
        frames.append(g)

    return pd.concat(frames, ignore_index=True)
