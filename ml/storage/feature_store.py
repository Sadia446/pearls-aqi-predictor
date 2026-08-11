"""Hopsworks-backed feature store access layer.

Provides the two things a feature store exists for:
  - OFFLINE retrieval : historical feature rows, for training and evaluation.
  - ONLINE retrieval  : the single latest feature row per city, for real-time
                        inference (what the dashboard/alerts read).
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ml.storage.hopsworks_store import (
    FEATURES_FG,
    FEATURES_VERSION,
    read_fg,
)


def read_features(
    city_ids: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    stride: int = 1,
) -> pd.DataFrame:
    """OFFLINE retrieval: historical feature rows, optionally filtered.

    city_ids: restrict to these cities (default: all).
    start/end: ISO timestamps to bound event_time (inclusive).
    stride:   keep every Nth row per city (for sampling only, NOT training).
    """
    df = read_fg(FEATURES_FG, version=FEATURES_VERSION)

    if df.empty:
        return df

    # Ensure event_time is datetime with timezone
    if not pd.api.types.is_datetime64_any_dtype(df["event_time"]):
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    if city_ids:
        df = df[df["city_id"].isin(city_ids)]
    if start:
        df = df[df["event_time"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["event_time"] <= pd.Timestamp(end, tz="UTC")]

    df = df.sort_values(["city_id", "event_time"]).reset_index(drop=True)

    if stride > 1:
        # Sample every Nth row per city (for SHAP backgrounds, not training).
        df = (
            df.groupby("city_id", sort=False, group_keys=False)
            .apply(lambda g: g.iloc[::stride])
            .reset_index(drop=True)
        )

    return df


def get_latest_features(
    city_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """ONLINE retrieval: the latest feature row per city (for live inference)."""
    df = read_fg(FEATURES_FG, version=FEATURES_VERSION)

    if df.empty:
        return df

    if not pd.api.types.is_datetime64_any_dtype(df["event_time"]):
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    if city_ids:
        df = df[df["city_id"].isin(city_ids)]

    # Keep the newest row per city
    latest = (
        df.sort_values("event_time")
        .groupby("city_id")
        .tail(1)
        .sort_values("city_id")
        .reset_index(drop=True)
    )
    return latest
