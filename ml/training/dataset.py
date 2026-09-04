"""Turn the feature store into a supervised learning problem.

A feature row describes conditions *at* time T. To learn forecasting, we attach
the answer: the AQI that actually occurred at T + 24h / 48h / 72h. Those become
the model's targets. This is the single most important step conceptually — it's
what teaches the model to predict the future instead of describe the present.

We also:
  - one-hot encode `city_id` so one model can serve all cities, and
  - split train/test by *time* (never randomly) — the test set is the most
    recent slice, so we measure how well we'd have forecast the near future
    from the past. A random split would leak future information into training.
"""
from __future__ import annotations

import pandas as pd

# The forecast horizons, in hours. The product promises "next 3 days".
HORIZONS: tuple[int, ...] = (24, 48, 72)

# Columns that are identifiers, not features.
KEY_COLS = ["city_id", "event_time"]


def add_targets(features: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Add target_aqi_{h}h columns: the AQI h hours into the future, per city.

    shift(-h) pulls a future row's AQI back onto the current row. Done per city
    so one city's future never leaks into another's.
    """
    out: list[pd.DataFrame] = []
    for _, group in features.groupby("city_id", sort=False):
        g = group.sort_values("event_time").reset_index(drop=True)
        for h in horizons:
            g[f"target_aqi_{h}h"] = g["aqi"].shift(-h)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def feature_columns(data: pd.DataFrame) -> list[str]:
    """The numeric feature columns: everything except keys and targets."""
    return [
        c
        for c in data.columns
        if c not in KEY_COLS and not c.startswith("target_aqi_")
    ]


def build_design_matrix(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Model input X: numeric features + one-hot city columns."""
    X = data[feature_cols].copy()
    city_dummies = pd.get_dummies(data["city_id"], prefix="city")
    return pd.concat([X, city_dummies], axis=1)


def time_split_mask(
    data: pd.DataFrame, test_frac: float = 0.2, max_horizon_h: int = 0
) -> tuple[pd.Series, pd.Series]:
    """Boolean train/test masks split at a single calendar cutoff.

    The most recent `test_frac` of the timeline is the test set, shared across
    all cities so the boundary is a real point in time.

    `max_horizon_h` reserves a buffer at the very end of the timeline: test
    rows within that buffer of the last observed timestamp can't have a real
    target_aqi_{h}h yet (there's no future data to pull it from), so they're
    excluded from the test window rather than silently becoming an empty
    (or shrinking) test set for the longer horizons.
    """
    unique_times = data["event_time"].drop_duplicates().sort_values()
    max_time = unique_times.max()
    buffer_cutoff = max_time - pd.Timedelta(hours=max_horizon_h)

    eligible = unique_times[unique_times <= buffer_cutoff]
    if len(eligible) < 2:
        raise ValueError(
            f"Not enough history to reserve a {max_horizon_h}h buffer before "
            f"the test cutoff. Data spans up to {max_time}, but the buffer "
            f"requires data at or before {buffer_cutoff}. Collect more "
            f"history, or reduce the max forecast horizon."
        )

    cutoff = eligible.iloc[int(len(eligible) * (1 - test_frac))]
    train_mask = data["event_time"] < cutoff
    test_mask = (data["event_time"] >= cutoff) & (data["event_time"] <= buffer_cutoff)
    return train_mask, test_mask
