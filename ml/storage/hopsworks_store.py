"""Hopsworks connection + feature group helpers.

Central gateway to the Hopsworks serverless platform. Every module that needs
the Feature Store or Model Registry comes through here.

Dual-mode support:
  - Cloud / CI / with Hopsworks: connects directly to Hopsworks Feature Store & Model Registry.
  - Local fallback: If Hopsworks package is unavailable or keys are unconfigured,
    stores feature groups locally in `data/feature_store/` so the full pipeline
    runs locally without any C++ build tool installation dependencies.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import pandas as pd

from ml.common.config import PROJECT_ROOT, get_settings

try:
    import hopsworks
    _HAS_HOPSWORKS = True
except ImportError:
    _HAS_HOPSWORKS = False

FEATURES_FG = "aqi_features"
FEATURES_VERSION = 1
PREDICTIONS_FG = "predictions"
PREDICTIONS_VERSION = 1
ALERTS_FG = "alerts"
ALERTS_VERSION = 1
DRIVERS_FG = "forecast_drivers"
DRIVERS_VERSION = 1

LOCAL_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"


def _get_local_file(name: str) -> Path:
    LOCAL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_STORE_DIR / f"{name}.parquet"


@functools.lru_cache(maxsize=1)
def get_project():
    """Cached Hopsworks project — logs in once per process."""
    if not _HAS_HOPSWORKS:
        return None
    settings = get_settings()
    if not settings.hopsworks_api_key:
        return None
    try:
        return hopsworks.login(
            api_key_value=settings.hopsworks_api_key,
            project=settings.hopsworks_project,
        )
    except Exception as exc:
        print(f"  [Hopsworks] Login note: {exc} — using local feature storage.")
        return None


def get_feature_store():
    """The project's feature store."""
    proj = get_project()
    if proj:
        try:
            return proj.get_feature_store()
        except Exception:
            return None
    return None


def get_model_registry():
    """The project's model registry."""
    proj = get_project()
    if proj:
        try:
            return proj.get_model_registry()
        except Exception:
            return None
    return None


def write_features(df: pd.DataFrame) -> int:
    """Write a features DataFrame to the Hopsworks feature store (or local fallback)."""
    fs = get_feature_store()
    if fs:
        try:
            fg = fs.get_or_create_feature_group(
                name=FEATURES_FG,
                version=FEATURES_VERSION,
                primary_key=["city_id", "event_time"],
                event_time="event_time",
                time_travel_format="HUDI",
                description="Hourly AQI and weather features",
            )
            fg.insert(df, overwrite=False, write_options={"wait_for_job": True})
            return len(df)
        except Exception as exc:
            print(f"  [Hopsworks] Insert fallback: {exc}")

    # Local fallback
    path = _get_local_file(FEATURES_FG)
    df.to_parquet(path, index=False)
    return len(df)


def upsert_features(df: pd.DataFrame) -> int:
    """Upsert a recent window of features (idempotent hourly refresh)."""
    fs = get_feature_store()
    if fs:
        try:
            fg = fs.get_or_create_feature_group(
                name=FEATURES_FG,
                version=FEATURES_VERSION,
                primary_key=["city_id", "event_time"],
                event_time="event_time",
                time_travel_format="HUDI",
            )
            fg.insert(df, overwrite=False, write_options={"wait_for_job": True})
            return len(df)
        except Exception as exc:
            print(f"  [Hopsworks] Upsert fallback: {exc}")

    # Local fallback: merge by (city_id, event_time)
    path = _get_local_file(FEATURES_FG)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["city_id", "event_time"], keep="last")
        combined.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return len(df)


def write_predictions(df: pd.DataFrame) -> int:
    """Write predictions to Hopsworks and local fallback."""
    # Always persist locally first so dashboard/alerts have zero latency
    path = _get_local_file(PREDICTIONS_FG)
    df.to_parquet(path, index=False)

    fs = get_feature_store()
    if fs:
        try:
            fg = fs.get_or_create_feature_group(
                name=PREDICTIONS_FG,
                version=PREDICTIONS_VERSION,
                primary_key=["city_id", "horizon_h"],
                event_time="forecast_time",
                time_travel_format="HUDI",
            )
            fg.insert(df, overwrite=False, write_options={"wait_for_job": True})
        except Exception as exc:
            print(f"  [Hopsworks] Prediction insert fallback: {exc}")

    return len(df)


def write_alerts(df: pd.DataFrame) -> int:
    """Write alerts to Hopsworks and local fallback."""
    if df.empty:
        return 0

    path = _get_local_file(ALERTS_FG)
    df.to_parquet(path, index=False)

    fs = get_feature_store()
    if fs:
        try:
            fg = fs.get_or_create_feature_group(
                name=ALERTS_FG,
                version=ALERTS_VERSION,
                primary_key=["city_id"],
                event_time="alert_time",
                time_travel_format="HUDI",
            )
            fg.insert(df, overwrite=False, write_options={"wait_for_job": True})
        except Exception as exc:
            print(f"  [Hopsworks] Alerts insert fallback: {exc}")

    return len(df)


def write_drivers(df: pd.DataFrame) -> int:
    """Write forecast drivers (SHAP) to Hopsworks and local fallback."""
    if df.empty:
        return 0

    path = _get_local_file(DRIVERS_FG)
    df.to_parquet(path, index=False)

    fs = get_feature_store()
    if fs:
        try:
            fg = fs.get_or_create_feature_group(
                name=DRIVERS_FG,
                version=DRIVERS_VERSION,
                primary_key=["city_id", "feature"],
                time_travel_format="HUDI",
            )
            fg.insert(df, overwrite=False, write_options={"wait_for_job": True})
        except Exception as exc:
            print(f"  [Hopsworks] Drivers insert fallback: {exc}")

    return len(df)


def read_fg(name: str, version: int = 1) -> pd.DataFrame:
    """Read all rows from a feature group with automatic local fallback."""
    fs = get_feature_store()
    if fs:
        try:
            fg = fs.get_feature_group(name, version=version)
            df = fg.read()
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            print(f"  [Hopsworks] read_fg('{name}') failed, using local fallback: {exc}") 

    path = _get_local_file(name)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def table_summary() -> pd.DataFrame:
    """Row counts and time span per city — a quick sanity check after a backfill."""
    df = read_fg(FEATURES_FG, FEATURES_VERSION)
    if df.empty:
        return df
    return (
        df.groupby("city_id")
        .agg(
            rows=("event_time", "count"),
            first_hour=("event_time", "min"),
            last_hour=("event_time", "max"),
            avg_aqi=("aqi", "mean"),
            max_aqi=("aqi", "max"),
        )
        .round(0)
        .reset_index()
        .sort_values("city_id")
    )
