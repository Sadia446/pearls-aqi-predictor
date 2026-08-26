"""AeroSense forecast API.

A read-only JSON layer over the forecasts the pipelines produce. It computes
nothing itself: the feature, training and inference pipelines write to the
Hopsworks Feature Store, and this serves what they wrote.

Run locally:

    pip install -r api/requirements.txt
    uvicorn api.main:app --reload

Interactive docs are then at http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ml.common.cities import CITIES
from ml.storage.feature_store import get_latest_features
from ml.storage.hopsworks_store import (
    ALERTS_FG,
    ALERTS_VERSION,
    DRIVERS_FG,
    DRIVERS_VERSION,
    PREDICTIONS_FG,
    PREDICTIONS_VERSION,
    read_fg,
)
from ml.storage.registry import get_best_models

app = FastAPI(
    title="AeroSense Forecast API",
    description=(
        "Three-day air-quality forecasts for Pakistani cities, produced by an "
        "automated machine-learning pipeline. Read-only."
    ),
    version="1.0.0",
)

# The forecasts are public data, so any origin may read them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/", summary="Service description")
def root() -> dict:
    return {
        "service": "AeroSense Forecast API",
        "description": "3-day air-quality forecasts, updated hourly.",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/cities",
            "/current/{city_id}",
            "/forecast/{city_id}",
            "/alerts",
            "/models",
        ],
    }


@app.get("/health", summary="Liveness and data freshness")
def health() -> dict:
    """Confirms storage is reachable and reports how fresh the data is."""
    try:
        latest = get_latest_features()
        preds = read_fg(PREDICTIONS_FG, PREDICTIONS_VERSION)
        latest_obs = str(latest["event_time"].max()) if not latest.empty else None
        latest_fc = str(preds["base_time"].max()) if not preds.empty and "base_time" in preds.columns else None
        cities_count = int(latest["city_id"].nunique()) if not latest.empty else 0
    except Exception as exc:  # noqa: BLE001 — surfaced as a 503, not a crash
        raise HTTPException(503, f"Storage unavailable: {type(exc).__name__}") from exc
    return {
        "status": "ok",
        "latest_observation": latest_obs,
        "latest_forecast": latest_fc,
        "cities": cities_count,
    }


@app.get("/cities", summary="Cities covered, with current AQI")
def cities() -> list[dict]:
    latest = get_latest_features()
    if latest.empty:
        return []
    cols = ["city_id", "event_time", "aqi", "pm25", "temp_c"]
    available_cols = [c for c in cols if c in latest.columns]
    res = latest[available_cols].rename(columns={"event_time": "observed_at"}).sort_values("aqi", ascending=False)
    # Convert timestamps to string
    res["observed_at"] = res["observed_at"].astype(str)
    return res.to_dict(orient="records")


@app.get("/current/{city_id}", summary="Latest measured conditions for a city")
def current(city_id: str) -> dict:
    latest = get_latest_features(city_ids=[city_id])
    if latest.empty:
        raise HTTPException(404, f"Unknown city '{city_id}'. See /cities.")
    row = latest.iloc[0].to_dict()
    row["observed_at"] = str(row.pop("event_time", None))
    return row


@app.get("/forecast/{city_id}", summary="3-day forecast, with the drivers behind it")
def forecast(city_id: str) -> dict:
    preds = read_fg(PREDICTIONS_FG, PREDICTIONS_VERSION)
    if preds.empty or city_id not in preds["city_id"].values:
        raise HTTPException(404, f"No forecast for '{city_id}'. See /cities.")

    horizons = preds[preds["city_id"] == city_id].sort_values("horizon_h").to_dict(orient="records")
    for h in horizons:
        if "forecast_time" in h:
            h["forecast_time"] = str(h["forecast_time"])
        if "base_time" in h:
            h["base_time"] = str(h["base_time"])

    drivers_df = read_fg(DRIVERS_FG, DRIVERS_VERSION)
    drivers = []
    if not drivers_df.empty and city_id in drivers_df["city_id"].values:
        city_drivers = drivers_df[drivers_df["city_id"] == city_id]
        if "contribution" in city_drivers.columns:
            city_drivers = city_drivers.reindex(city_drivers["contribution"].abs().sort_values(ascending=False).index)[:5]
        drivers = city_drivers.to_dict(orient="records")

    latest = get_latest_features(city_ids=[city_id])
    now_aqi = float(latest["aqi"].iloc[0]) if not latest.empty else None
    now_time = str(latest["event_time"].iloc[0]) if not latest.empty else None

    return {
        "city_id": city_id,
        "current_aqi": now_aqi,
        "observed_at": now_time,
        "forecast": horizons,
        "drivers": drivers,
    }


@app.get("/alerts", summary="Cities forecast to reach unhealthy air")
def alerts() -> list[dict]:
    alerts_df = read_fg(ALERTS_FG, ALERTS_VERSION)
    if alerts_df.empty:
        return []
    res = alerts_df.sort_values("peak_aqi", ascending=False)
    return res.to_dict(orient="records")


@app.get("/models", summary="Model accuracy, and which model is serving")
def models() -> dict:
    benchmark = get_best_models()
    return {
        "benchmark": benchmark.to_dict(orient="records") if not benchmark.empty else [],
        "note": (
            "Models are benchmarked from statistical to deep learning and registered in Hopsworks."
        ),
    }
