"""Inference pipeline — the live 3-day forecast.

This is the third decoupled pipeline. It reads the *latest* features per city
(the online store), loads the current best model for each horizon from the
registry, predicts AQI at +24h/+48h/+72h, and writes the results to the
Hopsworks feature store. The Streamlit app and alerts read from there.

    python -m ml.pipelines.inference_pipeline
"""
from __future__ import annotations

import sys
from datetime import timedelta

import pandas as pd

from ml.common.aqi import aqi_category
from ml.common.cities import get_city
from ml.storage.feature_store import get_latest_features
from ml.storage.hopsworks_store import write_predictions
from ml.storage.registry import load_active_model
from ml.training.dataset import HORIZONS, build_design_matrix, feature_columns

PREDICTIONS_TABLE = "predictions"


def run_inference() -> pd.DataFrame:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ONLINE features: newest row per city.
    latest = get_latest_features()
    X = build_design_matrix(latest, feature_columns(latest))

    records: list[dict] = []
    for horizon in HORIZONS:
        model, model_name = load_active_model(horizon)

        # Align columns exactly to what the model saw at fit time (order matters,
        # and a missing city dummy becomes 0). This makes inference bullet-proof.
        names = getattr(model, "feature_names_in_", X.columns)
        predictions = model.predict(X.reindex(columns=names, fill_value=0))

        for i, city_id in enumerate(latest["city_id"]):
            base_time = latest["event_time"].iloc[i]
            aqi = round(float(predictions[i]), 1)
            records.append(
                {
                    "city_id": city_id,
                    "base_time": base_time,
                    "horizon_h": horizon,
                    "forecast_time": base_time + timedelta(hours=horizon),
                    "predicted_aqi": aqi,
                    "category": aqi_category(aqi),
                    "model_name": model_name,
                }
            )

    forecast = (
        pd.DataFrame(records)
        .sort_values(["city_id", "horizon_h"])
        .reset_index(drop=True)
    )

    # Write predictions to Hopsworks feature store.
    write_predictions(forecast)

    _print_forecast(forecast, latest)
    return forecast


def _print_forecast(forecast: pd.DataFrame, latest: pd.DataFrame) -> None:
    print("\n  AeroSense — 3-day AQI forecast")
    print("  " + "=" * 60)
    for city_id, group in forecast.groupby("city_id"):
        city = get_city(str(city_id))
        now_aqi = int(latest.loc[latest.city_id == city_id, "aqi"].iloc[0])
        print(f"\n  {city.name}   (now: AQI {now_aqi})")
        for r in group.itertuples():
            when = pd.Timestamp(r.forecast_time).tz_convert(city.timezone)
            print(
                f"    +{r.horizon_h:>2}h  {when:%a %d %b %H:%M}   "
                f"AQI {r.predicted_aqi:>5}   {r.category}"
            )
    print("\n  " + "=" * 60)
    print(f"  Wrote {len(forecast)} predictions to Hopsworks feature store.\n")


if __name__ == "__main__":
    run_inference()
