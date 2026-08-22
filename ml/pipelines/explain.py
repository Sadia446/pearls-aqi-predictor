"""SHAP explainability — why does the model forecast what it forecasts?

A forecast people are meant to *act* on has to be defensible. SHAP assigns each
feature a contribution (in AQI points) to a specific prediction, grounded in
game theory, so we can say "this forecast is high because PM2.5 is elevated and
wind is weak" rather than "the model said so".

Two outputs:
  1. A global summary plot (docs/shap_summary.png) — what drives forecasts overall.
  2. Per-city drivers for the *current* forecast, written to Hopsworks
     `forecast_drivers` feature group, so the web dashboard can show a plain-language "why".

    python -m ml.pipelines.explain
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from ml.common.config import PROJECT_ROOT  # noqa: E402
from ml.storage.feature_store import get_latest_features, read_features  # noqa: E402
from ml.storage.hopsworks_store import write_drivers  # noqa: E402
from ml.storage.registry import load_active_model  # noqa: E402
from ml.training.dataset import add_targets, build_design_matrix, feature_columns  # noqa: E402

HORIZON = 24  # explain the next-day forecast (the one users act on most)
SAMPLE_ROWS = 2000  # background sample — SHAP is expensive on the full table
EXPLAIN_STRIDE = 24  # read one row per day; SHAP needs a spread, not every hour
TOP_DRIVERS = 5
PLOT_PATH = PROJECT_ROOT / "docs" / "shap_summary.png"

# Plain-language names so the UI can show human text, not column names.
FRIENDLY = {
    "pm25": "PM2.5", "pm10": "PM10", "aqi": "AQI",
    "o3": "ozone", "no2": "nitrogen dioxide", "so2": "sulphur dioxide", "co": "carbon monoxide",
    "temp_c": "temperature", "humidity": "humidity", "pressure": "air pressure",
    "wind_speed": "wind speed", "precip": "rainfall", "clouds": "cloud cover",
    "is_calm_wind": "calm wind", "pm_ratio": "fine/coarse particle mix",
    "aqi_change_1h": "AQI trend (1h)", "aqi_change_24h": "AQI trend (24h)",
}

# suffix -> how to phrase it, given the base name and window
_DERIVED = {
    "_lag_": lambda base, w: f"{base} {w} ago",
    "_roll_mean_": lambda base, w: f"average {base} over {w}",
    "_roll_max_": lambda base, w: f"peak {base} in {w}",
    "_roll_std_": lambda base, w: f"{base} volatility over {w}",
}


def _friendly(name: str) -> str:
    if name in FRIENDLY:
        return f"current {FRIENDLY[name]}" if name in ("aqi", "pm25", "pm10") else FRIENDLY[name]
    if name.startswith("city_"):
        return f"city ({name[5:]})"
    for key, phrase in _DERIVED.items():
        if key in name:
            base, _, window = name.partition(key)
            return phrase(FRIENDLY.get(base, base), window)
    return name.replace("_", " ")


def run_explain() -> pd.DataFrame:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    model, model_name = load_active_model(HORIZON)
    print(f"\n  Explaining the +{HORIZON}h model ({model_name}) with SHAP ...")

    # Historical sample -> global explanation.
    data = add_targets(read_features(stride=EXPLAIN_STRIDE))
    X = build_design_matrix(data, feature_columns(data))
    names = list(getattr(model, "feature_names_in_", X.columns))
    X = X.reindex(columns=names, fill_value=0)
    sample = X.sample(n=min(SAMPLE_ROWS, len(X)), random_state=42)

    # Explain the fitted estimator on transformed inputs, so SHAP sees exactly
    # what the final regressor sees (imputed/scaled), not the raw frame.
    pre = model[:-1]
    estimator = model[-1]
    sample_t = pd.DataFrame(pre.transform(sample), columns=names)
    explainer = shap.Explainer(estimator, sample_t)
    shap_values = explainer(sample_t)

    PLOT_PATH.parent.mkdir(exist_ok=True)
    shap.summary_plot(shap_values, sample_t, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved global summary plot -> {PLOT_PATH.relative_to(PROJECT_ROOT)}")

    # Per-city drivers for the *current* forecast.
    latest = get_latest_features()
    X_now = build_design_matrix(latest, feature_columns(latest)).reindex(
        columns=names, fill_value=0
    )
    now_t = pd.DataFrame(pre.transform(X_now), columns=names)
    now_shap = explainer(now_t)

    records: list[dict] = []
    print("\n  Why the next-day forecast looks like it does:")
    for i, city_id in enumerate(latest["city_id"]):
        contribs = pd.Series(now_shap.values[i], index=names)
        top = contribs.reindex(contribs.abs().sort_values(ascending=False).index)[:TOP_DRIVERS]
        print(f"\n    {city_id}:")
        for feature, value in top.items():
            direction = "raises" if value > 0 else "lowers"
            print(f"      {_friendly(feature):<28} {direction:>6} AQI by {abs(value):5.1f}")
            records.append(
                {
                    "city_id": city_id,
                    "horizon_h": HORIZON,
                    "feature": feature,
                    "label": _friendly(feature),
                    "contribution": round(float(value), 2),
                    "direction": "up" if value > 0 else "down",
                }
            )

    drivers = pd.DataFrame(records)
    write_drivers(drivers)
    print(f"\n  Wrote {len(drivers)} drivers to Hopsworks feature store.\n")
    return drivers


if __name__ == "__main__":
    run_explain()
