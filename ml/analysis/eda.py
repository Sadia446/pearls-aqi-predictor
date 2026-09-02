"""Exploratory Data Analysis — what does 2.5 years of air-quality data tell us?

Answers the questions that shape the product and the model:
  1. How bad is the air, per city? (distribution across health categories)
  2. Is there a seasonal pattern? (the winter smog season)
  3. Is there a daily rhythm? (rush hours, night-time inversions)
  4. Which weather conditions go with bad air? (correlations)
  5. How persistent is AQI? (autocorrelation — why lag features work)

Writes charts to docs/eda/ and prints the findings.

    python -m ml.analysis.eda
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ml.common.aqi import aqi_category  # noqa: E402
from ml.common.cities import CITIES, get_city  # noqa: E402
from ml.common.config import PROJECT_ROOT  # noqa: E402
from ml.storage.feature_store import read_features  # noqa: E402

OUT_DIR = PROJECT_ROOT / "docs" / "eda"
CITY_COLORS = {"lahore": "#cc0033", "islamabad": "#ff9933", "karachi": "#009966"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved docs/eda/{name}")


def run_eda() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = read_features()
    # Local time matters for daily-rhythm analysis (all our cities share a tz).
    df["local"] = df["event_time"].dt.tz_convert(get_city("lahore").timezone)
    df["month"] = df["local"].dt.month
    df["hour_local"] = df["local"].dt.hour

    print(f"\n  EDA over {len(df):,} hourly rows, "
          f"{df['event_time'].min().date()} -> {df['event_time'].max().date()}\n")

    # 1. Per-city summary + health categories -------------------------------
    print("  1. How bad is the air, per city?")
    summary = df.groupby("city_id")["aqi"].agg(["mean", "median", "max"]).round(1)
    for city_id, row in summary.iterrows():
        unhealthy = (df.loc[df.city_id == city_id, "aqi"] > 100).mean() * 100
        print(f"    {city_id:<11} mean {row['mean']:6.1f}  median {row['median']:6.1f}  "
              f"max {row['max']:6.0f}   unhealthy-for-sensitive: {unhealthy:4.1f}% of hours")

    df["category"] = df["aqi"].apply(aqi_category)
    order = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
             "Unhealthy", "Very Unhealthy", "Hazardous"]
    share = (
        df.groupby("city_id")["category"].value_counts(normalize=True)
        .unstack(fill_value=0).reindex(columns=order, fill_value=0) * 100
    )
    fig, ax = plt.subplots(figsize=(9, 3.6))
    share.plot(kind="barh", stacked=True, ax=ax,
               color=["#009966", "#e6b800", "#ff9933", "#cc0033", "#660099", "#7e0023"])
    ax.set_xlabel("% of hours"); ax.set_ylabel("")
    ax.set_title("Air quality distribution by city (2024–2026)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, "01_category_distribution.png")

    # 2. Seasonality --------------------------------------------------------
    print("\n  2. Seasonal pattern (monthly mean AQI):")
    monthly = df.groupby(["city_id", "month"])["aqi"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4))
    for city_id in monthly.index:
        ax.plot(monthly.columns, monthly.loc[city_id], marker="o",
                label=get_city(city_id).name, color=CITY_COLORS.get(city_id))
    ax.axhline(100, ls="--", c="grey", lw=1)
    ax.text(0.3, 103, "unhealthy for sensitive groups", fontsize=7, color="grey")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS)
    ax.set_ylabel("mean AQI"); ax.set_title("Seasonality: AQI by month")
    ax.legend()
    _save(fig, "02_seasonality.png")
    for city_id in monthly.index:
        worst, best = monthly.loc[city_id].idxmax(), monthly.loc[city_id].idxmin()
        print(f"    {city_id:<11} worst {MONTHS[worst-1]} ({monthly.loc[city_id, worst]:.0f})   "
              f"best {MONTHS[best-1]} ({monthly.loc[city_id, best]:.0f})")

    # 3. Daily rhythm -------------------------------------------------------
    print("\n  3. Daily rhythm (mean AQI by local hour):")
    hourly = df.groupby(["city_id", "hour_local"])["aqi"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4))
    for city_id in hourly.index:
        ax.plot(hourly.columns, hourly.loc[city_id], marker=".",
                label=get_city(city_id).name, color=CITY_COLORS.get(city_id))
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("hour (local)")
    ax.set_ylabel("mean AQI"); ax.set_title("Daily rhythm of air quality")
    ax.legend()
    _save(fig, "03_daily_rhythm.png")
    for city_id in hourly.index:
        peak = hourly.loc[city_id].idxmax()
        print(f"    {city_id:<11} peaks around {peak:02d}:00 local "
              f"({hourly.loc[city_id, peak]:.0f} AQI)")

    # 4. Weather correlations ----------------------------------------------
    print("\n  4. What weather goes with bad air? (correlation with AQI)")
    weather = ["temp_c", "humidity", "wind_speed", "pressure", "precip", "clouds"]
    corr = df[weather + ["aqi"]].corr()["aqi"].drop("aqi").sort_values()
    for name, value in corr.items():
        direction = "cleaner air" if value < 0 else "dirtier air"
        print(f"    {name:<12} {value:+.3f}   (higher {name} -> {direction})")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = ["#009966" if v < 0 else "#cc0033" for v in corr]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, c="black", lw=0.8)
    ax.set_title("Correlation between weather and AQI")
    _save(fig, "04_weather_correlation.png")

    # 5. Persistence / autocorrelation -------------------------------------
    print("\n  5. How persistent is AQI? (autocorrelation — why lags work)")
    lags = [1, 3, 6, 12, 24, 48, 72]
    rows = {}
    for city_id, g in df.groupby("city_id"):
        s = g.sort_values("event_time")["aqi"].reset_index(drop=True)
        rows[city_id] = [s.autocorr(lag=l) for l in lags]
        print(f"    {city_id:<11} " + "  ".join(
            f"{l}h={v:.2f}" for l, v in zip(lags, rows[city_id])))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for city_id, values in rows.items():
        ax.plot(lags, values, marker="o", label=get_city(city_id).name,
                color=CITY_COLORS.get(city_id))
    ax.set_xlabel("lag (hours)"); ax.set_ylabel("correlation with itself")
    ax.set_title("AQI persistence: today's air predicts tomorrow's")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "05_autocorrelation.png")

    print(f"\n  EDA complete — {len(list(OUT_DIR.glob('*.png')))} charts in docs/eda/\n")


if __name__ == "__main__":
    run_eda()
