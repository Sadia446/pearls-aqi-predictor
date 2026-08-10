"""Phase 0 smoke test — the project's first heartbeat.

For every city we plan to forecast, this fetches the live AQI (AQICN) and the
current weather (OpenWeather) and prints them. If real numbers show up, our two
data sources and API keys are wired correctly and we're ready to build the
feature pipeline (Phase 1).

Run from the project root:

    python -m ml.smoke
"""
from __future__ import annotations

import sys

from ml.clients.aqicn import AqicnClient
from ml.clients.openweather import OpenWeatherClient
from ml.common.cities import CITIES
from ml.common.config import get_settings


def main() -> None:
    # Windows terminals often default to cp1252, which can't render °/— etc.
    # Force UTF-8 so our output looks right everywhere.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    aqicn = AqicnClient(settings.aqicn_token)
    weather = OpenWeatherClient(settings.openweather_api_key)

    print("\n  AeroSense — live data heartbeat  (Phase 0 smoke test)")
    print("  " + "-" * 72)

    failures = 0
    for city in CITIES:
        try:
            air = aqicn.get_current(city.lat, city.lon)
            wx = weather.get_current_weather(city.lat, city.lon)
        except Exception as exc:  # noqa: BLE001 — surface any per-city failure
            print(f"  {city.name:<11}  ERROR: {exc}")
            failures += 1
            continue

        pm25 = air.pollutants.get("pm25", "-")
        print(
            f"  {city.name:<11}  "
            f"AQI {str(air.aqi):>3}  "
            f"PM2.5 {str(pm25):>4}  "
            f"| {wx.temp_c}°C  {wx.humidity}% RH  wind {wx.wind_speed} m/s  "
            f"| {wx.description}"
        )

    print("  " + "-" * 72)
    if failures == 0:
        print("  All cities returned live data. Phase 0 is GREEN.\n")
    else:
        print(f"  {failures} city/cities failed. Check the errors above.\n")


if __name__ == "__main__":
    main()
