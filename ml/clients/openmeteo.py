"""Open-Meteo client — our historical backfill source.

The live APIs (AQICN, OpenWeather) only give us *current* readings. To train a
model we need *history*. Open-Meteo provides free historical air-quality and
weather data with no API key and generous limits, going back to 2022 — perfect
for building a training dataset.

We pull from two Open-Meteo endpoints and merge them on the hourly timestamp:
  - Air-Quality API : AQI + pollutant concentrations  (our target + features)
  - Archive API     : historical weather              (our predictive features)

All column names are normalised to the same vocabulary we use everywhere else
(pm25, o3, temp_c, humidity, ...) so the rest of the pipeline doesn't care which
source the data came from.

Docs: https://open-meteo.com/en/docs/air-quality-api
      https://open-meteo.com/en/docs/historical-weather-api
"""
from __future__ import annotations

import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ml.common.cities import City

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Connectivity to Open-Meteo from CI runners is unreliable — we saw SSL
# handshakes and reads time out there while the same calls succeed locally.
# Split the timeout so a stalled *connection* fails fast (and is retried)
# instead of burning the full read budget on a handshake that will never finish.
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 45
TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 5
PAUSE_BETWEEN_CALLS = 1.0


def _make_session() -> requests.Session:
    """Session with connection-level retries and keep-alive.

    Reusing one connection across our 24 calls avoids repeating the TLS
    handshake that was timing out, and urllib3's Retry handles the low-level
    connect/read failures beneath `requests`.
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=2,  # 0s, 2s, 4s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=4))
    return session


_SESSION = _make_session()


def _get(url: str, params: dict) -> dict:
    """GET with retry/backoff on rate limits, timeouts and server errors."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = _SESSION.get(url, params=params, timeout=TIMEOUT)
            if response.status_code == 429 or response.status_code >= 500:
                # Honour Retry-After when the server sends one.
                wait = float(
                    response.headers.get("Retry-After", BACKOFF_SECONDS * attempt)
                )
                last_error = RuntimeError(
                    f"HTTP {response.status_code} from {url}: {response.text[:200]}"
                )
                if attempt < MAX_ATTEMPTS:
                    print(
                        f"    rate-limited ({response.status_code}), "
                        f"retrying in {wait:.0f}s [{attempt}/{MAX_ATTEMPTS}]"
                    )
                    time.sleep(wait)
                    continue
            response.raise_for_status()
            time.sleep(PAUSE_BETWEEN_CALLS)  # be a polite API citizen
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                wait = BACKOFF_SECONDS * attempt
                print(
                    f"    {type(exc).__name__} contacting Open-Meteo, "
                    f"retrying in {wait}s [{attempt}/{MAX_ATTEMPTS}]"
                )
                time.sleep(wait)

    raise RuntimeError(f"Open-Meteo request failed after {MAX_ATTEMPTS} attempts: {last_error}")

# Open-Meteo variable name  ->  our canonical name
_AIR_QUALITY_VARS = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "ozone": "o3",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
    "us_aqi": "aqi",  # US EPA AQI — same scale as AQICN, this is our target
}
_WEATHER_VARS = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_deg",
    "pressure_msl": "pressure",
    "precipitation": "precip",
    "cloud_cover": "clouds",
}


def _hourly_to_frame(payload: dict, rename: dict[str, str]) -> pd.DataFrame:
    """Turn an Open-Meteo `hourly` block into a tidy DataFrame with a UTC time column."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError(f"Open-Meteo returned no hourly data: {payload.get('reason', payload)}")

    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.rename(columns=rename)
    # Keep only the time column + the variables we asked for
    keep = ["time"] + [name for name in rename.values() if name in frame.columns]
    return frame[keep]


class OpenMeteoClient:
    def get_historical_air_quality(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Hourly AQI + pollutants between start_date and end_date (YYYY-MM-DD, inclusive)."""
        payload = _get(
            AIR_QUALITY_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(_AIR_QUALITY_VARS.keys()),
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "UTC",
            },
        )
        return _hourly_to_frame(payload, _AIR_QUALITY_VARS)

    def get_historical_weather(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Hourly weather between start_date and end_date (YYYY-MM-DD, inclusive)."""
        payload = _get(
            ARCHIVE_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(_WEATHER_VARS.keys()),
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "UTC",
            },
        )
        return _hourly_to_frame(payload, _WEATHER_VARS)

    def get_history(self, city: City, start_date: str, end_date: str) -> pd.DataFrame:
        """Combined hourly air-quality + weather for one city, tagged with city_id.

        Returns one row per hour with columns:
          city_id, time, aqi, pm25, pm10, o3, no2, so2, co,
          temp_c, humidity, wind_speed, wind_deg, pressure, precip, clouds
        """
        air = self.get_historical_air_quality(city.lat, city.lon, start_date, end_date)
        weather = self.get_historical_weather(city.lat, city.lon, start_date, end_date)

        merged = pd.merge(air, weather, on="time", how="inner")
        merged.insert(0, "city_id", city.id)
        return merged.sort_values("time").reset_index(drop=True)
