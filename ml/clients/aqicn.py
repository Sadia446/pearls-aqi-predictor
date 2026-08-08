"""AQICN (World Air Quality Index) client.

AQICN gives us the *target* we want to predict: the real, observed AQI for a
location plus the individual pollutant sub-indices (PM2.5, PM10, O3, ...).
Free API; get a token at https://aqicn.org/data-platform/token/.
API docs: https://aqicn.org/json-api/doc/
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

BASE_URL = "https://api.waqi.info"
TIMEOUT_SECONDS = 15

# Pollutant sub-indices we care about, as AQICN names them in the `iaqi` block.
_POLLUTANT_KEYS = ("pm25", "pm10", "o3", "no2", "so2", "co")


@dataclass
class AqicnReading:
    aqi: int | None                  # headline AQI value
    dominant_pollutant: str | None   # e.g. "pm25"
    pollutants: dict[str, float]     # sub-indices present in this reading
    observed_at: str | None          # ISO timestamp reported by the station
    station: str | None              # nearest station name


class AqicnClient:
    def __init__(self, token: str) -> None:
        self._token = token

    def get_current(self, lat: float, lon: float) -> AqicnReading:
        """Latest reading from the station nearest to (lat, lon)."""
        url = f"{BASE_URL}/feed/geo:{lat};{lon}/"
        response = requests.get(
            url, params={"token": self._token}, timeout=TIMEOUT_SECONDS
        )
        response.raise_for_status()
        payload = response.json()

        # AQICN always returns HTTP 200; success/failure is in the `status` field.
        if payload.get("status") != "ok":
            raise RuntimeError(f"AQICN error: {payload.get('data', 'unknown error')}")

        data: dict[str, Any] = payload["data"]
        iaqi = data.get("iaqi", {})  # each entry looks like {"v": 42.0}

        pollutants = {
            key: iaqi[key]["v"] for key in _POLLUTANT_KEYS if key in iaqi
        }

        return AqicnReading(
            aqi=data.get("aqi"),
            # Note: the API misspells this field as "dominentpol".
            dominant_pollutant=data.get("dominentpol"),
            pollutants=pollutants,
            observed_at=(data.get("time") or {}).get("iso"),
            station=(data.get("city") or {}).get("name"),
        )
