"""OpenWeather client.

OpenWeather gives us the *predictive features*: current weather (and, in later
phases, the multi-day forecast we feed the model to predict *future* AQI).
Weather is the main physical driver of air quality — wind disperses pollution,
rain washes it out, temperature inversions trap it — so these are core inputs.

Free tier; get a key at https://openweathermap.org/api.
Docs: https://openweathermap.org/current
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

BASE_URL = "https://api.openweathermap.org/data/2.5"
TIMEOUT_SECONDS = 15


@dataclass
class WeatherReading:
    temp_c: float | None
    humidity: int | None      # %
    pressure: int | None      # hPa
    wind_speed: float | None  # m/s
    wind_deg: int | None      # degrees
    clouds: int | None        # % cloud cover
    description: str | None    # e.g. "haze"


class OpenWeatherClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def get_current_weather(self, lat: float, lon: float) -> WeatherReading:
        url = f"{BASE_URL}/weather"
        response = requests.get(
            url,
            params={
                "lat": lat,
                "lon": lon,
                "appid": self._api_key,
                "units": "metric",  # temps in Celsius, wind in m/s
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = (data.get("weather") or [{}])[0]

        return WeatherReading(
            temp_c=main.get("temp"),
            humidity=main.get("humidity"),
            pressure=main.get("pressure"),
            wind_speed=wind.get("speed"),
            wind_deg=wind.get("deg"),
            clouds=(data.get("clouds") or {}).get("all"),
            description=weather.get("description"),
        )
