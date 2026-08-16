"""Sequence builder for the deep-learning (LSTM) model.

Classical models see one flat row. An LSTM instead reads a *sequence* — the last
N hours of conditions — and learns the temporal shape of how air quality evolves.

For every city and every hour T (with enough history and future), we build:
  - input  X: the past `window` hours of features  -> shape (window, n_features)
  - target y: the actual AQI at T+24h, T+48h, T+72h -> shape (3,)

City identity is appended as a one-hot to every timestep, so one model still
serves all cities. Sequences are built per city so history never crosses cities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.training.dataset import HORIZONS

# Per-hour features fed to the LSTM (raw signals + cyclical time; lag/rolling are
# unnecessary here because the sequence itself carries the recent past).
SEQ_FEATURES = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temp_c", "humidity", "wind_speed", "wind_deg", "pressure", "precip", "clouds",
    "hour_sin", "hour_cos",
]

WINDOW = 24   # hours of history per sequence (one full day)
STRIDE = 3    # take every 3rd window — AQI is autocorrelated, so we lose little
              # and cut memory ~3x (keeps CPU training within a laptop's RAM)


def build_sequences(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    window: int = WINDOW,
    stride: int = STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, end_times).

    X:         (n, window, n_features) float32
    y:         (n, len(horizons))      float32  — future AQI at each horizon
    end_times: (n,)                    datetime — timestamp of each window's last
               hour, used for the time-based train/test split.
    """
    cities = sorted(data["city_id"].unique())
    max_h = max(horizons)

    X_list: list[np.ndarray] = []
    y_list: list[list[float]] = []
    end_times: list = []

    for city_id, group in data.groupby("city_id"):
        g = group.sort_values("event_time").reset_index(drop=True)

        feats = np.nan_to_num(g[SEQ_FEATURES].to_numpy(dtype="float32"))
        onehot = np.zeros((len(g), len(cities)), dtype="float32")
        onehot[:, cities.index(city_id)] = 1.0
        feats = np.concatenate([feats, onehot], axis=1)

        aqi = g["aqi"].to_numpy(dtype="float32")
        times = g["event_time"].to_numpy()

        for i in range(window - 1, len(g) - max_h, stride):
            X_list.append(feats[i - window + 1 : i + 1])
            y_list.append([aqi[i + h] for h in horizons])
            end_times.append(times[i])

    return (
        np.asarray(X_list, dtype="float32"),
        np.asarray(y_list, dtype="float32"),
        np.asarray(end_times),
    )
