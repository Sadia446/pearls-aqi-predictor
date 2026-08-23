"""Historical backfill pipeline.

Runs the whole "raw history -> features -> feature store" path, building the
training data a model can actually learn from.

    python -m ml.pipelines.backfill                        # all cities, full rebuild
    python -m ml.pipelines.backfill --cities lahore        # just these, others untouched
    python -m ml.pipelines.backfill --start 2024-06-01     # custom window

Re-running is safe: Hopsworks upserts by primary key (city_id, event_time).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from ml.clients.openmeteo import OpenMeteoClient
from ml.common.cities import City, resolve
from ml.features.engineering import build_features
from ml.storage.hopsworks_store import table_summary, write_features

# Start of 2024 gives us 1.5+ years including a full winter smog season —
# the most important signal for cities like Lahore.
DEFAULT_START = "2024-01-01"


def run_backfill(
    cities: list[City] | None = None,
    start_date: str = DEFAULT_START,
    end_date: str | None = None,
) -> pd.DataFrame:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cities = cities or resolve(None)
    # Stop a few days short of today so we only request dates with settled data.
    if end_date is None:
        end_date = (date.today() - timedelta(days=2)).isoformat()

    client = OpenMeteoClient()
    frames: list[pd.DataFrame] = []

    print(f"\n  Backfill {start_date} -> {end_date} for {len(cities)} city/cities")
    print("  " + "-" * 60)
    for city in cities:
        raw = client.get_history(city, start_date, end_date)
        features = build_features(raw).rename(columns={"time": "event_time"})
        frames.append(features)
        print(
            f"  {city.name:<13} {len(features):>6} rows   "
            f"AQI {int(features['aqi'].min())}-{int(features['aqi'].max())}"
        )

    combined = pd.concat(frames, ignore_index=True)
    print("  " + "-" * 60)

    print(f"  Writing {len(combined)} rows to Hopsworks feature store ...")
    n = write_features(combined)

    summary = table_summary()
    if not summary.empty:
        print("\n  Feature store now holds:")
        print(summary.to_string(index=False))
    print()
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical AQI features.")
    parser.add_argument(
        "--cities", nargs="*", help="city ids to backfill (default: all)"
    )
    parser.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    run_backfill(
        cities=resolve(args.cities), start_date=args.start, end_date=args.end
    )


if __name__ == "__main__":
    main()
