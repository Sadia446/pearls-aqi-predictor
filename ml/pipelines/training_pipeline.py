"""Training pipeline — reads features, trains & benchmarks models, registers the best.

    python -m ml.pipelines.training_pipeline

For each forecast horizon (24h, 48h, 72h) it:
  1. builds the supervised dataset (features + future-AQI target),
  2. trains a persistence baseline, Ridge, and Random Forest,
  3. scores each on a time-held-out test set (RMSE / MAE / R²),
  4. saves the best model and records every score in the model registry.
"""
from __future__ import annotations

import gzip
import pickle
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

# A model within this margin of the best RMSE is considered equally good, so we
# serve the cheapest one. 2% of ~20 AQI points is ~0.4 points — far below the
# model's own error, and invisible to a user reading a health category.
SELECTION_TOLERANCE = 0.02

# How much history to train on.
TRAIN_WINDOW_DAYS = 90


def _artifact_size(model) -> int:
    """Stored size in bytes — what the hourly inference job has to download."""
    return len(gzip.compress(pickle.dumps(model), compresslevel=6))


from ml.storage.feature_store import read_features
from ml.storage.registry import register_run, save_active_model, save_artifact
from ml.training.dataset import (
    HORIZONS,
    add_targets,
    build_design_matrix,
    feature_columns,
    time_split_mask,
)
from ml.training.models import evaluate, make_models


def _print_importances(model, feature_names: list[str], top: int = 8) -> None:
    """Show what a Random Forest leaned on most (native feature importances)."""
    rf = model.named_steps.get("model")
    if not hasattr(rf, "feature_importances_"):
        return
    ranked = sorted(
        zip(feature_names, rf.feature_importances_), key=lambda t: t[1], reverse=True
    )
    print("\n  Top drivers (Random Forest, +24h):")
    for name, imp in ranked[:top]:
        bar = "#" * int(imp * 60)
        print(f"    {name:<22} {imp:6.3f}  {bar}")


def run_training(test_frac: float = 0.1) -> pd.DataFrame:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n  Loading features from the feature store ...")
    since = (datetime.now(timezone.utc) - timedelta(days=TRAIN_WINDOW_DAYS)).isoformat()
    features = read_features(start=since)
    data = add_targets(features)
    feat_cols = feature_columns(data)
    X_all = build_design_matrix(data, feat_cols)

    # Validate that we have enough data before proceeding
    if len(data) < 2:
        raise ValueError(
            f"Insufficient training data: {len(data)} rows found, "
            "minimum 2 required. Check feature store connectivity and data sources."
        )

    # The split must reserve a buffer at the end of the timeline equal to the
    # longest forecast horizon. Without this, test rows near "now" have no
    # future AQI reading yet to compute target_aqi_{h}h from — shift(-h) in
    # add_targets() pulls in NaN, and the longest horizon's test set can end
    # up empty even though shorter horizons look fine. Sharing one buffer
    # across all horizons keeps the comparison fair (same test window).
    max_horizon_h = max(HORIZONS)
    train_mask, test_mask = time_split_mask(
        data, test_frac, max_horizon_h=max_horizon_h
    )

    split_time = data.loc[test_mask, "event_time"].min()
    test_end_time = data.loc[test_mask, "event_time"].max()
    print(
        f"  {len(data)} rows, {X_all.shape[1]} model inputs. "
        f"Test set = {split_time} to {test_end_time} "
        f"(holding back the last {max_horizon_h}h so every horizon has a real target).\n"
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trained_at = datetime.now(timezone.utc)

    rows: list[dict] = []
    registry_records: list[dict] = []

    for h in HORIZONS:
        target = data[f"target_aqi_{h}h"]
        valid = target.notna()
        tr, te = train_mask & valid, test_mask & valid
        X_tr, y_tr = X_all[tr], target[tr]
        X_te, y_te = X_all[te], target[te]

        # Validate train/test splits have data
        if len(X_tr) == 0:
            raise ValueError(
                f"Empty training set for {h}h horizon. "
                f"Check that feature store has data and targets are not all NaN."
            )
        if len(X_te) == 0:
            raise ValueError(
                f"Empty test set for {h}h horizon ({test_mask.sum()} test rows before filtering). "
                f"Even after reserving a {max_horizon_h}h buffer at the end of the "
                f"timeline, no test rows have a valid target_aqi_{h}h. This means "
                f"there isn't enough history yet to fairly evaluate this horizon — "
                f"collect more days of data, increase TRAIN_WINDOW_DAYS, or "
                f"temporarily drop this horizon from HORIZONS."
            )

        # 1) Persistence baseline: "AQI in h hours == AQI now".
        scored = {"persistence": evaluate(y_te, data.loc[te, "aqi"])}

        # 2) Real models.
        fitted = {}
        for name, model in make_models().items():
            model.fit(X_tr, y_tr)
            scored[name] = evaluate(y_te, model.predict(X_te))
            fitted[name] = model

        # 3) Choose what to SERVE, not merely what scores highest.
        best_rmse = min(scored[n]["rmse"] for n in fitted)
        contenders = {
            n: m
            for n, m in fitted.items()
            if scored[n]["rmse"] <= best_rmse * (1 + SELECTION_TOLERANCE)
        }
        best_name = min(contenders, key=lambda n: _artifact_size(contenders[n]))
        if scored[best_name]["rmse"] > best_rmse:
            print(
                f"  +{h}h: serving {best_name} "
                f"(RMSE {scored[best_name]['rmse']:.2f} vs best {best_rmse:.2f}) "
                f"— within {SELECTION_TOLERANCE:.0%}, and far cheaper to serve."
            )
        artifact = save_artifact(fitted[best_name], f"aqi_{h}h_{best_name}")
        # Store the active model in Hopsworks so serverless inference can load it.
        save_active_model(fitted[best_name], h, best_name, scored[best_name])

        for name, m in scored.items():
            rows.append({"horizon_h": h, "model": name, **m})
            registry_records.append(
                {
                    "run_id": run_id,
                    "trained_at": trained_at,
                    "horizon_h": h,
                    "model_name": name,
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "r2": m["r2"],
                    "is_best": name == best_name,
                    "artifact_path": str(artifact) if name == best_name else None,
                }
            )

        if h == 24 and best_name == "random_forest":
            _print_importances(fitted[best_name], list(X_all.columns))

    # Comparison table.
    results = pd.DataFrame(rows)
    print("\n  Model comparison (lower RMSE/MAE better, higher R² better):")
    for h in HORIZONS:
        print(f"\n  --- +{h}h forecast ---")
        block = results[results.horizon_h == h].drop(columns="horizon_h")
        print(block.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    n = register_run(registry_records)
    print(f"\n  Registered {n} scorecards to Hopsworks Model Registry. Best models saved to models/.\n")
    return results


if __name__ == "__main__":
    run_training()
