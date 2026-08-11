"""Model registry — where trained models and their scorecards live.

Two parts:
  - the *artifact* (the fitted model) is saved under models/ and registered in
    Hopsworks Model Registry.
  - the *metadata* (which model, which horizon, its RMSE/MAE/R², whether it's
    the current best) is stored in Hopsworks with local fallback.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from ml.common.config import PROJECT_ROOT
from ml.storage.hopsworks_store import get_model_registry

MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_FILE = MODELS_DIR / "registry.json"


def save_artifact(model, name: str) -> Path:
    """Persist a fitted model to models/<name>.pkl and return the path."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    return path


def register_run(records: list[dict], *_args, **_kwargs) -> int:
    """Register one training run's metrics in Hopsworks Model Registry & local cache."""
    MODELS_DIR.mkdir(exist_ok=True)

    # Save locally
    existing_records = []
    if REGISTRY_FILE.exists():
        try:
            existing_records = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing_records = []
    existing_records.extend(records)
    REGISTRY_FILE.write_text(json.dumps(existing_records, indent=2, default=str), encoding="utf-8")

    mr = get_model_registry()
    if mr:
        for rec in records:
            model_name = f"aqi_{rec['horizon_h']}h_{rec['model_name']}"
            metrics = {"rmse": rec["rmse"], "mae": rec["mae"], "r2": rec["r2"]}

            with tempfile.TemporaryDirectory() as tmpdir:
                meta_path = os.path.join(tmpdir, "metadata.json")
                with open(meta_path, "w") as f:
                    json.dump(
                        {
                            "run_id": str(rec.get("run_id", "")),
                            "trained_at": str(rec.get("trained_at", "")),
                            "horizon_h": rec["horizon_h"],
                            "model_name": rec["model_name"],
                            "is_best": rec.get("is_best", False),
                        },
                        f,
                    )

                artifact_path = rec.get("artifact_path")
                input_dir = tmpdir
                if artifact_path and Path(artifact_path).exists():
                    input_dir = str(Path(artifact_path).parent)

                try:
                    model = mr.python.create_model(
                        name=model_name,
                        metrics=metrics,
                        description=f"AQI +{rec['horizon_h']}h forecast ({rec['model_name']})",
                    )
                    model.save(input_dir)
                except Exception as exc:
                    print(f"    [Hopsworks] Notice for {model_name}: {exc}")

    return len(records)


def save_active_model(
    model, horizon_h: int, model_name: str, metrics: dict, *_args, **_kwargs
) -> None:
    """Store the current best model in Hopsworks Model Registry and local models/."""
    MODELS_DIR.mkdir(exist_ok=True)
    local_path = MODELS_DIR / f"aqi_{horizon_h}h_active.pkl"
    joblib.dump(model, local_path)

    mr = get_model_registry()
    if mr:
        reg_name = f"aqi_{horizon_h}h_active"
        try:
            hw_model = mr.python.create_model(
                name=reg_name,
                metrics={"rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"]},
                description=f"Active serving model for +{horizon_h}h (algorithm: {model_name})",
            )
            hw_model.save(str(MODELS_DIR))
        except Exception as exc:
            print(f"    [Hopsworks] Notice saving active model: {exc}")


def load_active_model(horizon_h: int, *_args, **_kwargs) -> tuple[object, str]:
    """Load the active model for a horizon."""
    mr = get_model_registry()
    if mr:
        reg_name = f"aqi_{horizon_h}h_active"
        try:
            hw_model = mr.get_model(reg_name)
            model_dir = hw_model.download()
            local_path = Path(model_dir) / f"aqi_{horizon_h}h_active.pkl"
            model = joblib.load(local_path)
            model_name = hw_model.description.split("algorithm: ")[-1].rstrip(")")
            return model, model_name
        except Exception:
            pass

    # Fall back to local models/ directory.
    local_path = MODELS_DIR / f"aqi_{horizon_h}h_active.pkl"
    if local_path.exists():
        model = joblib.load(local_path)
        return model, "active_model"
    raise RuntimeError(
        f"No active model for +{horizon_h}h — run the training pipeline first."
    )


def get_best_models() -> pd.DataFrame:
    """The current best model per horizon from Hopsworks or local registry."""
    mr = get_model_registry()
    rows = []
    if mr:
        for h in (24, 48, 72):
            reg_name = f"aqi_{h}h_active"
            try:
                hw_model = mr.get_model(reg_name)
                rows.append(
                    {
                        "horizon_h": h,
                        "model_name": hw_model.description.split("algorithm: ")[-1].rstrip(")"),
                        "rmse": hw_model.training_metrics.get("rmse"),
                        "mae": hw_model.training_metrics.get("mae"),
                        "r2": hw_model.training_metrics.get("r2"),
                    }
                )
            except Exception:
                continue

    if not rows and REGISTRY_FILE.exists():
        try:
            records = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            df = pd.DataFrame(records)
            if not df.empty and "is_best" in df.columns:
                best = df[df["is_best"] == True].drop_duplicates(subset=["horizon_h"], keep="last")  # noqa: E712
                return best[["horizon_h", "model_name", "rmse", "mae", "r2"]]
        except Exception:
            pass

    return pd.DataFrame(rows)
