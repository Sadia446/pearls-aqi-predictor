"""Model definitions + evaluation metrics.

We benchmark from simple to complex, exactly as the brief asks ("statistical to
deep learning"):

  - persistence : the naive baseline — "AQI in 3 days == AQI now". Every real
                  model must beat this to justify its existence.
  - ridge       : regularised linear regression. Fast, interpretable.
  - random_forest: nonlinear, robust, handles feature interactions.

(XGBoost and an LSTM come in later phases.)

Each model is a scikit-learn Pipeline that imputes missing values first, so a
stray NaN (e.g. an early-history lag) never crashes training.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_models() -> dict[str, Pipeline]:
    """The trainable models (persistence is handled separately, it needs no fit)."""
    return {
        "ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        # Depth and leaf-size caps matter at our scale: unbounded
                        # trees on 270k rows produce a model too large to store
                        # (and to load in a serverless job), while these limits
                        # keep it small AND reduce overfitting.
                        max_depth=20,
                        min_samples_leaf=20,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate(y_true, y_pred) -> dict[str, float]:
    """The three metrics the brief requires: RMSE, MAE, R²."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
