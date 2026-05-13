from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from flight_prices.config import ARTIFACTS_DIR, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from flight_prices.preprocess import align_raw_training_frame, build_model_pipeline, load_raw


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _score_model(name: str, pipe: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    pred = pipe.predict(X_test)
    return {
        "model": name,
        "r2": float(r2_score(y_test, pred)),
        "rmse": _rmse(np.asarray(y_test), pred),
        "mse": float(mean_squared_error(y_test, pred)),
    }


def train(
    data_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Clean data, split, fit several pipelines, persist the best by test R²."""
    out_dir = Path(out_dir or ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(data_path)
    if "Price" not in raw.columns:
        raise ValueError("Dataset must include a Price column.")
    y = raw["Price"].astype(float)
    X_raw = raw.drop(columns=["Price"])
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state
    )

    X_train, y_train = align_raw_training_frame(X_train, y_train)
    X_test, y_test = align_raw_training_frame(X_test, y_test)

    candidates: dict[str, Any] = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=30,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    results: list[dict[str, Any]] = []
    best_name: str | None = None
    best_pipe: Any = None
    best_r2 = -np.inf

    for name, estimator in candidates.items():
        pipe = build_model_pipeline(estimator)
        pipe.fit(X_train, y_train)
        row = _score_model(name, pipe, X_test, y_test)
        results.append(row)
        if row["r2"] > best_r2:
            best_r2 = row["r2"]
            best_name = name
            best_pipe = pipe

    assert best_pipe is not None and best_name is not None

    feature_names = list(best_pipe.named_steps["prep"].get_feature_names_out())
    raw_feature_columns = list(X_raw.columns)

    bundle = {
        "best_model": best_name,
        "metrics": {"per_model": results, "holdout_size": float(test_size), "random_state": random_state},
        "feature_names_transformed": feature_names,
        "raw_feature_columns": raw_feature_columns,
    }

    (out_dir / "metrics.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    joblib.dump(best_pipe, out_dir / "model.joblib")

    return bundle
