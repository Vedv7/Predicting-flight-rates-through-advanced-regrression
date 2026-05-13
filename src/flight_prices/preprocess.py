from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

from flight_prices.config import DEFAULT_Z_THRESHOLD

# --- Canonicalization (matches original notebook intent) ---

_AIRLINE_MERGE = {
    "Multiple carriers Premium economy": "Multiple carriers",
    "Jet Airways Business": "Jet Airways",
    "Vistara Premium economy": "Vistara",
}

_CITY_FIX = {"Delhi": "New Delhi", "Banglore": "Bangalore"}

_STOPS_MERGE = {"3 stops": "2+-stop", "4 stops": "2+-stop"}


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean_common(df: pd.DataFrame, *, require_price: bool = True) -> pd.DataFrame:
    """Row-level cleaning and feature construction up to (but not including) encoders."""
    out = df.copy()
    if require_price:
        if "Price" not in out.columns:
            raise ValueError("Expected a Price column for training data.")
    else:
        out = out.drop(columns=["Price"], errors="ignore")

    out = out.drop(labels=["Route", "Additional_Info"], axis=1, errors="ignore")
    out = out.dropna(subset=["Total_Stops"])
    out = out.drop_duplicates()

    out["Total_Stops"] = out["Total_Stops"].replace(_STOPS_MERGE)
    out["Airline"] = out["Airline"].replace(_AIRLINE_MERGE)
    out["Source"] = out["Source"].replace(_CITY_FIX)
    out["Destination"] = out["Destination"].replace(_CITY_FIX)

    out["Date_of_Journey"] = pd.to_datetime(out["Date_of_Journey"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["Date_of_Journey"])
    out["Day"] = out["Date_of_Journey"].dt.strftime("%A")
    out["Date"] = out["Date_of_Journey"].dt.day.astype(np.int32)
    out["Month"] = out["Date_of_Journey"].dt.strftime("%B")
    out["Year"] = out["Date_of_Journey"].dt.year.astype(np.int32)

    dur = out["Duration"].astype(str)
    out = out.loc[dur.str.match(r"\d+h \d+m", na=False)].copy()
    extracted = out["Duration"].str.extract(r"(\d+)h (\d+)m")
    out["hours"] = extracted[0].astype(int)
    out["minutes"] = extracted[1].astype(int)
    out["Duration(mins)"] = out["hours"] * 60 + out["minutes"]
    out = out.drop(["Duration", "hours", "minutes"], axis=1)

    out["Arrival_Time"] = pd.to_datetime(out["Arrival_Time"], errors="coerce", format="mixed")
    out = out.dropna(subset=["Arrival_Time"])
    out["Arrival Date"] = out["Arrival_Time"].dt.date
    out["Arrival Time"] = out["Arrival_Time"].dt.time

    out = out.drop(
        columns=["Date_of_Journey", "Year", "Arrival_Time", "Dep_Time", "Arrival Date", "Arrival Time"],
        errors="ignore",
    )

    if require_price:
        out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
        out = out.dropna(subset=["Price"])

    out["Duration(mins)"] = pd.to_numeric(out["Duration(mins)"], errors="coerce")
    out = out.dropna(subset=["Duration(mins)"])
    return out.reset_index(drop=True)


def align_raw_training_frame(X_raw: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Apply `clean_common` so feature rows and targets stay aligned (same length)."""
    merged = X_raw.copy()
    merged["Price"] = np.asarray(y)
    tab = clean_common(merged, require_price=True)
    y_out = tab["Price"].astype(float)
    X_out = tab.drop(columns=["Price"])
    return X_out.reset_index(drop=True), y_out.reset_index(drop=True)


def align_raw_inference_frame(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw booking rows for prediction (no Price column)."""
    return clean_common(X_raw, require_price=False).reset_index(drop=True)


@dataclass
class PreprocessArtifacts:
    duration_mu: float
    duration_sigma: float
    z_threshold: float


class FlightPreprocessor(BaseEstimator, TransformerMixin):
    """Encoder + scaling prep on **already-cleaned** feature tables (see `align_raw_*`)."""

    def __init__(self, z_threshold: float = DEFAULT_Z_THRESHOLD) -> None:
        self.z_threshold = z_threshold

    def get_feature_names_out(self, input_features: list | None = None) -> np.ndarray:  # noqa: ARG002
        if not hasattr(self, "column_transformer_"):
            raise RuntimeError("Call fit before get_feature_names_out.")
        names = self.column_transformer_.get_feature_names_out()
        names = [n for n in names if n != "Airline_Trujet"]
        return np.asarray(names)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FlightPreprocessor:  # noqa: ARG002
        feats = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        d = feats["Duration(mins)"].astype(float)
        mu, sig = float(d.mean()), float(d.std(ddof=0))
        sig = sig if sig > 1e-9 else 1.0
        mask = (d - mu).abs() / sig < self.z_threshold
        self.duration_mu_ = mu
        self.duration_sigma_ = sig
        feats_fit = feats.loc[mask].copy()

        self.column_transformer_ = ColumnTransformer(
            transformers=[
                ("duration", PowerTransformer(method="yeo-johnson", standardize=True), ["Duration(mins)"]),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["Airline", "Total_Stops", "Day", "Month"],
                ),
                (
                    "geo",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    ["Source", "Destination"],
                ),
                ("date", "passthrough", ["Date"]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.column_transformer_.fit(feats_fit)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        feats = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Xt = self.column_transformer_.transform(feats)
        names = list(self.column_transformer_.get_feature_names_out())
        frame = pd.DataFrame(Xt, columns=names, index=feats.index)
        if "Airline_Trujet" in frame.columns:
            frame = frame.drop(columns=["Airline_Trujet"])
        return frame.astype(float).values

    def describe(self) -> PreprocessArtifacts:
        return PreprocessArtifacts(
            duration_mu=self.duration_mu_,
            duration_sigma=self.duration_sigma_,
            z_threshold=self.z_threshold,
        )


def build_model_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", FlightPreprocessor()),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def duration_z_mask(series: pd.Series, *, threshold: float = DEFAULT_Z_THRESHOLD) -> pd.Series:
    z = zscore(series.astype(float), nan_policy="omit")
    z = np.nan_to_num(z, nan=0.0)
    return pd.Series(np.abs(z) < threshold, index=series.index)
