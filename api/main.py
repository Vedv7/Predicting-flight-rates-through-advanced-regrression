from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from flight_prices.preprocess import align_raw_inference_frame

app = FastAPI(title="Flight fare inference", version="0.1.0")


class FlightInput(BaseModel):
    Airline: str = Field(examples=["IndiGo"])
    Date_of_Journey: str = Field(description="DD/MM/YYYY", examples=["24/03/2019"])
    Source: str
    Destination: str
    Dep_Time: str = Field(examples=["22:20"])
    Arrival_Time: str = Field(examples=["01:10 22 Mar"])
    Duration: str = Field(examples=["2h 50m"])
    Total_Stops: str = Field(examples=["non-stop"])
    Route: str = Field(default="BLR → DEL")
    Additional_Info: str = Field(default="No info")


def _model_path() -> Path:
    raw = os.environ.get("FLIGHT_MODEL_PATH", "artifacts/model.joblib")
    return Path(raw).resolve()


@app.on_event("startup")
def _load_model() -> None:
    path = _model_path()
    if not path.is_file():
        app.state.model = None
        return
    app.state.model = joblib.load(path)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "model_path": str(_model_path()),
    }


@app.post("/predict")
def predict(row: FlightInput) -> dict:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded; train first or set FLIGHT_MODEL_PATH.")
    raw = pd.DataFrame([row.model_dump()])
    feats = align_raw_inference_frame(raw)
    if feats.empty:
        raise HTTPException(
            status_code=422,
            detail="Row failed cleaning (duration format, missing stops, or invalid dates).",
        )
    pred = float(model.predict(feats)[0])
    return {"predicted_price": pred, "currency": "INR"}
