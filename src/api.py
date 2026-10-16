"""Small FastAPI inference service for the trained demo model."""
import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", "artifacts/feature_names.json"))

app = FastAPI(title="MLOps Demo Prediction API", version="1.0.0")
_model = None
_features: list[str] = []


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(
        ..., description="Mapping of model feature names to numeric values"
    )


@app.on_event("startup")
def load_model() -> None:
    global _model, _features
    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        _features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded; train first.")
    supplied = request.features
    missing = sorted(set(_features) - set(supplied))
    extra = sorted(set(supplied) - set(_features))
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "extra_features": extra},
        )
    try:
        row = pd.DataFrame([[float(supplied[name]) for name in _features]], columns=_features)
        prediction = int(_model.predict(row)[0])
        result: dict[str, Any] = {"prediction": prediction}
        if hasattr(_model, "predict_proba"):
            result["probabilities"] = [
                float(value) for value in _model.predict_proba(row)[0]
            ]
        return result
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid numeric feature values.") from exc
