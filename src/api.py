"""Small FastAPI inference service for the trained demo model."""
import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", "artifacts/feature_names.json"))
VERSION_MANIFEST_PATH = MODEL_PATH.parent / "model_version.json"

HTTP_REQUESTS = Counter(
    "mlops_http_requests_total",
    "Total HTTP requests handled by the inference API.",
    ("method", "route", "status_code"),
)
HTTP_REQUEST_DURATION = Histogram(
    "mlops_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "route"),
)
PREDICTIONS = Counter(
    "mlops_predictions_total",
    "Total successful model predictions.",
    ("prediction",),
)

app = FastAPI(title="MLOps Demo Prediction API", version="1.0.0")
_model = None
_features: list[str] = []
_model_version: str | None = None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(
        ..., description="Mapping of model feature names to numeric values"
    )


@app.on_event("startup")
def load_model() -> None:
    global _model, _features, _model_version
    requested_version = os.getenv("MODEL_VERSION", "latest")
    if requested_version != "latest" and not _is_valid_model_version(requested_version):
        raise ValueError(
            "MODEL_VERSION must contain only ASCII letters, digits, '_' and '-'."
        )

    if requested_version == "latest":
        if VERSION_MANIFEST_PATH.exists():
            requested_version = json.loads(
                VERSION_MANIFEST_PATH.read_text(encoding="utf-8")
            ).get("model_version")
            if not _is_valid_model_version(requested_version):
                raise ValueError("Model version manifest contains an invalid version.")
        else:
            requested_version = ""

    if requested_version:
        model_path = MODEL_PATH.parent / "versions" / requested_version / "model.joblib"
        features_path = (
            MODEL_PATH.parent / "versions" / requested_version / "feature_names.json"
        )
        if not model_path.is_file() or not features_path.is_file():
            raise FileNotFoundError(
                f"Model version {requested_version!r} is missing from "
                f"{model_path.parent}."
            )
        _model_version = requested_version
    else:
        model_path = MODEL_PATH
        features_path = FEATURES_PATH
        _model_version = "unversioned" if model_path.exists() else None

    if model_path.exists() and features_path.exists():
        _model = joblib.load(model_path)
        _features = json.loads(features_path.read_text(encoding="utf-8"))


def _is_valid_model_version(version: Any) -> bool:
    return isinstance(version, str) and bool(version) and all(
        char.isascii() and (char.isalnum() or char in "_-") for char in version
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _model_version,
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def collect_http_metrics(request: Request, call_next):
    route_name = (
        request.url.path
        if request.url.path in {"/health", "/metrics", "/predict"}
        else "unmatched"
    )
    status_code = "500"
    with HTTP_REQUEST_DURATION.labels(request.method, route_name).time():
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        finally:
            HTTP_REQUESTS.labels(
                request.method, route_name, status_code
            ).inc()


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
        PREDICTIONS.labels(str(prediction)).inc()
        result: dict[str, Any] = {"prediction": prediction}
        if hasattr(_model, "predict_proba"):
            result["probabilities"] = [
                float(value) for value in _model.predict_proba(row)[0]
            ]
        return result
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid numeric feature values.") from exc
