"""
FastAPI model serving for the Telecom Churn prediction pipeline.

Usage:
    uvicorn api.serve:app --reload
    # or: docker-compose up
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
    CustomerFeatures,
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Production-style model serving for Telecom Customer Churn. "
        "Trained with sklearn Pipeline (StandardScaler → XGBoost), "
        "cost-sensitive threshold, and SHAP explanations."
    ),
    version="1.0.0",
)

_pipeline = None
_metadata: dict = {}
_feature_names: list[str] = []
_threshold: float = 0.5


def _load_model() -> None:
    global _pipeline, _metadata, _feature_names, _threshold
    model_path = ARTIFACTS_DIR / "churn_pipeline.joblib"
    meta_path = ARTIFACTS_DIR / "model_metadata.json"
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. Run: python -m api.train"
        )
    _pipeline = joblib.load(model_path)
    _metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    _feature_names = _metadata.get("feature_names", [])
    _threshold = _metadata.get("optimal_threshold", 0.5)


@app.on_event("startup")
async def startup() -> None:
    _load_model()


def _customer_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert a single customer payload to the one-hot encoded DataFrame the pipeline expects."""
    raw = pd.DataFrame([customer.model_dump()])
    encoded = pd.get_dummies(raw, drop_first=True).astype(float)
    aligned = pd.DataFrame(columns=_feature_names, data=np.zeros((1, len(_feature_names))))
    for col in encoded.columns:
        if col in aligned.columns:
            aligned[col] = encoded[col].values
    return aligned


def _classify(prob: float) -> tuple[str, str, float]:
    """Return (risk_tier, recommended_action, estimated_cost)."""
    if prob >= 0.7:
        return (
            "High",
            "Immediate outreach: personalized retention offer + account review",
            500.0,
        )
    if prob >= _threshold:
        return (
            "Medium",
            "Proactive engagement: loyalty program enrollment + satisfaction survey",
            300.0,
        )
    return ("Low", "Standard service: no immediate action required", 0.0)


def _predict_single(customer: CustomerFeatures) -> PredictionResponse:
    X = _customer_to_dataframe(customer)
    prob = float(_pipeline.predict_proba(X)[0, 1])
    churn = prob >= _threshold
    tier, action, cost = _classify(prob)
    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=churn,
        risk_tier=tier,
        recommended_action=action,
        threshold_used=round(_threshold, 4),
        estimated_cost_if_missed=cost,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    if not _metadata:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    return ModelInfo(
        model_type=_metadata.get("model_type", "unknown"),
        n_features=_metadata.get("n_features", 0),
        feature_names=_feature_names,
        metrics={**_metadata.get("cv_metrics", {}), **_metadata.get("holdout_metrics", {})},
        threshold=_threshold,
        cost_assumptions=_metadata.get("cost_assumptions", {}),
        trained_at=_metadata.get("trained_at", "unknown"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predict_single(customer)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest) -> BatchPredictionResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    preds = [_predict_single(c) for c in req.customers]
    churners = [p for p in preds if p.churn_prediction]
    summary = BatchSummary(
        total=len(preds),
        predicted_churners=len(churners),
        high_risk=sum(1 for p in preds if p.risk_tier == "High"),
        medium_risk=sum(1 for p in preds if p.risk_tier == "Medium"),
        low_risk=sum(1 for p in preds if p.risk_tier == "Low"),
        total_risk_exposure=sum(p.estimated_cost_if_missed for p in churners),
    )
    return BatchPredictionResponse(predictions=preds, summary=summary)


@app.post("/explain", response_model=ExplainResponse)
async def explain(customer: CustomerFeatures) -> ExplainResponse:
    """SHAP-based explanation for a single prediction."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        import shap
    except ImportError:
        raise HTTPException(
            status_code=501, detail="SHAP not installed. pip install shap"
        )

    X = _customer_to_dataframe(customer)
    prob = float(_pipeline.predict_proba(X)[0, 1])

    estimator = _pipeline.named_steps["clf"]
    X_scaled = _pipeline.named_steps["scaler"].transform(X)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    contributions = sorted(
        zip(_feature_names, shap_values[0]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:10]

    return ExplainResponse(
        churn_probability=round(prob, 4),
        top_drivers=[
            FeatureContribution(
                feature=name,
                shap_value=round(float(val), 4),
                direction="increases_churn" if val > 0 else "decreases_churn",
            )
            for name, val in contributions
        ],
    )
