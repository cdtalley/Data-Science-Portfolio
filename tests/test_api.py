"""Tests for the FastAPI churn prediction API (uses synthetic model artifacts)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from api.schemas import CustomerFeatures

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
}


@pytest.fixture(scope="module")
def _artifacts(tmp_path_factory):
    """Train a small synthetic model and write artifacts to a temp dir."""
    artifacts = tmp_path_factory.mktemp("artifacts")

    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "tenure": np.random.randint(0, 72, n),
        "PhoneService": np.random.choice(["Yes", "No"], n),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod": np.random.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], n),
        "MonthlyCharges": np.round(np.random.uniform(18, 118, n), 2),
    })
    y = np.random.choice([0, 1], n, p=[0.73, 0.27])
    X = pd.get_dummies(df, drop_first=True).astype(float)
    feature_names = X.columns.tolist()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=10)),
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, artifacts / "churn_pipeline.joblib")

    metadata = {
        "model_type": "XGBClassifier",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "cv_metrics": {"accuracy": 0.75, "f1": 0.60},
        "holdout_metrics": {"precision": 0.70, "recall": 0.65, "f1": 0.67, "roc_auc": 0.78},
        "optimal_threshold": 0.35,
        "cost_assumptions": {"cost_fn": 500, "cost_fp": 75},
        "trained_at": "2026-03-01T00:00:00+00:00",
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata))
    return artifacts, feature_names


@pytest.fixture(scope="module")
def client(_artifacts):
    """TestClient with patched artifacts directory."""
    artifacts_dir, _ = _artifacts
    with patch("api.serve.ARTIFACTS_DIR", artifacts_dir):
        from api.serve import app, _load_model
        _load_model()
        from fastapi.testclient import TestClient
        return TestClient(app)


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True


class TestModelInfo:
    def test_model_info(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        body = r.json()
        assert body["model_type"] == "XGBClassifier"
        assert body["n_features"] > 0
        assert isinstance(body["feature_names"], list)
        assert "threshold" in body


class TestPredict:
    def test_single_prediction(self, client):
        r = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["churn_probability"] <= 1
        assert isinstance(body["churn_prediction"], bool)
        assert body["risk_tier"] in ("Low", "Medium", "High")

    def test_batch_prediction(self, client):
        r = client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER] * 3})
        assert r.status_code == 200
        body = r.json()
        assert len(body["predictions"]) == 3
        assert body["summary"]["total"] == 3

    def test_invalid_payload_rejected(self, client):
        r = client.post("/predict", json={"gender": "Male"})
        assert r.status_code == 422


class TestExplain:
    def test_explain_returns_drivers(self, client):
        shap = pytest.importorskip("shap")
        r = client.post("/explain", json=SAMPLE_CUSTOMER)
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["churn_probability"] <= 1
        assert len(body["top_drivers"]) > 0
        assert body["top_drivers"][0]["direction"] in (
            "increases_churn", "decreases_churn"
        )
