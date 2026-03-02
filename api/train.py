"""
Train the Telecom Churn pipeline and serialize artifacts for serving.

Usage:
    python -m api.train              # train with defaults
    python -m api.train --output-dir artifacts/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from portfolio_utils import set_seed
from portfolio_utils.data_loader import load_telecom_churn

COST_FN = 500  # missed churner: lose customer lifetime value
COST_FP = 75   # unnecessary retention offer
DROP_COLS = [
    "customerID", "CustomerID", "TotalCharges", "Total Charges",
    "Churn Label", "Churn Value", "Churn Score", "CLTV", "Churn Reason",
    "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
]


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean, encode, and split features/target."""
    drop = [c for c in DROP_COLS if c in df.columns]
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    X_raw = df.drop(columns=["Churn"] + drop, errors="ignore")
    X = pd.get_dummies(X_raw, drop_first=True).astype(float)
    return X, y


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> tuple[float, dict]:
    """Find threshold minimizing business cost (FN*cost_fn + FP*cost_fp)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_cost = float("inf")
    best_thresh = 0.5
    best_detail = {}
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    for i, t in enumerate(thresholds):
        preds = (y_prob >= t).astype(int)
        fn = int(((y_true == 1) & (preds == 0)).sum())
        fp = int(((y_true == 0) & (preds == 1)).sum())
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost = cost
            best_thresh = float(t)
            best_detail = {
                "threshold": float(t),
                "false_negatives": fn,
                "false_positives": fp,
                "total_cost": float(cost),
                "missed_churner_cost": float(fn * cost_fn),
                "unnecessary_offer_cost": float(fp * cost_fp),
            }
    return best_thresh, best_detail


def train(output_dir: str = "artifacts") -> None:
    set_seed(42)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading telecom churn data...")
    df = load_telecom_churn()
    X, y = preprocess(df)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = float(len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
        )),
    ])

    print("Cross-validating (5-fold stratified)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=False,
    )
    cv_metrics = {
        k.replace("test_", ""): float(np.mean(v))
        for k, v in cv_results.items()
        if k.startswith("test_")
    }
    print(f"CV metrics: {cv_metrics}")

    print("Fitting final pipeline on full training set...")
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    optimal_threshold, cost_detail = find_optimal_threshold(y_test.values, y_prob)

    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    holdout_metrics = {
        "precision": float(precision_score(y_test, y_pred_optimal)),
        "recall": float(recall_score(y_test, y_pred_optimal)),
        "f1": float(f1_score(y_test, y_pred_optimal)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    print(f"Holdout metrics (threshold={optimal_threshold:.3f}): {holdout_metrics}")
    print(f"Cost analysis: {cost_detail}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred_optimal, target_names=["No Churn", "Churn"]))

    print("Saving artifacts...")
    joblib.dump(pipe, out / "churn_pipeline.joblib")

    metadata = {
        "model_type": "XGBClassifier",
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "cv_metrics": cv_metrics,
        "holdout_metrics": holdout_metrics,
        "optimal_threshold": optimal_threshold,
        "cost_assumptions": {"cost_fn": COST_FN, "cost_fp": COST_FP},
        "cost_detail": cost_detail,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "class_distribution": {
            "train_churn_pct": float(y_train.mean()),
            "test_churn_pct": float(y_test.mean()),
        },
    }
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Artifacts saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--output-dir", default="artifacts", help="Where to save model artifacts")
    args = parser.parse_args()
    train(args.output_dir)
