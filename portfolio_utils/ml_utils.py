"""
2026-style ML utilities: reproducibility, pipelines, optional SHAP.
Use in notebooks for consistent seeds and pipeline patterns.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set global seeds for reproducibility (numpy, random; extend for TF/torch if needed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass


def build_classification_pipeline(
    estimator: Any,
    scale: bool = True,
    impute: bool = False,
    selector: Optional[str] = None,
    selector_k: int = 20,
) -> Any:
    """
    Build a sklearn Pipeline for classification (preprocess + estimator).
    Use with cross_validate so preprocessing is fitted only on train folds.

    Parameters
    ----------
    estimator : classifier instance
    scale : use StandardScaler before estimator
    impute : use SimpleImputer(strategy='median') for numeric NaNs
    selector : 'kbest' for SelectKBest(f_classif), None to skip
    selector_k : k for SelectKBest

    Returns
    -------
    Pipeline
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_classif

    steps = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("scaler", StandardScaler()))
    if selector == "kbest":
        steps.append(("selector", SelectKBest(f_classif, k=max(1, selector_k))))
    steps.append(("estimator", estimator))
    return Pipeline(steps=steps)


def shap_summary_if_available(
    model: Any,
    X: Any,
    max_display: int = 15,
    title: Optional[str] = None,
) -> Optional[Any]:
    """
    Compute and plot SHAP summary if the shap package is installed.
    For tree models uses TreeExplainer; else uses KernelExplainer (slow on large X).
    Pass a small sample (e.g. X_test[:500]) for KernelExplainer.

    Returns
    -------
    shap.Explanation or None if shap not installed / error
    """
    try:
        import shap
    except ImportError:
        return None
    try:
        if hasattr(model, "predict_proba") and getattr(model, "feature_importances_", None) is not None:
            explainer = shap.TreeExplainer(model, X)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, min(100, len(X))))
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        return shap_values
    except Exception:
        return None


__all__ = ["set_seed", "build_classification_pipeline", "shap_summary_if_available"]
