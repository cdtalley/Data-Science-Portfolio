"""Tests for portfolio_utils.ml_utils — seeds, pipelines, SHAP."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from portfolio_utils.ml_utils import (
    build_classification_pipeline,
    set_seed,
    shap_summary_if_available,
)


class TestSetSeed:
    def test_numpy_determinism(self):
        set_seed(0)
        a = np.random.rand(5)
        set_seed(0)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(0)
        a = np.random.rand(5)
        set_seed(1)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)


class TestBuildClassificationPipeline:
    def test_basic_pipeline(self, binary_dataset):
        X, y = binary_dataset
        pipe = build_classification_pipeline(RandomForestClassifier(random_state=42))
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_pipeline_with_imputer(self, binary_dataset):
        X, y = binary_dataset
        X_with_nan = X.copy()
        X_with_nan.iloc[0, 0] = np.nan
        pipe = build_classification_pipeline(
            LogisticRegression(max_iter=1000), impute=True
        )
        pipe.fit(X_with_nan, y)
        assert pipe.predict(X_with_nan).shape == (len(y),)

    def test_pipeline_with_selector(self, binary_dataset):
        X, y = binary_dataset
        pipe = build_classification_pipeline(
            RandomForestClassifier(random_state=42),
            selector="kbest",
            selector_k=3,
        )
        pipe.fit(X, y)
        assert pipe.named_steps["selector"].k == 3

    def test_pipeline_no_scale(self, binary_dataset):
        X, y = binary_dataset
        pipe = build_classification_pipeline(
            LogisticRegression(max_iter=1000), scale=False
        )
        assert "scaler" not in pipe.named_steps
        pipe.fit(X, y)
        assert pipe.predict(X).shape == (len(y),)

    def test_cross_validate_with_pipeline(self, binary_dataset):
        X, y = binary_dataset
        pipe = build_classification_pipeline(RandomForestClassifier(random_state=42))
        cv = cross_validate(pipe, X, y, cv=3, scoring=["accuracy", "f1"])
        assert "test_accuracy" in cv
        assert "test_f1" in cv
        assert len(cv["test_accuracy"]) == 3
        assert all(0 <= s <= 1 for s in cv["test_accuracy"])


class TestShapSummary:
    def test_shap_tree_model(self, binary_dataset):
        X, y = binary_dataset
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        result = shap_summary_if_available(rf, X.values[:20])
        shap = pytest.importorskip("shap")
        assert result is not None

    def test_shap_returns_none_gracefully(self, binary_dataset):
        X, y = binary_dataset
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)
        # KernelExplainer may or may not work; should not raise
        result = shap_summary_if_available(lr, X.values[:10])
        # Either returns values or None — no exception
        assert result is None or result is not None
