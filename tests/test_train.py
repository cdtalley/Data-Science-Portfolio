"""Tests for api.train — preprocessing and threshold optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.train import find_optimal_threshold, preprocess


class TestPreprocess:
    def test_returns_features_and_target(self, churn_sample_df):
        X, y = preprocess(churn_sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == len(churn_sample_df)

    def test_drops_customer_id(self, churn_sample_df):
        X, y = preprocess(churn_sample_df)
        assert "customerID" not in X.columns

    def test_drops_total_charges(self, churn_sample_df):
        X, y = preprocess(churn_sample_df)
        assert "TotalCharges" not in X.columns

    def test_target_is_binary(self, churn_sample_df):
        _, y = preprocess(churn_sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_features_are_numeric(self, churn_sample_df):
        X, _ = preprocess(churn_sample_df)
        assert all(X.dtypes == float)

    def test_no_target_in_features(self, churn_sample_df):
        X, _ = preprocess(churn_sample_df)
        assert "Churn" not in X.columns


class TestFindOptimalThreshold:
    def test_returns_valid_threshold(self):
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9, 0.4, 0.6, 0.15, 0.85])
        thresh, detail = find_optimal_threshold(y_true, y_prob)
        assert 0 <= thresh <= 1
        assert "total_cost" in detail
        assert detail["total_cost"] >= 0

    def test_high_fn_cost_lowers_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9, 0.4, 0.6, 0.15, 0.85])
        thresh_high, _ = find_optimal_threshold(y_true, y_prob, cost_fn=10000, cost_fp=1)
        thresh_low, _ = find_optimal_threshold(y_true, y_prob, cost_fn=1, cost_fp=10000)
        assert thresh_high <= thresh_low
