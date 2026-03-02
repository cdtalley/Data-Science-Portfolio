"""Tests for portfolio_utils.db_utils — DuckDB data access layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from portfolio_utils.db_utils import DuckDBLoader


@pytest.fixture
def db():
    loader = DuckDBLoader(":memory:")
    yield loader
    loader.close()


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(path, index=False)
    return str(path)


class TestDuckDBLoader:
    def test_load_csv(self, db, sample_csv):
        count = db.load_csv("test_table", sample_csv)
        assert count == 3

    def test_load_dataframe(self, db):
        df = pd.DataFrame({"x": [10, 20], "y": ["a", "b"]})
        count = db.load_dataframe("df_table", df)
        assert count == 2

    def test_query(self, db, sample_csv):
        db.load_csv("t", sample_csv)
        result = db.query("SELECT SUM(a) AS total FROM t")
        assert result["total"].iloc[0] == 6

    def test_tables(self, db, sample_csv):
        db.load_csv("alpha", sample_csv)
        db.load_csv("beta", sample_csv)
        tables = db.tables()
        assert "alpha" in tables
        assert "beta" in tables

    def test_describe(self, db, sample_csv):
        db.load_csv("t", sample_csv)
        desc = db.describe("t")
        assert len(desc) > 0

    def test_context_manager(self, sample_csv):
        with DuckDBLoader() as db:
            db.load_csv("t", sample_csv)
            result = db.query("SELECT COUNT(*) AS n FROM t")
            assert result["n"].iloc[0] == 3

    def test_missing_csv_raises(self, db):
        with pytest.raises(FileNotFoundError):
            db.load_csv("t", "/nonexistent/path.csv")

    def test_churn_queries_valid_sql(self, db):
        """Verify pre-built churn SQL queries parse without error."""
        from portfolio_utils.db_utils import CHURN_QUERIES

        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "customerID": [f"C{i}" for i in range(n)],
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "tenure": np.random.randint(1, 72, n),
            "MonthlyCharges": np.round(np.random.uniform(20, 100, n), 2),
            "Churn": np.random.choice(["Yes", "No"], n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
            "TechSupport": np.random.choice(["Yes", "No"], n),
        })
        db.load_dataframe("churn", df)
        for name, sql in CHURN_QUERIES.items():
            result = db.query(sql)
            assert len(result) > 0, f"Query '{name}' returned empty"
