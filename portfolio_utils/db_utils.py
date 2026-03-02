"""
DuckDB-backed data access for portfolio projects.

Demonstrates SQL-based EDA and feature engineering alongside pandas — a common
pattern in production data science where data lives in warehouses (Snowflake,
BigQuery, Redshift) and analysts query with SQL before modeling in Python.

Usage:
    from portfolio_utils.db_utils import DuckDBLoader
    db = DuckDBLoader()
    db.load_csv("nj_transit", "data/nj_transit/2020_05.csv")
    df = db.query("SELECT * FROM nj_transit WHERE delay_minutes > 10")
    db.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class DuckDBLoader:
    """Lightweight DuckDB wrapper for SQL-based EDA and feature engineering."""

    def __init__(self, db_path: str = ":memory:"):
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "DuckDB not installed. Install with: pip install duckdb"
            )
        self._con = duckdb.connect(db_path)

    def load_csv(
        self,
        table_name: str,
        csv_path: str,
        **read_csv_kwargs,
    ) -> int:
        """Load a CSV into a DuckDB table. Returns row count."""
        path = Path(csv_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        self._con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS "
            f"SELECT * FROM read_csv_auto('{path}', ignore_errors=true)"
        )
        count = self._con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return count

    def load_dataframe(self, table_name: str, df: pd.DataFrame) -> int:
        """Register a pandas DataFrame as a DuckDB table."""
        self._con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df"
        )
        count = self._con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return count

    def query(self, sql: str) -> pd.DataFrame:
        """Run a SQL query and return results as a DataFrame."""
        return self._con.execute(sql).fetchdf()

    def tables(self) -> list[str]:
        """List all tables in the database."""
        result = self._con.execute("SHOW TABLES").fetchdf()
        return result["name"].tolist()

    def describe(self, table_name: str) -> pd.DataFrame:
        """Return column-level summary statistics for a table."""
        return self._con.execute(f"SUMMARIZE {table_name}").fetchdf()

    def profile(self, table_name: str) -> pd.DataFrame:
        """Quick data profile: types, nulls, distinct counts."""
        return self.query(f"""
            SELECT
                column_name,
                column_type,
                COUNT(*) AS total_rows,
                COUNT(*) - COUNT(column_name) AS null_count,
                ROUND(100.0 * (COUNT(*) - COUNT(column_name)) / COUNT(*), 2) AS null_pct
            FROM (SELECT * FROM {table_name})
            UNPIVOT (value FOR column_name IN (
                SELECT column_name FROM information_schema.columns
                WHERE table_name = '{table_name}'
            ))
            GROUP BY column_name, column_type
            ORDER BY null_pct DESC
        """)

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Pre-built SQL queries for common EDA patterns
NJ_TRANSIT_QUERIES = {
    "delay_by_station": """
        SELECT
            "from" AS station,
            COUNT(*) AS trips,
            ROUND(AVG("delay_minutes"), 1) AS avg_delay,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "delay_minutes"), 1) AS median_delay,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY "delay_minutes"), 1) AS p95_delay,
            SUM(CASE WHEN "delay_minutes" > 5 THEN 1 ELSE 0 END) AS delayed_trips,
            ROUND(100.0 * SUM(CASE WHEN "delay_minutes" > 5 THEN 1 ELSE 0 END) / COUNT(*), 1) AS delay_pct
        FROM nj_transit
        GROUP BY "from"
        ORDER BY avg_delay DESC
        LIMIT 20
    """,
    "delay_by_day_of_week": """
        SELECT
            DAYNAME(CAST("date" AS DATE)) AS day_of_week,
            DAYOFWEEK(CAST("date" AS DATE)) AS day_num,
            COUNT(*) AS trips,
            ROUND(AVG("delay_minutes"), 1) AS avg_delay,
            ROUND(100.0 * SUM(CASE WHEN "delay_minutes" > 5 THEN 1 ELSE 0 END) / COUNT(*), 1) AS delay_pct
        FROM nj_transit
        GROUP BY day_of_week, day_num
        ORDER BY day_num
    """,
    "worst_routes": """
        SELECT
            "from" || ' → ' || "to" AS route,
            COUNT(*) AS trips,
            ROUND(AVG("delay_minutes"), 1) AS avg_delay,
            ROUND(MAX("delay_minutes"), 1) AS max_delay
        FROM nj_transit
        GROUP BY "from", "to"
        HAVING COUNT(*) >= 10
        ORDER BY avg_delay DESC
        LIMIT 15
    """,
}

CHURN_QUERIES = {
    "churn_by_contract": """
        SELECT
            "Contract",
            COUNT(*) AS customers,
            SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) AS churned,
            ROUND(100.0 * SUM(CASE WHEN "Churn" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate,
            ROUND(AVG("MonthlyCharges"), 2) AS avg_monthly,
            ROUND(AVG("tenure"), 1) AS avg_tenure
        FROM churn
        GROUP BY "Contract"
        ORDER BY churn_rate DESC
    """,
    "high_value_at_risk": """
        SELECT
            "customerID",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "InternetService",
            "TechSupport"
        FROM churn
        WHERE "Churn" = 'Yes'
            AND "MonthlyCharges" > 80
            AND "tenure" > 12
        ORDER BY "MonthlyCharges" DESC
        LIMIT 20
    """,
    "revenue_impact": """
        SELECT
            "Churn",
            COUNT(*) AS customers,
            ROUND(SUM("MonthlyCharges"), 2) AS total_monthly_revenue,
            ROUND(AVG("MonthlyCharges"), 2) AS avg_monthly,
            ROUND(AVG("tenure"), 1) AS avg_tenure
        FROM churn
        GROUP BY "Churn"
    """,
}
