"""
Portfolio utilities: data loading, ML helpers (2026-style pipelines, seeds, SHAP).
Use DATA_DIR env var or default ./data; run setup_data.py to download datasets.
"""

from portfolio_utils.data_loader import (
    get_data_dir,
    ensure_dataset,
    load_bankruptcy,
    load_telecom_churn,
    load_nj_transit,
    load_heart,
    load_nyc_bus,
    load_jane_street,
)
from portfolio_utils.ml_utils import (
    set_seed,
    build_classification_pipeline,
    shap_summary_if_available,
)

__all__ = [
    "get_data_dir",
    "ensure_dataset",
    "load_bankruptcy",
    "load_telecom_churn",
    "load_nj_transit",
    "load_heart",
    "load_nyc_bus",
    "load_jane_street",
    "set_seed",
    "build_classification_pipeline",
    "shap_summary_if_available",
]
