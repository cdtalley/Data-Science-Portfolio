"""
Portfolio utilities: data loading, ML helpers (2026-style pipelines, seeds, SHAP).
Use DATA_DIR env var or default ./data; run setup_data.py to download datasets.

ML helpers (set_seed, build_classification_pipeline, shap_summary_if_available) are
imported first and only require numpy (and sklearn for pipeline/SHAP). Data loaders
require pandas and other deps; if not installed, use: pip install -r requirements.txt
"""

from portfolio_utils.ml_utils import (
    set_seed,
    build_classification_pipeline,
    shap_summary_if_available,
)

try:
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
    _data_loader_available = True
except ImportError as e:
    _data_loader_available = False
    _data_loader_error = e

    def _require_data_loader(*args, **kwargs):
        raise ImportError(
            "Data loaders require pandas and other dependencies. "
            "Install with: pip install -r requirements.txt"
        ) from e

    get_data_dir = ensure_dataset = load_bankruptcy = load_telecom_churn = _require_data_loader
    load_nj_transit = load_heart = load_nyc_bus = load_jane_street = _require_data_loader

try:
    from portfolio_utils.db_utils import DuckDBLoader
except ImportError:
    DuckDBLoader = None

__all__ = [
    "set_seed",
    "build_classification_pipeline",
    "shap_summary_if_available",
    "get_data_dir",
    "ensure_dataset",
    "load_bankruptcy",
    "load_telecom_churn",
    "load_nj_transit",
    "load_heart",
    "load_nyc_bus",
    "load_jane_street",
    "DuckDBLoader",
]
