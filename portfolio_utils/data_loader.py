"""
Centralized data loading for portfolio notebooks.
Uses Kaggle API when data is missing (run: pip install kaggle, configure ~/.kaggle/kaggle.json).
Set DATA_DIR to override default ./data.
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

# Kaggle dataset slug -> (subdir, expected filename or None to use first CSV in zip)
KAGGLE_DATASETS = {
    "corporate_bankruptcy": ("chihfongtsai/taiwanese-bankruptcy-prediction", None),
    "telecom_churn": ("waseemalastal/telco-customer-churn-ibm-dataset", None),
    "nj_transit": ("pranavbadami/nj-transit-amtrak-nec-performance", "2020_05.csv"),
    "heart_disease": ("johnsmith88/heart-disease-dataset", "Heart.csv"),
    "nyc_bus": ("stoney71/new-york-city-transport-statistics", "mta_1706.csv"),
}


def get_data_dir() -> str:
    """Return data directory; create if needed."""
    out = os.path.abspath(DATA_DIR)
    os.makedirs(out, exist_ok=True)
    return out


def _dataset_path(key: str) -> Path:
    base = Path(get_data_dir())
    return base / key


def _download_kaggle_dataset(slug: str, subdir: str, expected_file: Optional[str]) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("Install kaggle: pip install kaggle. Configure ~/.kaggle/kaggle.json from Kaggle account.")
    api = KaggleApi()
    api.authenticate()
    dest = Path(get_data_dir()) / subdir
    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(slug, path=str(dest), unzip=True)
    if expected_file:
        # Kaggle often extracts into a folder with dataset name; find file
        found = list(Path(dest).rglob(expected_file))
        if found:
            return found[0].parent
        # try direct in dest
        if (dest / expected_file).exists():
            return dest
    # use first csv in dest or any subdir
    for csv in Path(dest).rglob("*.csv"):
        return csv.parent
    return dest


def ensure_dataset(key: str) -> Path:
    """Download dataset from Kaggle if not present. Returns directory containing data."""
    if key not in KAGGLE_DATASETS:
        raise ValueError(f"Unknown dataset: {key}. Choose from {list(KAGGLE_DATASETS)}")
    slug, expected_file = KAGGLE_DATASETS[key]
    subdir = key
    path = _dataset_path(subdir)
    if path.exists() and (any(path.rglob("*.csv")) or any(path.rglob("*.xlsx"))):
        return path
    return _download_kaggle_dataset(slug, subdir, expected_file)


def _find_csv(path: Path, preferred: Optional[str] = None) -> Path:
    if preferred:
        for ext in ("", ".csv"):
            p = path / (preferred if preferred.endswith(".csv") else preferred + ext)
            if p.exists():
                return p
        for p in path.rglob("*.csv"):
            if p.name.lower() == preferred.lower() or (preferred and preferred.lower() in p.name.lower()):
                return p
    csvs = list(path.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {path}")
    if preferred:
        for c in csvs:
            if c.name == preferred or c.name.lower() == preferred.lower():
                return c
    return csvs[0]


def _find_xlsx(path: Path) -> Optional[Path]:
    xlsx = list(path.rglob("*.xlsx"))
    return xlsx[0] if xlsx else None


def load_bankruptcy() -> pd.DataFrame:
    """Load Corporate Bankruptcy (Taiwan) dataset."""
    base = ensure_dataset("corporate_bankruptcy")
    fp = _find_csv(Path(base), "data.csv")
    try:
        df = pd.read_csv(fp)
    except UnicodeDecodeError:
        df = pd.read_csv(fp, encoding="latin-1")
    # Normalize target name for portfolio notebooks (Bankrupt? vs Flag)
    if "Flag" in df.columns and "Bankrupt?" not in df.columns:
        df = df.rename(columns={"Flag": "Bankrupt?"})
    return df


def load_telecom_churn() -> pd.DataFrame:
    """Load IBM Telco Customer Churn (CSV or XLSX from Kaggle)."""
    base = Path(ensure_dataset("telecom_churn"))
    xlsx = _find_xlsx(base)
    if xlsx:
        df = pd.read_excel(xlsx)
    else:
        df = pd.read_csv(_find_csv(base, None))
    # Normalize column names for notebook compatibility (customerID, TotalCharges, Churn)
    rename = {}
    if "CustomerID" in df.columns and "customerID" not in df.columns:
        rename["CustomerID"] = "customerID"
    if "Total Charges" in df.columns and "TotalCharges" not in df.columns:
        rename["Total Charges"] = "TotalCharges"
    if rename:
        df = df.rename(columns=rename)
    if "Churn" not in df.columns and "Churn Label" in df.columns:
        df["Churn"] = df["Churn Label"].astype(str).str.strip()
    elif "Churn" not in df.columns and "Churn Value" in df.columns:
        df["Churn"] = df["Churn Value"].map({1: "Yes", 0: "No"})
    return df


def load_nj_transit() -> pd.DataFrame:
    """Load NJ Transit + Amtrak NEC performance (2020_05.csv)."""
    base = ensure_dataset("nj_transit")
    fp = _find_csv(Path(base), "2020_05.csv")
    return pd.read_csv(fp)


def load_heart() -> pd.DataFrame:
    """Load Heart Disease (Heart.csv)."""
    base = ensure_dataset("heart_disease")
    fp = _find_csv(Path(base), "Heart.csv")
    return pd.read_csv(fp)


def load_nyc_bus() -> pd.DataFrame:
    """Load NYC MTA bus data. Uses on_bad_lines='skip' for malformed rows."""
    base = ensure_dataset("nyc_bus")
    fp = _find_csv(Path(base), "mta_1706.csv")
    return pd.read_csv(fp, on_bad_lines="skip", low_memory=False)


def load_jane_street() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Jane Street train + features. Requires competition data (accept rules on Kaggle)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("pip install kaggle")
    api = KaggleApi()
    api.authenticate()
    dest = Path(get_data_dir()) / "jane_street"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("train.csv", "features.csv"):
        if not (dest / fname).exists():
            api.competition_download_file("jane-street-market-prediction", fname, path=str(dest))
    # train.csv can be huge; we load it
    train = pd.read_csv(dest / "train.csv")
    features = pd.read_csv(dest / "features.csv")
    return train, features
