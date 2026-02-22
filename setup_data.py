#!/usr/bin/env python3
"""
Download all portfolio datasets from Kaggle into data/.
Requires: pip install kaggle, and ~/.kaggle/kaggle.json (from Kaggle Account > Create New Token).
Optional: Jane Street competition requires accepting rules at
  https://www.kaggle.com/c/jane-street-market-prediction (skip with --no-jane-street).
"""

import argparse
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from portfolio_utils.data_loader import (
    get_data_dir,
    ensure_dataset,
    KAGGLE_DATASETS,
)


def main():
    parser = argparse.ArgumentParser(description="Download portfolio datasets from Kaggle.")
    parser.add_argument(
        "--no-jane-street",
        action="store_true",
        help="Skip Jane Street (competition; requires accepting rules and large download).",
    )
    args = parser.parse_args()
    data_dir = get_data_dir()
    print(f"Data directory: {data_dir}")

    for key in KAGGLE_DATASETS:
        print(f"Ensuring dataset: {key} ...")
        try:
            ensure_dataset(key)
            print(f"  OK: {key}")
        except Exception as e:
            print(f"  FAIL: {key} - {e}")
            if key == "jane_street" and "competition" in str(e).lower():
                print("  Tip: Accept competition rules at https://www.kaggle.com/c/jane-street-market-prediction")
            raise

    if not args.no_jane_street:
        print("Ensuring Jane Street (competition) ...")
        try:
            from portfolio_utils.data_loader import load_jane_street
            load_jane_street()
            print("  OK: jane_street")
        except Exception as e:
            print(f"  SKIP or FAIL: {e}")
            print("  To skip: run with --no-jane-street")

    print("Done.")


if __name__ == "__main__":
    main()
