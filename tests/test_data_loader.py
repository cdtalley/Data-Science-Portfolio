"""Tests for portfolio_utils.data_loader — config, path helpers, preprocessing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from portfolio_utils.data_loader import (
    DATA_DIR,
    KAGGLE_DATASETS,
    _find_csv,
    get_data_dir,
)


class TestGetDataDir:
    def test_returns_absolute_path(self):
        result = get_data_dir()
        assert os.path.isabs(result)

    def test_creates_directory(self, tmp_path):
        target = tmp_path / "test_data"
        with patch("portfolio_utils.data_loader.DATA_DIR", str(target)):
            result = get_data_dir()
            assert Path(result).exists()

    def test_respects_env_var(self, tmp_path):
        custom = str(tmp_path / "custom_data")
        with patch.dict(os.environ, {"DATA_DIR": custom}):
            with patch("portfolio_utils.data_loader.DATA_DIR", custom):
                result = get_data_dir()
                assert Path(result).exists()


class TestFindCsv:
    def test_finds_preferred_file(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")
        (tmp_path / "other.csv").write_text("x,y\n3,4\n")
        result = _find_csv(tmp_path, "data.csv")
        assert result.name == "data.csv"

    def test_finds_any_csv_when_no_preferred(self, tmp_path):
        (tmp_path / "only.csv").write_text("a,b\n1,2\n")
        result = _find_csv(tmp_path, None)
        assert result.suffix == ".csv"

    def test_raises_when_no_csv(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _find_csv(tmp_path, None)

    def test_case_insensitive_match(self, tmp_path):
        (tmp_path / "Data.csv").write_text("a,b\n1,2\n")
        result = _find_csv(tmp_path, "data.csv")
        assert result.name == "Data.csv"


class TestKaggleDatasets:
    def test_all_keys_present(self):
        expected = {"corporate_bankruptcy", "telecom_churn", "nj_transit", "heart_disease", "nyc_bus"}
        assert expected.issubset(set(KAGGLE_DATASETS.keys()))

    def test_slugs_are_strings(self):
        for key, (slug, _) in KAGGLE_DATASETS.items():
            assert isinstance(slug, str)
            assert "/" in slug, f"Kaggle slug for {key} should be 'user/dataset'"
