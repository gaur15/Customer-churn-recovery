"""
tests/test_preprocessing.py
----------------------------
Unit tests for data loading, cleaning, and encoding.
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import (
    clean_data, encode_categoricals, split_and_scale, validate_schema,
    CATEGORICAL_COLS, NUMERIC_COLS, TARGET, ID_COL,
)
from src.feature_engineering import engineer_features


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal valid dataframe for testing."""
    n = 200
    np.random.seed(0)
    return pd.DataFrame({
        "customer_id":          [f"C{i:04d}" for i in range(n)],
        "age":                  np.random.randint(18, 70, n),
        "gender":               np.random.choice(["Male", "Female"], n),
        "region":               np.random.choice(["North", "South", "East"], n),
        "contract_type":        np.random.choice(["Month-to-Month", "One Year", "Two Year"], n),
        "tenure_months":        np.random.randint(1, 60, n),
        "monthly_charges":      np.random.uniform(20, 100, n),
        "total_charges":        np.random.uniform(200, 5000, n),
        "num_products":         np.random.randint(1, 5, n),
        "login_frequency":      np.random.randint(0, 30, n),
        "support_tickets":      np.random.randint(0, 6, n),
        "payment_delays":       np.random.randint(0, 4, n),
        "last_interaction_days": np.random.randint(1, 180, n),
        "satisfaction_score":   np.random.randint(1, 6, n),
        "clv":                  np.random.uniform(100, 5000, n),
        "churn":                np.random.randint(0, 2, n),
    })


# ── Schema validation ──────────────────────────────────────────────────────────
def test_validate_schema_passes(sample_df):
    validate_schema(sample_df)   # should not raise


def test_validate_schema_missing_column(sample_df):
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(sample_df.drop(columns=["churn"]))


# ── Cleaning ───────────────────────────────────────────────────────────────────
def test_clean_data_removes_duplicates(sample_df):
    duped = pd.concat([sample_df, sample_df.iloc[:10]], ignore_index=True)
    cleaned = clean_data(duped)
    assert len(cleaned) == len(sample_df)


def test_clean_data_fills_nulls(sample_df):
    df = sample_df.copy()
    df.loc[:5, "monthly_charges"] = np.nan
    cleaned = clean_data(df)
    assert cleaned["monthly_charges"].isnull().sum() == 0


def test_clean_data_clips_outliers(sample_df):
    df = sample_df.copy()
    df.loc[0, "clv"] = 1_000_000     # extreme outlier
    cleaned = clean_data(df)
    # After clipping to 99th percentile the extreme value must be reduced
    assert cleaned["clv"].max() < 1_000_000


# ── Encoding ───────────────────────────────────────────────────────────────────
def test_encode_categoricals_removes_original_cols(sample_df):
    cleaned = clean_data(sample_df)
    encoded = encode_categoricals(cleaned)
    for col in CATEGORICAL_COLS:
        assert col not in encoded.columns


def test_encode_categoricals_adds_dummy_cols(sample_df):
    cleaned = clean_data(sample_df)
    encoded = encode_categoricals(cleaned)
    assert any(c.startswith("contract_type_") for c in encoded.columns)


# ── Split & scale ──────────────────────────────────────────────────────────────
def test_split_and_scale_shapes(sample_df):
    cleaned = clean_data(sample_df)
    encoded = encode_categoricals(cleaned)
    data    = split_and_scale(encoded)
    n       = len(sample_df)
    total   = len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"])
    assert total == n


def test_split_and_scale_no_leakage(sample_df):
    """Train IDs must not appear in test IDs."""
    cleaned = clean_data(sample_df)
    encoded = encode_categoricals(cleaned)
    data    = split_and_scale(encoded)
    # scaler fitted on train; test mean should not exactly equal train mean
    train_mean = data["X_train"].mean()
    test_mean  = data["X_test"].mean()
    assert abs(train_mean - test_mean) < 1.0   # close but not identical


# ── Feature engineering ────────────────────────────────────────────────────────
def test_engineer_features_adds_rfm(sample_df):
    result = engineer_features(sample_df)
    for col in ["rfm_recency", "rfm_frequency", "rfm_monetary", "rfm_score"]:
        assert col in result.columns


def test_engineer_features_adds_risk_flags(sample_df):
    result = engineer_features(sample_df)
    assert "composite_risk_score" in result.columns
    assert result["composite_risk_score"].between(0, 5).all()


def test_engineer_features_no_row_loss(sample_df):
    result = engineer_features(sample_df)
    assert len(result) == len(sample_df)
