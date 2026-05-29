"""
tests/test_revenue_recovery.py
-------------------------------
Unit tests for the revenue recovery scoring and tiering logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from src.revenue_recovery import (
    compute_recovery_scores,
    assign_tiers,
    attach_offers,
    estimate_recovery_value,
    build_recovery_plan,
    OFFER_CATALOGUE,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_inputs():
    n = 300
    np.random.seed(99)
    return {
        "customer_ids": np.array([f"C{i:04d}" for i in range(n)]),
        "churn_proba":  np.random.uniform(0, 1, n),
        "clv":          np.random.uniform(100, 5000, n),
    }


# ── compute_recovery_scores ────────────────────────────────────────────────────
def test_recovery_scores_shape(sample_inputs):
    df = compute_recovery_scores(**sample_inputs)
    assert len(df) == len(sample_inputs["customer_ids"])
    assert "recovery_score" in df.columns


def test_recovery_scores_sorted(sample_inputs):
    df = compute_recovery_scores(**sample_inputs)
    assert df["recovery_score"].is_monotonic_decreasing


def test_recovery_score_formula(sample_inputs):
    df = compute_recovery_scores(**sample_inputs)
    expected = sample_inputs["churn_proba"] * sample_inputs["clv"]
    # values may be reordered — check via merge
    merged = df.merge(
        pd.DataFrame({"customer_id": sample_inputs["customer_ids"], "expected": np.round(expected, 2)}),
        on="customer_id"
    )
    np.testing.assert_allclose(merged["recovery_score"], merged["expected"], rtol=1e-3)


# ── assign_tiers ───────────────────────────────────────────────────────────────
def test_assign_tiers_valid_values(sample_inputs):
    df = compute_recovery_scores(**sample_inputs)
    df = assign_tiers(df)
    assert set(df["tier"].unique()).issubset({"High", "Medium", "Low"})


def test_assign_tiers_no_nulls(sample_inputs):
    df = compute_recovery_scores(**sample_inputs)
    df = assign_tiers(df)
    assert df["tier"].isnull().sum() == 0


# ── attach_offers ──────────────────────────────────────────────────────────────
def test_attach_offers_all_tiers_covered(sample_inputs):
    df = compute_recovery_scores(**sample_inputs)
    df = assign_tiers(df)
    df = attach_offers(df)
    for tier in ["High", "Medium", "Low"]:
        rows = df[df["tier"] == tier]
        if len(rows) > 0:
            assert (rows["offer"] == OFFER_CATALOGUE[tier]["offer"]).all()


def test_attach_offers_discount_pct_positive(sample_inputs):
    df = compute_recovery_scores(**sample_inputs)
    df = assign_tiers(df)
    df = attach_offers(df)
    assert (df["discount_pct"] > 0).all()


# ── estimate_recovery_value ────────────────────────────────────────────────────
def test_estimate_recovery_value_non_negative(sample_inputs):
    df = compute_recovery_scores(**sample_inputs)
    df = assign_tiers(df)
    df = attach_offers(df)
    df = estimate_recovery_value(df)
    assert (df["projected_recovery"] >= 0).all()


# ── build_recovery_plan (integration) ─────────────────────────────────────────
def test_build_recovery_plan_integration(sample_inputs, tmp_path, monkeypatch):
    # Redirect CSV output to tmp dir
    monkeypatch.chdir(tmp_path)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    result = build_recovery_plan(**sample_inputs)
    assert isinstance(result, pd.DataFrame)
    assert "tier" in result.columns
    assert "projected_recovery" in result.columns
    assert os.path.exists("data/processed/recovery_plan.csv")
