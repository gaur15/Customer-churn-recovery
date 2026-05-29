"""
feature_engineering.py
-----------------------
Creates RFM scores, engagement ratios, interaction features,
and tenure buckets on top of the raw cleaned dataframe.

Call `engineer_features(df)` BEFORE encode_categoricals / split_and_scale.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ── RFM Scoring ────────────────────────────────────────────────────────────────
def add_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recency  → last_interaction_days  (lower = better, so invert)
    Frequency → login_frequency
    Monetary  → monthly_charges
    Score each 1–5, then combine into rfm_score.
    """
    df = df.copy()
    df["rfm_recency"]   = pd.qcut(df["last_interaction_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["rfm_frequency"] = pd.qcut(df["login_frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["rfm_monetary"]  = pd.qcut(df["monthly_charges"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["rfm_score"]     = df["rfm_recency"] + df["rfm_frequency"] + df["rfm_monetary"]
    logger.info("RFM scores added ✓")
    return df


# ── Engagement Ratio ──────────────────────────────────────────────────────────
def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Support-to-login ratio signals frustration; high = bad."""
    df = df.copy()
    df["support_to_login_ratio"] = df["support_tickets"] / (df["login_frequency"] + 1)
    df["charge_per_product"]     = df["monthly_charges"] / (df["num_products"] + 1)
    df["avg_monthly_spend"]      = df["total_charges"] / (df["tenure_months"] + 1)
    logger.info("Engagement features added ✓")
    return df


# ── Tenure Buckets ─────────────────────────────────────────────────────────────
def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Discretise tenure into human-readable bands."""
    df = df.copy()
    bins   = [0, 6, 12, 24, 48, float("inf")]
    labels = ["0-6m", "6-12m", "1-2yr", "2-4yr", "4yr+"]
    df["tenure_bucket"] = pd.cut(df["tenure_months"], bins=bins, labels=labels)
    df["tenure_bucket"] = df["tenure_bucket"].astype(str)
    logger.info("Tenure bucket added ✓")
    return df


# ── Risk Flag ─────────────────────────────────────────────────────────────────
def add_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags that are strong churn signals."""
    df = df.copy()
    df["flag_low_satisfaction"]  = (df["satisfaction_score"] <= 2).astype(int)
    df["flag_high_support"]      = (df["support_tickets"] >= 3).astype(int)
    df["flag_payment_delay"]     = (df["payment_delays"] >= 2).astype(int)
    df["flag_inactive"]          = (df["last_interaction_days"] > 90).astype(int)
    df["flag_mtm_contract"]      = (df["contract_type"] == "Month-to-Month").astype(int)
    df["composite_risk_score"]   = (
        df["flag_low_satisfaction"]
        + df["flag_high_support"]
        + df["flag_payment_delay"]
        + df["flag_inactive"]
        + df["flag_mtm_contract"]
    )
    logger.info("Risk flags added ✓")
    return df


# ── Master entry point ─────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps in sequence."""
    df = add_rfm_scores(df)
    df = add_engagement_features(df)
    df = add_tenure_bucket(df)
    df = add_risk_flags(df)
    logger.info(f"Feature engineering complete → {df.shape[1]} columns total")
    return df
