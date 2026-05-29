"""
data_preprocessing.py
---------------------
Handles all data loading, cleaning, and preprocessing steps.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["gender", "region", "contract_type"]
NUMERIC_COLS = [
    "age", "tenure_months", "monthly_charges", "total_charges",
    "num_products", "login_frequency", "support_tickets",
    "payment_delays", "last_interaction_days", "satisfaction_score", "clv",
]
TARGET = "churn"
ID_COL = "customer_id"


# ── Loader ─────────────────────────────────────────────────────────────────────
def load_data(path: str = "data/processed/customer_data.csv") -> pd.DataFrame:
    """Load customer data from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found at '{path}'. "
            "Run `python data/generate_data.py` first."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> None:
    """Assert required columns exist and target is binary."""
    required = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET, ID_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    assert df[TARGET].isin([0, 1]).all(), "Target must be 0/1"
    logger.info("Schema validation passed ✓")


# ── Cleaning ───────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop duplicates
    - Handle nulls (median for numeric, mode for categorical)
    - Clip outliers at 1st/99th percentile for numeric columns
    """
    df = df.copy()

    before = len(df)
    df.drop_duplicates(subset=ID_COL, inplace=True)
    logger.info(f"Removed {before - len(df)} duplicate rows")

    # Fill nulls
    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"  Filled nulls in '{col}' with median={median_val:.2f}")

    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    # Clip outliers
    for col in ["monthly_charges", "total_charges", "clv", "support_tickets"]:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    logger.info("Cleaning complete ✓")
    return df


# ── Encoding ───────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    logger.info(f"After encoding: {df.shape[1]} columns")
    return df


# ── Split & Scale ──────────────────────────────────────────────────────────────
def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> dict:
    """
    Returns a dict with:
      X_train, X_val, X_test, y_train, y_val, y_test,
      scaler, feature_names, customer_ids_test
    """
    drop_cols = [ID_COL, TARGET]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[TARGET].values
    customer_ids = df[ID_COL].values

    # Train / temp split
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X, y, customer_ids, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    # Val / test split from temp
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp, y_temp, ids_temp, test_size=1 - val_ratio, random_state=random_state, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    logger.info(
        f"Split → train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}"
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val,   "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols,
        "customer_ids_test": ids_test,
    }


# ── Full pipeline ──────────────────────────────────────────────────────────────
def run_preprocessing(path: str = "data/processed/customer_data.csv") -> dict:
    df = load_data(path)
    validate_schema(df)
    df = clean_data(df)
    df = encode_categoricals(df)
    return split_and_scale(df)
