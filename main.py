"""
main.py
-------
End-to-end pipeline runner for the Customer Churn & Revenue Recovery project.

Steps:
  1. Load & preprocess data
  2. Feature engineering
  3. EDA visualizations
  4. Model training (Random Forest + GridSearchCV)
  5. Model evaluation & SHAP explanation
  6. Hypothesis testing (statistical revenue analysis)
  7. Revenue recovery scoring & offer assignment
  8. Save all artefacts

Usage:
    python main.py
    python main.py --fast       # quick training (fewer hyperparams)
    python main.py --no-shap    # skip SHAP (slow on large datasets)
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ── imports ────────────────────────────────────────────────────────────────────
from src.data_preprocessing  import load_data, validate_schema, clean_data, encode_categoricals, split_and_scale
from src.feature_engineering import engineer_features
from src.visualizations      import run_all_eda_plots
from src.model               import train_model, evaluate_model, explain_with_shap, save_model
from src.hypothesis_testing  import run_all_tests
from src.revenue_recovery    import build_recovery_plan


NUMERIC_COLS = [
    "age", "tenure_months", "monthly_charges", "total_charges",
    "num_products", "login_frequency", "support_tickets",
    "payment_delays", "last_interaction_days", "satisfaction_score", "clv",
]


# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(fast: bool = True, run_shap: bool = True) -> None:

    logger.info("=" * 60)
    logger.info("  CUSTOMER CHURN & REVENUE RECOVERY — FULL PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Data ───────────────────────────────────────────────────────────
    logger.info("\n[1/7] Loading & preprocessing data …")
    raw_df = load_data("data/processed/customer_data.csv")
    validate_schema(raw_df)
    clean_df = clean_data(raw_df)

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    logger.info("\n[2/7] Engineering features …")
    feat_df = engineer_features(clean_df)

    # Keep a copy before encoding for EDA + hypothesis testing
    analysis_df = feat_df.copy()

    # ── Step 3: EDA ───────────────────────────────────────────────────────────
    logger.info("\n[3/7] Generating EDA visualizations …")
    run_all_eda_plots(analysis_df, NUMERIC_COLS)

    # ── Step 4: Encode + split + scale ────────────────────────────────────────
    encoded_df  = encode_categoricals(feat_df)
    data        = split_and_scale(encoded_df)

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    feature_names           = data["feature_names"]
    customer_ids_test       = data["customer_ids_test"]

    # ── Step 5: Model training & evaluation ───────────────────────────────────
    logger.info("\n[4/7] Training Random Forest …")
    model = train_model(X_train, y_train, X_val, y_val, fast=fast)
    save_model(model)

    logger.info("\n[5/7] Evaluating model …")
    metrics = evaluate_model(model, X_test, y_test, feature_names)

    if run_shap:
        explain_with_shap(model, X_test, feature_names)

    # ── Step 6: Hypothesis testing ────────────────────────────────────────────
    logger.info("\n[6/7] Running hypothesis tests …")
    run_all_tests(analysis_df)

    # ── Step 7: Revenue recovery plan ─────────────────────────────────────────
    logger.info("\n[7/7] Building revenue recovery plan …")

    # Get churn probabilities for test set customers
    churn_proba = model.predict_proba(X_test)[:, 1]

    # Look up CLV from original data for test customers
    clv_lookup = analysis_df.set_index("customer_id")["clv"].to_dict()
    clv_values = np.array([clv_lookup.get(cid, analysis_df["clv"].median()) for cid in customer_ids_test])

    recovery_df = build_recovery_plan(customer_ids_test, churn_proba, clv_values)

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    logger.info(f"  F1 Score  : {metrics['f1']:.4f}")
    logger.info(f"  Revenue at risk  : ₹{recovery_df['recovery_score'].sum():,.2f}")
    logger.info(f"  Projected recovery: ₹{recovery_df['projected_recovery'].sum():,.2f}")
    logger.info("\n  Artefacts saved:")
    logger.info("    models/random_forest_churn.pkl")
    logger.info("    data/processed/recovery_plan.csv")
    logger.info("    reports/figures/*.png")
    logger.info("\n  To explore the dashboard, run:")
    logger.info("    python dashboards/dashboard.py")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn & Revenue Recovery Pipeline")
    parser.add_argument("--fast",    action="store_true", help="Use reduced hyperparameter grid")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP explanation")
    args = parser.parse_args()

    # Auto-generate data if not present
    if not os.path.exists("data/processed/customer_data.csv"):
        logger.info("No data found — generating synthetic dataset …")
        import subprocess
        subprocess.run([sys.executable, "data/generate_data.py"], check=True)

    run_pipeline(fast=args.fast, run_shap=not args.no_shap)
