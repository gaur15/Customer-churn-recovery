"""
model.py
--------
Trains a Random Forest Classifier with GridSearchCV tuning.
Evaluates on test set and explains predictions with SHAP.
"""

import numpy as np
import pandas as pd
import logging
import os
import joblib
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

MODEL_PATH = "models/random_forest_churn.pkl"
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


# ── Hyperparameter Grid ────────────────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
    "class_weight":      ["balanced"],        # handle class imbalance
}

FAST_PARAM_GRID = {                            # for quick runs
    "n_estimators":      [100, 200],
    "max_depth":         [10, 20],
    "min_samples_split": [2, 5],
    "class_weight":      ["balanced"],
}


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fast: bool = True,
) -> RandomForestClassifier:
    """
    Train with GridSearchCV on train+val combined (using CV inside).
    Returns the best estimator.
    """
    logger.info("Starting GridSearchCV hyperparameter tuning …")
    grid = FAST_PARAM_GRID if fast else PARAM_GRID

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        estimator=rf,
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    # Combine train + val for tuning
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    gs.fit(X_all, y_all)

    best = gs.best_estimator_
    logger.info(f"Best params : {gs.best_params_}")
    logger.info(f"Best CV AUC : {gs.best_score_:.4f}")
    return best


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> Dict[str, Any]:
    """Compute & print all metrics; save figures."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }

    logger.info("\n" + "=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k.upper():12s}: {v:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curve(model, X_test, y_test)
    _plot_feature_importance(model, feature_names)

    return metrics


def _plot_confusion_matrix(y_test, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Saved confusion_matrix.png")


def _plot_roc_curve(model, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="Random Forest")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/roc_curve.png", dpi=150)
    plt.close()
    logger.info("Saved roc_curve.png")


def _plot_feature_importance(model, feature_names, top_n: int = 20) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", color="#2563EB", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=14)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png", dpi=150)
    plt.close()
    logger.info("Saved feature_importance.png")


# ── SHAP Explanation ───────────────────────────────────────────────────────────
def explain_with_shap(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    feature_names: list,
    sample_size: int = 500,
) -> None:
    """Generate SHAP summary plot for model explainability."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP explanation.")
        return

    logger.info("Computing SHAP values …")
    idx = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
    X_sample = X_test[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification shap_values is a list [class0, class1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_summary.png")


# ── Persistence ────────────────────────────────────────────────────────────────
def save_model(model: RandomForestClassifier, path: str = MODEL_PATH) -> None:
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def load_model(path: str = MODEL_PATH) -> RandomForestClassifier:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at '{path}'. Train first.")
    return joblib.load(path)


# ── Scoring new data ───────────────────────────────────────────────────────────
def predict_churn_proba(
    model: RandomForestClassifier,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predicted_labels, churn_probabilities)."""
    proba  = model.predict_proba(X)[:, 1]
    labels = (proba >= 0.5).astype(int)
    return labels, proba
