"""
tests/test_model.py
-------------------
Unit tests for model training, evaluation, and persistence.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import tempfile
from sklearn.datasets import make_classification

from src.model import train_model, evaluate_model, save_model, load_model, predict_churn_proba


# ── Fixture: small synthetic dataset ──────────────────────────────────────────
@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=600, n_features=20, n_informative=10,
        n_redundant=5, random_state=42, weights=[0.7, 0.3]
    )
    split = int(0.7 * len(X))
    val_split = int(0.85 * len(X))
    return {
        "X_train": X[:split],       "y_train": y[:split],
        "X_val":   X[split:val_split], "y_val":   y[split:val_split],
        "X_test":  X[val_split:],   "y_test":  y[val_split:],
        "feature_names": [f"f{i}" for i in range(20)],
    }


# ── Training ───────────────────────────────────────────────────────────────────
def test_train_model_returns_fitted_model(classification_data):
    d = classification_data
    model = train_model(d["X_train"], d["y_train"], d["X_val"], d["y_val"], fast=True)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_train_model_output_shape(classification_data):
    d = classification_data
    model = train_model(d["X_train"], d["y_train"], d["X_val"], d["y_val"], fast=True)
    preds = model.predict(d["X_test"])
    assert preds.shape == (len(d["X_test"]),)


# ── Evaluation ─────────────────────────────────────────────────────────────────
def test_evaluate_model_returns_dict(classification_data):
    d = classification_data
    model  = train_model(d["X_train"], d["y_train"], d["X_val"], d["y_val"], fast=True)
    metrics = evaluate_model(model, d["X_test"], d["y_test"], d["feature_names"])
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_model_accuracy_above_baseline(classification_data):
    """Model should beat random (>0.5 AUC)."""
    d = classification_data
    model  = train_model(d["X_train"], d["y_train"], d["X_val"], d["y_val"], fast=True)
    metrics = evaluate_model(model, d["X_test"], d["y_test"], d["feature_names"])
    assert metrics["roc_auc"] > 0.5


# ── Prediction ─────────────────────────────────────────────────────────────────
def test_predict_churn_proba_range(classification_data):
    d = classification_data
    model = train_model(d["X_train"], d["y_train"], d["X_val"], d["y_val"], fast=True)
    labels, proba = predict_churn_proba(model, d["X_test"])
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(labels).issubset({0, 1})


# ── Persistence ────────────────────────────────────────────────────────────────
def test_save_and_load_model(classification_data):
    d = classification_data
    model = train_model(d["X_train"], d["y_train"], d["X_val"], d["y_val"], fast=True)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name

    try:
        save_model(model, tmp_path)
        loaded = load_model(tmp_path)
        preds_original = model.predict(d["X_test"])
        preds_loaded   = loaded.predict(d["X_test"])
        np.testing.assert_array_equal(preds_original, preds_loaded)
    finally:
        os.unlink(tmp_path)
