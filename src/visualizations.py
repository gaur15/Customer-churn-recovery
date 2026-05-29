"""
visualizations.py
-----------------
Standalone plotting helpers for EDA and model reports.
All figures saved to reports/figures/.
"""

import pandas as pd
import numpy as np
import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

logger = logging.getLogger(__name__)
os.makedirs("reports/figures", exist_ok=True)

PALETTE = {"Churned": "#EF4444", "Retained": "#22C55E"}


# ── EDA Overview ──────────────────────────────────────────────────────────────
def plot_churn_overview(df: pd.DataFrame) -> None:
    """4-panel overview of churn distribution."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Churn count
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["churn"].value_counts().rename({0: "Retained", 1: "Churned"})
    counts.plot(kind="bar", ax=ax1, color=["#22C55E", "#EF4444"], edgecolor="white")
    ax1.set_title("Churn vs Retained Count")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=0)

    # 2. Churn by contract type
    ax2 = fig.add_subplot(gs[0, 1])
    rates = df.groupby("contract_type")["churn"].mean().sort_values(ascending=False)
    rates.plot(kind="bar", ax=ax2, color="#3B82F6", edgecolor="white")
    ax2.set_title("Churn Rate by Contract Type")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.tick_params(axis="x", rotation=15)

    # 3. Tenure histogram
    ax3 = fig.add_subplot(gs[1, 0])
    for val, label in [(0, "Retained"), (1, "Churned")]:
        ax3.hist(df.loc[df["churn"] == val, "tenure_months"], bins=40,
                 alpha=0.6, label=label, color=PALETTE[label], density=True)
    ax3.set_title("Tenure Distribution")
    ax3.set_xlabel("Tenure (months)")
    ax3.legend()

    # 4. Satisfaction score
    ax4 = fig.add_subplot(gs[1, 1])
    churn_by_sat = df.groupby("satisfaction_score")["churn"].mean()
    churn_by_sat.plot(kind="bar", ax=ax4, color="#8B5CF6", edgecolor="white")
    ax4.set_title("Churn Rate by Satisfaction Score")
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax4.tick_params(axis="x", rotation=0)

    fig.suptitle("Customer Churn — EDA Overview", fontsize=16, y=1.01)
    plt.savefig("reports/figures/eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved eda_overview.png")


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> None:
    corr = df[numeric_cols + ["churn"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png", dpi=150)
    plt.close()
    logger.info("Saved correlation_heatmap.png")


def plot_monthly_charges_vs_churn(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for val, label in [(0, "Retained"), (1, "Churned")]:
        ax.hist(df.loc[df["churn"] == val, "monthly_charges"], bins=50,
                alpha=0.6, label=label, color=PALETTE[label], density=True)
    ax.set_title("Monthly Charges Distribution by Churn Status")
    ax.set_xlabel("Monthly Charges (₹)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/monthly_charges_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved monthly_charges_distribution.png")


def plot_rfm_heatmap(df: pd.DataFrame) -> None:
    if "rfm_recency" not in df.columns:
        logger.warning("RFM columns not found — skipping rfm_heatmap")
        return
    pivot = df.groupby(["rfm_recency", "rfm_frequency"])["churn"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Churn Rate by RFM Recency & Frequency Scores")
    ax.set_xlabel("Frequency Score")
    ax.set_ylabel("Recency Score")
    plt.tight_layout()
    plt.savefig("reports/figures/rfm_heatmap.png", dpi=150)
    plt.close()
    logger.info("Saved rfm_heatmap.png")


def run_all_eda_plots(df: pd.DataFrame, numeric_cols: list) -> None:
    plot_churn_overview(df)
    plot_correlation_heatmap(df, numeric_cols)
    plot_monthly_charges_vs_churn(df)
    plot_rfm_heatmap(df)
    logger.info("All EDA plots generated ✓")
