"""
revenue_recovery.py
-------------------
Scores at-risk customers by expected revenue recovery value,
segments them into tiers, and generates offer recommendations.

Revenue Recovery Score = churn_probability × CLV
"""

import pandas as pd
import numpy as np
import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
os.makedirs("reports/figures", exist_ok=True)


# ── Tier Thresholds (percentile-based) ────────────────────────────────────────
HIGH_TIER_PCT   = 0.75   # top 25% by recovery score = High
MEDIUM_TIER_PCT = 0.40   # next 35% = Medium, rest = Low


# ── Offer Catalogue ────────────────────────────────────────────────────────────
OFFER_CATALOGUE = {
    "High": {
        "offer":        "Personalised VIP Retention Package",
        "discount_pct": 25,
        "channel":      "Dedicated Account Manager Call",
        "description":  "Premium loyalty reward + 25% discount on annual upgrade + free product add-on.",
    },
    "Medium": {
        "offer":        "Loyalty Discount + Feature Unlock",
        "discount_pct": 15,
        "channel":      "Email + In-App Notification",
        "description":  "15% discount on next billing cycle + unlock premium feature for 3 months.",
    },
    "Low": {
        "offer":        "Standard Re-engagement Offer",
        "discount_pct": 5,
        "channel":      "Automated Email Campaign",
        "description":  "5% loyalty discount + helpful onboarding tips / tutorial push.",
    },
}


# ── Scoring ────────────────────────────────────────────────────────────────────
def compute_recovery_scores(
    customer_ids: np.ndarray,
    churn_proba: np.ndarray,
    clv: np.ndarray,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    customer_ids : array-like of customer IDs
    churn_proba  : array-like of churn probabilities [0, 1]
    clv          : array-like of customer lifetime values

    Returns
    -------
    DataFrame sorted by recovery_score descending
    """
    df = pd.DataFrame({
        "customer_id":    customer_ids,
        "churn_proba":    np.round(churn_proba, 4),
        "clv":            np.round(clv, 2),
        "recovery_score": np.round(churn_proba * clv, 2),
    })
    df.sort_values("recovery_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'tier' column: High / Medium / Low."""
    df = df.copy()
    high_thresh   = df["recovery_score"].quantile(HIGH_TIER_PCT)
    medium_thresh = df["recovery_score"].quantile(MEDIUM_TIER_PCT)

    conditions = [
        df["recovery_score"] >= high_thresh,
        (df["recovery_score"] >= medium_thresh) & (df["recovery_score"] < high_thresh),
    ]
    choices = ["High", "Medium"]
    df["tier"] = np.select(conditions, choices, default="Low")
    return df


def attach_offers(df: pd.DataFrame) -> pd.DataFrame:
    """Merge recommended offer details onto each customer row."""
    df = df.copy()
    offer_df = pd.DataFrame.from_dict(OFFER_CATALOGUE, orient="index").reset_index()
    offer_df.rename(columns={"index": "tier"}, inplace=True)
    df = df.merge(offer_df, on="tier", how="left")
    return df


def estimate_recovery_value(df: pd.DataFrame, avg_retention_rate: float = 0.15) -> pd.DataFrame:
    """
    Estimate projected revenue recovered if retention lift = avg_retention_rate.
    projected_recovery = recovery_score × avg_retention_rate × (1 - discount_pct/100)
    """
    df = df.copy()
    df["projected_recovery"] = np.round(
        df["recovery_score"] * avg_retention_rate * (1 - df["discount_pct"] / 100), 2
    )
    return df


# ── Reporting ──────────────────────────────────────────────────────────────────
def summarise_tiers(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("tier").agg(
        customers=("customer_id", "count"),
        avg_churn_proba=("churn_proba", "mean"),
        avg_clv=("clv", "mean"),
        total_recovery_score=("recovery_score", "sum"),
        total_projected_recovery=("projected_recovery", "sum"),
    ).round(2)
    summary.index = pd.CategoricalIndex(summary.index, categories=["High", "Medium", "Low"], ordered=True)
    summary.sort_index(inplace=True)
    logger.info("\n" + summary.to_string())
    return summary


def plot_recovery_by_tier(df: pd.DataFrame) -> None:
    summary = df.groupby("tier")[["recovery_score", "projected_recovery"]].sum()
    summary = summary.reindex(["High", "Medium", "Low"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart — total recovery score
    summary["recovery_score"].plot(
        kind="bar", ax=axes[0],
        color=["#EF4444", "#F97316", "#6B7280"]
    )
    axes[0].set_title("Total Revenue at Risk by Tier", fontsize=13)
    axes[0].set_ylabel("Recovery Score (₹)")
    axes[0].tick_params(axis="x", rotation=0)

    # Pie chart — projected recovery
    axes[1].pie(
        summary["projected_recovery"],
        labels=summary.index,
        autopct="%1.1f%%",
        colors=["#EF4444", "#F97316", "#6B7280"],
        startangle=90,
    )
    axes[1].set_title("Projected Recovery Share by Tier", fontsize=13)

    plt.tight_layout()
    plt.savefig("reports/figures/recovery_by_tier.png", dpi=150)
    plt.close()
    logger.info("Saved recovery_by_tier.png")


def plot_churn_proba_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for tier, color in [("High", "#EF4444"), ("Medium", "#F97316"), ("Low", "#22C55E")]:
        subset = df.loc[df["tier"] == tier, "churn_proba"]
        ax.hist(subset, bins=30, alpha=0.6, label=tier, color=color, density=True)
    ax.set_title("Churn Probability Distribution by Tier")
    ax.set_xlabel("Churn Probability")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/churn_proba_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved churn_proba_distribution.png")


# ── Master pipeline ────────────────────────────────────────────────────────────
def build_recovery_plan(
    customer_ids: np.ndarray,
    churn_proba: np.ndarray,
    clv: np.ndarray,
) -> pd.DataFrame:
    """
    Full recovery plan: score → tier → offer → projected revenue → summary.
    Returns the full customer-level dataframe.
    """
    df = compute_recovery_scores(customer_ids, churn_proba, clv)
    df = assign_tiers(df)
    df = attach_offers(df)
    df = estimate_recovery_value(df)

    logger.info("\n" + "=" * 50)
    logger.info("REVENUE RECOVERY PLAN SUMMARY")
    logger.info("=" * 50)
    summarise_tiers(df)
    plot_recovery_by_tier(df)
    plot_churn_proba_distribution(df)

    # Save to CSV for Power BI / dashboard consumption
    out_path = "data/processed/recovery_plan.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Recovery plan saved → {out_path}")

    return df
