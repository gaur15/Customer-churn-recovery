"""
hypothesis_testing.py
---------------------
Statistical analysis to quantify revenue at risk from churned customers.

Tests performed:
  1. Welch's t-test — mean CLV: Churned vs Retained
  2. Mann-Whitney U  — non-parametric revenue comparison
  3. Chi-square test — churn rate by contract type
  4. ANOVA           — CLV across satisfaction score groups
"""

import pandas as pd
import numpy as np
import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)
os.makedirs("reports/figures", exist_ok=True)


# ── 1. Welch's t-test on CLV ──────────────────────────────────────────────────
def test_clv_difference(df: pd.DataFrame) -> dict:
    """
    H0: Mean CLV of churned == Mean CLV of retained
    H1: Mean CLV of churned != Mean CLV of retained
    """
    churned  = df.loc[df["churn"] == 1, "clv"].dropna()
    retained = df.loc[df["churn"] == 0, "clv"].dropna()

    t_stat, p_value = stats.ttest_ind(churned, retained, equal_var=False)
    result = {
        "test":            "Welch's t-test (CLV)",
        "t_statistic":     round(t_stat, 4),
        "p_value":         round(p_value, 6),
        "churned_mean":    round(churned.mean(), 2),
        "retained_mean":   round(retained.mean(), 2),
        "revenue_at_risk": round((retained.mean() - churned.mean()) * len(churned), 2),
        "reject_h0":       p_value < 0.05,
    }
    _log_result(result)
    return result


# ── 2. Mann-Whitney U (non-parametric) ────────────────────────────────────────
def test_revenue_mannwhitney(df: pd.DataFrame) -> dict:
    """Non-parametric test for revenue distribution differences."""
    churned  = df.loc[df["churn"] == 1, "monthly_charges"].dropna()
    retained = df.loc[df["churn"] == 0, "monthly_charges"].dropna()

    u_stat, p_value = stats.mannwhitneyu(churned, retained, alternative="two-sided")
    result = {
        "test":             "Mann-Whitney U (Monthly Charges)",
        "u_statistic":      round(u_stat, 4),
        "p_value":          round(p_value, 6),
        "churned_median":   round(churned.median(), 2),
        "retained_median":  round(retained.median(), 2),
        "reject_h0":        p_value < 0.05,
    }
    _log_result(result)
    return result


# ── 3. Chi-square: contract type vs churn ─────────────────────────────────────
def test_contract_churn_association(df: pd.DataFrame) -> dict:
    """H0: Contract type and churn are independent."""
    contingency = pd.crosstab(df["contract_type"], df["churn"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    result = {
        "test":        "Chi-Square (Contract Type vs Churn)",
        "chi2":        round(chi2, 4),
        "p_value":     round(p_value, 6),
        "dof":         dof,
        "reject_h0":   p_value < 0.05,
        "contingency": contingency.to_dict(),
    }
    _log_result(result)
    return result


# ── 4. ANOVA: satisfaction score vs CLV ───────────────────────────────────────
def test_satisfaction_clv_anova(df: pd.DataFrame) -> dict:
    """H0: Mean CLV is equal across all satisfaction score groups."""
    groups = [df.loc[df["satisfaction_score"] == s, "clv"].dropna() for s in sorted(df["satisfaction_score"].unique())]
    f_stat, p_value = stats.f_oneway(*groups)
    result = {
        "test":      "One-way ANOVA (Satisfaction Score → CLV)",
        "f_statistic": round(f_stat, 4),
        "p_value":   round(p_value, 6),
        "reject_h0": p_value < 0.05,
        "group_means": {
            int(s): round(df.loc[df["satisfaction_score"] == s, "clv"].mean(), 2)
            for s in sorted(df["satisfaction_score"].unique())
        },
    }
    _log_result(result)
    return result


# ── Visualisations ─────────────────────────────────────────────────────────────
def plot_clv_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # KDE
    for label, group in df.groupby("churn"):
        name = "Churned" if label == 1 else "Retained"
        axes[0].hist(group["clv"], bins=60, alpha=0.6, label=name, density=True)
    axes[0].set_title("CLV Distribution: Churned vs Retained")
    axes[0].set_xlabel("Customer Lifetime Value (₹)")
    axes[0].legend()

    # Box
    sns.boxplot(x="churn", y="clv", data=df, ax=axes[1],
                palette={0: "#22C55E", 1: "#EF4444"})
    axes[1].set_xticklabels(["Retained", "Churned"])
    axes[1].set_title("CLV Spread by Churn Status")
    axes[1].set_ylabel("CLV (₹)")

    plt.tight_layout()
    plt.savefig("reports/figures/clv_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved clv_distribution.png")


def plot_churn_by_contract(df: pd.DataFrame) -> None:
    churn_rates = df.groupby("contract_type")["churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    churn_rates.plot(kind="bar", color=["#EF4444", "#F97316", "#22C55E"], ax=ax)
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reports/figures/churn_by_contract.png", dpi=150)
    plt.close()
    logger.info("Saved churn_by_contract.png")


# ── Runner ─────────────────────────────────────────────────────────────────────
def run_all_tests(df: pd.DataFrame) -> dict:
    logger.info("\n" + "=" * 50)
    logger.info("HYPOTHESIS TESTING")
    logger.info("=" * 50)
    results = {
        "clv_ttest":         test_clv_difference(df),
        "revenue_mannwhitney": test_revenue_mannwhitney(df),
        "contract_chisq":    test_contract_churn_association(df),
        "satisfaction_anova": test_satisfaction_clv_anova(df),
    }
    plot_clv_distribution(df)
    plot_churn_by_contract(df)
    return results


# ── Helpers ────────────────────────────────────────────────────────────────────
def _log_result(result: dict) -> None:
    logger.info(f"\n{'─'*40}")
    logger.info(f"Test      : {result['test']}")
    logger.info(f"p-value   : {result['p_value']}")
    logger.info(f"Reject H0 : {'YES ✗' if result['reject_h0'] else 'NO ✓'}")
