"""
generate_data.py
----------------
Generates a realistic synthetic customer dataset for churn modeling.
Run: python data/generate_data.py
Output: data/processed/customer_data.csv  (50,000 rows)
"""

import numpy as np
import pandas as pd
import os
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N = 50_000

def generate_customer_data(n: int = N) -> pd.DataFrame:
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, n + 1)]

    # Demographics
    age = np.random.randint(18, 75, n)
    gender = np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
    region = np.random.choice(["North", "South", "East", "West", "Central"], n)

    # Subscription
    contract_type = np.random.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.55, 0.25, 0.20])
    tenure_months = np.where(
        contract_type == "Month-to-Month",
        np.random.randint(1, 36, n),
        np.where(contract_type == "One Year", np.random.randint(6, 60, n), np.random.randint(12, 84, n))
    )
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges = np.round(monthly_charges * tenure_months * np.random.uniform(0.85, 1.05, n), 2)

    # Usage behavior
    num_products = np.random.randint(1, 6, n)
    login_frequency = np.random.randint(0, 30, n)       # per month
    support_tickets = np.random.poisson(1.5, n)
    payment_delays = np.random.poisson(0.5, n)
    last_interaction_days = np.random.randint(1, 180, n)

    # Satisfaction
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], n, p=[0.10, 0.15, 0.25, 0.30, 0.20])

    # Build feature matrix for churn probability
    churn_score = (
        0.30 * (contract_type == "Month-to-Month").astype(int)
        + 0.15 * (satisfaction_score <= 2).astype(int)
        + 0.15 * (support_tickets >= 3).astype(int)
        + 0.10 * (payment_delays >= 2).astype(int)
        + 0.10 * (login_frequency < 3).astype(int)
        + 0.10 * (last_interaction_days > 90).astype(int)
        - 0.10 * (tenure_months > 24).astype(int)
        - 0.10 * (num_products >= 4).astype(int)
        + np.random.normal(0, 0.05, n)   # noise
    )
    churn_prob = 1 / (1 + np.exp(-5 * (churn_score - 0.35)))   # sigmoid
    churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    # CLV (Customer Lifetime Value)
    clv = np.round(monthly_charges * tenure_months * (1 - 0.02 * support_tickets) * np.random.uniform(0.9, 1.1, n), 2)
    clv = np.clip(clv, 50, 20000)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "gender": gender,
        "region": region,
        "contract_type": contract_type,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_products": num_products,
        "login_frequency": login_frequency,
        "support_tickets": support_tickets,
        "payment_delays": payment_delays,
        "last_interaction_days": last_interaction_days,
        "satisfaction_score": satisfaction_score,
        "clv": clv,
        "churn": churn,
    })

    return df


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = generate_customer_data()
    out_path = "data/processed/customer_data.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df):,} records → {out_path}")
    print(f"   Churn rate: {df['churn'].mean():.2%}")
    print(f"   Avg CLV   : ₹{df['clv'].mean():,.2f}")
    print(df.head(3).to_string())
