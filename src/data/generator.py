"""
Customer Churn & Behavioral Data Generator
==========================================
Generates a realistic, high-fidelity customer dataset for Predictive Churn & Retention Decision Systems.
Includes customer demographics, usage metrics, financial records, experience ratings, and churn target.
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_customer_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic customer dataset with domain-specific churn signals.
    """
    np.random.seed(seed)
    
    # 1. Customer Profiles & Tenure
    customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
    ages = np.random.randint(18, 72, size=n_samples)
    genders = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.49, 0.51])
    locations = np.random.choice(['Urban', 'Suburban', 'Rural'], size=n_samples, p=[0.55, 0.30, 0.15])
    
    # Tenure in months (1 to 72 months)
    tenure = np.random.geometric(p=0.03, size=n_samples)
    tenure = np.clip(tenure, 1, 72)
    
    contract_types = np.random.choice(
        ['Month-to-Month', 'One-Year', 'Two-Year'], 
        size=n_samples, 
        p=[0.55, 0.28, 0.17]
    )
    
    base_date = datetime(2026, 9, 1)
    customer_since = [(base_date - timedelta(days=int(t * 30.4))).strftime('%Y-%m-%d') for t in tenure]
    
    # 2. Behavioral & Usage Metrics
    monthly_usage_hours = np.random.normal(loc=85, scale=30, size=n_samples).clip(5, 250)
    num_transactions = np.random.poisson(lam=6, size=n_samples).clip(0, 30)
    logins_per_month = np.random.poisson(lam=18, size=n_samples).clip(1, 60)
    call_minutes = np.random.normal(loc=450, scale=180, size=n_samples).clip(20, 1500)
    data_usage_gb = np.random.exponential(scale=65, size=n_samples).clip(2, 500)
    
    # 3. Financial Metrics
    # Higher charges generally associated with higher tier plans
    monthly_charges = np.round(np.random.uniform(25.0, 140.0, size=n_samples) + (data_usage_gb * 0.15), 2)
    # Total charges based on tenure with minor variance
    total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.95, 1.05, size=n_samples), 2)
    revenue = total_charges
    discounts_received = np.round(np.random.exponential(scale=35, size=n_samples), 2).clip(0, 450)
    
    # 4. Customer Experience
    support_calls_count = np.random.poisson(lam=1.8, size=n_samples).clip(0, 12)
    complaints_count = np.random.poisson(lam=0.7, size=n_samples).clip(0, 6)
    
    # Satisfaction score (1=Very Dissatisfied, 5=Very Satisfied)
    # Dissatisfaction increases with support calls & complaints
    sat_logits = 4.2 - (0.35 * support_calls_count) - (0.6 * complaints_count) + np.random.normal(0, 0.5, size=n_samples)
    satisfaction_score = np.round(sat_logits).clip(1, 5).astype(int)
    
    service_outages = np.random.poisson(lam=0.4, size=n_samples).clip(0, 5)
    
    # 5. Customer Lifetime Value (CLV) Calculation (Estimated remaining 24-month value)
    clv = np.round(monthly_charges * 24 * (satisfaction_score / 5.0) * (1 - 0.1 * complaints_count.clip(0, 5)), 2).clip(100, 10000)
    
    # 6. Realistic Churn Mechanism (Ground Truth Logit)
    # Risk drivers: Month-to-Month contract, low tenure, high monthly charges, high support calls, complaints, low satisfaction
    churn_score = (
        -1.5 
        + 1.8 * (contract_types == 'Month-to-Month')
        - 0.8 * (contract_types == 'Two-Year')
        - 0.04 * tenure
        + 0.015 * (monthly_charges - 60)
        + 0.45 * support_calls_count
        + 0.70 * complaints_count
        - 0.85 * (satisfaction_score - 3)
        + 0.40 * service_outages
        - 0.02 * logins_per_month
    )
    
    # Convert to probability via Sigmoid function
    churn_prob = 1.0 / (1.0 + np.exp(-churn_score))
    
    # Assign binary churn target
    churn = (np.random.uniform(0, 1, size=n_samples) < churn_prob).astype(int)
    
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': ages,
        'Gender': genders,
        'Location': locations,
        'Tenure': tenure,
        'ContractType': contract_types,
        'CustomerSince': customer_since,
        'MonthlyUsageHours': np.round(monthly_usage_hours, 1),
        'NumTransactions': num_transactions,
        'LoginsPerMonth': logins_per_month,
        'CallMinutes': np.round(call_minutes, 1),
        'DataUsageGB': np.round(data_usage_gb, 1),
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Revenue': revenue,
        'DiscountsReceived': discounts_received,
        'SupportCallsCount': support_calls_count,
        'ComplaintsCount': complaints_count,
        'SatisfactionScore': satisfaction_score,
        'ServiceOutagesReported': service_outages,
        'CLV': clv,
        'Churn': churn
    })
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate customer churn synthetic dataset.")
    parser.add_argument("--samples", type=int, default=5000, help="Number of customer records to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/raw/customer_churn_raw.csv", help="Output path")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = generate_customer_dataset(n_samples=args.samples, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"[SUCCESS] Generated {len(df)} customer records saved to '{args.output}'.")
    print(f"Churn Rate: {df['Churn'].mean():.2%}")
    print(f"Features: {list(df.columns)}")

if __name__ == "__main__":
    main()
