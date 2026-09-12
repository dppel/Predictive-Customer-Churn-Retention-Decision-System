"""
Exploratory Data Analysis (EDA) Module
======================================
Performs automated statistical analysis, correlation checks, feature distribution analysis,
and saves visualization artifacts to figures/ directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_exploratory_analysis(df: pd.DataFrame, output_dir: str = "figures") -> dict:
    """
    Runs automated EDA on customer dataframe and saves summary charts.
    Returns statistical summary dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    summary = {
        "total_customers": len(df),
        "churned_customers": int(df['Churn'].sum()),
        "churn_rate": float(df['Churn'].mean()),
        "avg_monthly_charges": float(df['MonthlyCharges'].mean()),
        "avg_tenure": float(df['Tenure'].mean()),
        "avg_clv": float(df['CLV'].mean())
    }
    
    # 1. Churn Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', hue='Churn', palette=['#2ecc71', '#e74c3c'], legend=False, ax=ax)
    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Stayed (0)', 'Churned (1)'])
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "churn_distribution.png"), dpi=300)
    plt.close(fig)
    
    # 2. Churn Rate by Contract Type
    fig, ax = plt.subplots(figsize=(8, 4.5))
    contract_churn = df.groupby('ContractType')['Churn'].mean().reset_index()
    sns.barplot(data=contract_churn, x='ContractType', y='Churn', hue='ContractType', palette='viridis', legend=False, ax=ax)
    ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight='bold')
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, 1.0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "churn_by_contract.png"), dpi=300)
    plt.close(fig)
    
    # 3. Churn Rate by Satisfaction Score & Support Calls
    fig, ax = plt.subplots(figsize=(9, 5))
    sat_supp = df.groupby(['SatisfactionScore', 'SupportCallsCount'])['Churn'].mean().unstack()
    sns.heatmap(sat_supp, annot=True, fmt=".0%", cmap="YlOrRd", ax=ax)
    ax.set_title("Churn Heatmap: Satisfaction Score vs. Support Calls Count", fontsize=14, fontweight='bold')
    ax.set_xlabel("Support Calls Count")
    ax.set_ylabel("Satisfaction Score (1=Low, 5=High)")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "churn_heatmap_satisfaction_support.png"), dpi=300)
    plt.close(fig)
    
    # 4. Feature Correlation Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close(fig)
    
    print(f"[EDA COMPLETED] Generated 4 summary charts saved in '{output_dir}/'.")
    return summary
