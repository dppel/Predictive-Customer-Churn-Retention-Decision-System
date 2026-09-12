"""
ROI & Economic Portfolio Analysis Module
========================================
Aggregates customer retention decisions into portfolio-level economic metrics, ROI lift,
and exports decision summary reports.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_roi_portfolio_report(
    df_decisions: pd.DataFrame, 
    figures_dir: str = "figures", 
    reports_dir: str = "reports"
) -> dict:
    """
    Computes portfolio-level retention financials and exports visual charts and decision CSV.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    total_customers = len(df_decisions)
    targeted_customers = (df_decisions['Recommended_Action'] != "No Intervention").sum()
    
    # Financial metrics
    total_baseline_churn_risk_val = (df_decisions['Churn_Probability'] * df_decisions['CLV']).sum()
    total_intervention_cost = df_decisions['Intervention_Cost'].sum()
    total_expected_net_gain = df_decisions['Expected_Net_Gain'].sum()
    
    roi_percentage = (total_expected_net_gain / total_intervention_cost * 100.0) if total_intervention_cost > 0 else 0.0
    
    summary = {
        "total_customers": int(total_customers),
        "targeted_customers": int(targeted_customers),
        "targeting_rate": float(targeted_customers / total_customers),
        "total_baseline_churn_risk_val": float(total_baseline_churn_risk_val),
        "total_intervention_budget_spent": float(total_intervention_cost),
        "total_expected_net_profit_saved": float(total_expected_net_gain),
        "portfolio_roi_percentage": float(roi_percentage)
    }
    
    # Save full CSV decision dataset
    csv_path = os.path.join(reports_dir, "customer_retention_decisions.csv")
    df_decisions.to_csv(csv_path, index=False)
    
    # 1. 2D Risk-Value Matrix Heatmap Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = df_decisions.pivot_table(index='Risk_Tier', columns='Value_Tier', values='Expected_Net_Gain', aggfunc='sum').fillna(0)
    # Reorder index & columns logically
    risk_order = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
    val_order = ['Bronze CLV', 'Silver CLV', 'Gold/Platinum CLV']
    pivot = pivot.reindex(index=[r for r in risk_order if r in pivot.index], columns=[v for v in val_order if v in pivot.columns])
    
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="Greens", cbar_kws={'label': 'Expected Net Gain ($)'}, ax=ax)
    ax.set_title("2D Risk-Value Matrix: Financial Net Gain ($)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Customer Value Tier (CLV)")
    ax.set_ylabel("Churn Risk Tier")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "risk_value_matrix_heatmap.png"), dpi=300)
    plt.close(fig)
    
    # 2. Recommended Action Breakdown Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    action_counts = df_decisions['Recommended_Action'].value_counts()
    sns.barplot(x=action_counts.values, y=action_counts.index, palette="Blues_r", ax=ax)
    ax.set_title("Retention Strategy Action Allocations", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Customers")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "action_allocations.png"), dpi=300)
    plt.close(fig)
    
    print(f"[ROI ANALYSIS] Portfolio Total Net Profit Preserved: ${total_expected_net_gain:,.2f} | Portfolio ROI: {roi_percentage:.1f}%")
    return summary
