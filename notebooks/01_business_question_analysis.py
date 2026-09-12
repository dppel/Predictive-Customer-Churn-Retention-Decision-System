"""
STAGE 3 Notebook Analysis — Answering the Core Business Question
================================================================

Business Question:
"Which customers are most likely to churn, why are they likely to churn,
and which customers should the company target with retention interventions to maximize net profit?"
"""

import pandas as pd
import numpy as np

def run_business_question_analysis():
    df = pd.read_csv("reports/customer_retention_decisions.csv")
    
    print("=" * 80)
    print("   STAGE 3: THE 4-LEVEL BUSINESS QUESTION ANALYSIS REPORT   ")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # LEVEL 1 — PREDICTION: Who is likely to churn?
    # -------------------------------------------------------------------------
    print("\n--- [LEVEL 1: PREDICTION] Who is likely to churn? ---")
    total_cust = len(df)
    high_critical_risk = df[df['Risk_Tier'].isin(['High Risk', 'Critical Risk'])]
    avg_churn_prob = df['Churn_Probability'].mean()
    
    print(f"• Total Customer Base Evaluated: {total_cust:,}")
    print(f"• Average Calibrated Churn Probability: {avg_churn_prob:.1%}")
    print(f"• High & Critical Risk Customers: {len(high_critical_risk):,} ({len(high_critical_risk)/total_cust:.1%})")
    
    print("\nRisk Tier Breakdown:")
    print(df['Risk_Tier'].value_counts(normalize=True).map('{:.1%}'.format))
    
    # -------------------------------------------------------------------------
    # LEVEL 2 — EXPLANATION: Why is the customer likely to churn?
    # -------------------------------------------------------------------------
    print("\n--- [LEVEL 2: EXPLANATION] Why are customers likely to churn? ---")
    print("Top Dissatisfaction Drivers across High-Risk Customers:")
    print("1. Contract Type (Month-to-Month contracts carry highest baseline risk)")
    print("2. Complaint Severity Index (ComplaintsCount x Low SatisfactionScore)")
    print("3. Support Ticket Density (High SupportCallsCount relative to short Tenure)")
    print("4. Monthly Charges (High price sensitivity relative to service engagement)")
    
    # -------------------------------------------------------------------------
    # LEVEL 3 — DECISION: Should the company intervene?
    # -------------------------------------------------------------------------
    print("\n--- [LEVEL 3: DECISION] Should the company intervene? ---")
    targeted = df[df['Recommended_Action'] != "No Intervention"]
    print(f"• Customers Prescribed Retention Action: {len(targeted):,} ({len(targeted)/total_cust:.1%})")
    print(f"• Customers Assigned 'No Intervention' (Low risk / negative ROI): {total_cust - len(targeted):,}")
    
    print("\nRecommended Retention Strategy Distribution:")
    print(df['Recommended_Action'].value_counts())
    
    # -------------------------------------------------------------------------
    # LEVEL 4 — ECONOMICS: Is the intervention financially worthwhile?
    # -------------------------------------------------------------------------
    print("\n--- [LEVEL 4: ECONOMICS] Is the intervention financially worthwhile? ---")
    total_cost = df['Intervention_Cost'].sum()
    total_gain = df['Expected_Net_Gain'].sum()
    roi = (total_gain / total_cost * 100.0) if total_cost > 0 else 0.0
    
    print(f"• Total Intervention Budget Required: ${total_cost:,.2f}")
    print(f"• Total Preserved Net Financial Profit: ${total_gain:,.2f}")
    print(f"• Overall Portfolio Return on Investment (ROI): {roi:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_business_question_analysis()
