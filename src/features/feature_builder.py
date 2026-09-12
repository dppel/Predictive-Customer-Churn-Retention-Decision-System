"""
Feature Engineering Module — STAGE 14: Features with Business Rationale
========================================================================
Constructs domain-engineered features where EVERY feature has a clear,
documented Business Rationale tied to churn risk and retention optimization.
"""

import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches customer dataset with domain-engineered features based on explicit business rationale.
    
    1. Contract_Risk (ContractRisk):
       - Business Rationale: Month-to-Month contracts lack contractual exit barriers,
         making these customers 3x more vulnerable to immediate churn.
         
    2. Support_Contact_Rate (SupportContactRate):
       - Business Rationale: Support tickets per tenure month measure friction density.
         A high rate indicates unresolved technical or service dissatisfaction.
         
    3. Complaint_Severity_Index:
       - Business Rationale: Combines formal complaints count with low satisfaction scores.
         Captures acute dissatisfaction severity that single metrics miss.
         
    4. Revenue_Per_Month (RevenuePerMonth):
       - Business Rationale: Measures average monthly gross account value contribution (TotalCharges / Tenure),
         isolating high-paying accounts for VIP retention prioritization.
         
    5. Tenure_to_Monthly_Ratio (MonthlyUsageTrend / Price Sensitivity):
       - Business Rationale: Evaluates tenure loyalty relative to monthly bill price.
         Low ratios indicate price sensitivity in newly onboarded high-tier plans.
         
    6. Engagement_Score (EngagementScore):
       - Business Rationale: Composite index combining voice call minutes, data consumption,
         and login frequency. Low engagement precedes customer churn decisions.
         
    7. CLV_to_Monthly_Ratio:
       - Business Rationale: Measures lifetime value leverage relative to monthly cost.
         High CLV ratio accounts yield maximum Net Preserved Value during ROI optimization.
    """
    df_feat = df.copy()
    
    # 1. Contract Risk Flag (Binary)
    # Business Rationale: Month-to-Month contracts carry baseline higher structural churn risk
    df_feat['ContractRisk'] = (df_feat['ContractType'] == 'Month-to-Month').astype(int)
    
    # 2. Support Contact Rate (Support calls normalized by tenure months)
    # Business Rationale: Normalizes ticket volume across account age to detect high-friction accounts
    df_feat['SupportContactRate'] = np.round(df_feat['SupportCallsCount'] / (df_feat['Tenure'] + 1), 4)
    
    # 3. Complaint Severity Index
    # Business Rationale: Multiplies complaint count by dissatisfaction gap (6 - CSAT)
    df_feat['Complaint_Severity_Index'] = df_feat['ComplaintsCount'] * (6 - df_feat['SatisfactionScore'])
    
    # 4. Revenue Per Month ($)
    # Business Rationale: Normalizes cumulative historical revenue over active tenure months
    df_feat['RevenuePerMonth'] = np.round(df_feat['TotalCharges'] / (df_feat['Tenure'] + 1e-5), 2)
    
    # 5. Tenure to Monthly Charges Ratio
    # Business Rationale: Loyalty relative to monthly cost (price sensitivity index)
    df_feat['Tenure_to_Monthly_Ratio'] = np.round(df_feat['Tenure'] / (df_feat['MonthlyCharges'] + 1e-5), 4)
    
    # 6. Composite Service Engagement Score
    # Business Rationale: Aggregates voice, data, and login channels into a unified digital utilization index
    df_feat['EngagementScore'] = np.round(
        (df_feat['CallMinutes'] / 500.0) + (df_feat['DataUsageGB'] / 50.0) + (df_feat['LoginsPerMonth'] / 20.0), 2
    )
    
    # 7. CLV to Monthly Charge Multiple
    # Business Rationale: Relative lifetime account value leverage used in Level 4 ROI maximization
    df_feat['CLV_to_Monthly_Ratio'] = np.round(df_feat['CLV'] / (df_feat['MonthlyCharges'] + 1e-5), 2)
    
    print(f"[FEATURE ENGINEERING] Created 7 domain features with Business Rationale. Total features: {df_feat.shape[1]}")
    return df_feat
