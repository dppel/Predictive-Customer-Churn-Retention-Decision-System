"""
Retention Decision & Expected Value Engine — STAGE 19
=====================================================
Computes Expected Benefit, Intervention Cost, and Expected Net Gain (ROI)
for each candidate retention action, selecting the optimal business intervention per customer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Retention Offer Catalog Definitions
RETENTION_CATALOG = {
    "NO_ACTION": {
        "name": "No Intervention",
        "cost": 0.0,
        "base_acceptance": 0.00
    },
    "DISCOUNT_VOUCHER": {
        "name": "15% Discount Voucher (3 Months)",
        "cost": 35.0,
        "base_acceptance": 0.45
    },
    "FEATURE_UPGRADE": {
        "name": "Free Service & Speed Upgrade",
        "cost": 65.0,
        "base_acceptance": 0.65
    },
    "DEDICATED_VIP": {
        "name": "Dedicated VIP Account Concierge",
        "cost": 150.0,
        "base_acceptance": 0.82
    }
}

def evaluate_retention_actions(
    churn_prob: float, 
    clv: float, 
    monthly_charges: float,
    risk_tier: str,
    catalog: Dict = None
) -> Dict[str, Any]:
    """
    Evaluates financial equations:
    - Annual Customer Value = MonthlyCharges * 12
    - Expected Benefit = Value * P(Churn) * P(Acceptance)
    - Expected Net Gain = Expected Benefit - Cost
    """
    if catalog is None:
        catalog = RETENTION_CATALOG
        
    annual_value = monthly_charges * 12.0
    best_action_key = "NO_ACTION"
    max_expected_gain = 0.0
    action_evaluations = {}
    
    for action_key, details in catalog.items():
        cost = details["cost"]
        p_acc = details["base_acceptance"]
        
        # Risk-tier adjusted acceptance rate
        if risk_tier == "Critical Risk":
            p_acc_adj = p_acc * 0.90
        elif risk_tier == "High Risk":
            p_acc_adj = p_acc * 0.95
        else:
            p_acc_adj = p_acc
            
        # Expected Benefit calculation using CLV
        expected_benefit = churn_prob * p_acc_adj * clv
        expected_net_gain = expected_benefit - cost
        
        action_evaluations[action_key] = {
            "name": details["name"],
            "cost": cost,
            "acceptance_prob": round(p_acc_adj, 3),
            "expected_benefit": round(expected_benefit, 2),
            "expected_net_gain": round(expected_net_gain, 2)
        }
        
        if expected_net_gain > max_expected_gain:
            max_expected_gain = expected_net_gain
            best_action_key = action_key
            
    optimal_details = action_evaluations[best_action_key]
    
    return {
        "annual_customer_value": round(annual_value, 2),
        "optimal_action_key": best_action_key,
        "optimal_action_name": optimal_details["name"],
        "intervention_cost": optimal_details["cost"],
        "expected_benefit": optimal_details["expected_benefit"],
        "expected_net_gain": optimal_details["expected_net_gain"],
        "all_evaluations": action_evaluations
    }

def apply_retention_decisions(df_segmented: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the retention decision engine across all customer records.
    """
    df_out = df_segmented.copy()
    
    annual_values = []
    optimal_actions = []
    costs = []
    expected_benefits = []
    net_gains = []
    
    for idx, row in df_out.iterrows():
        res = evaluate_retention_actions(
            churn_prob=row['Churn_Probability'],
            clv=row['CLV'],
            monthly_charges=row['MonthlyCharges'],
            risk_tier=row['Risk_Tier']
        )
        annual_values.append(res['annual_customer_value'])
        optimal_actions.append(res['optimal_action_name'])
        costs.append(res['intervention_cost'])
        expected_benefits.append(res['expected_benefit'])
        net_gains.append(res['expected_net_gain'])
        
    df_out['Annual_Customer_Value'] = annual_values
    df_out['Recommended_Action'] = optimal_actions
    df_out['Intervention_Cost'] = costs
    df_out['Expected_Benefit'] = expected_benefits
    df_out['Expected_Net_Gain'] = net_gains
    
    return df_out
