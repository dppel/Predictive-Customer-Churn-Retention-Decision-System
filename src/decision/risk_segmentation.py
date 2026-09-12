"""
Customer Risk & Value Segmentation Module — STAGE 18
=====================================================
Classifies customers into a 4-Tier Business Risk Matrix based on calibrated churn probability P(Churn)
where thresholds (30%, 60%, 80%) are explicitly derived from Business Cost & ROI economics.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def assign_risk_tier(prob: float) -> str:
    """
    Assigns Risk Tier based on business cost thresholds:
    - 🟢 Low Risk (< 30%): Preserved value < intervention cost; no intervention.
    - 🟡 Medium Risk (30% - 60%): Monitor & low-cost discount voucher.
    - 🟠 High Risk (60% - 80%): Feature/speed upgrade intervention.
    - 🔴 Critical Risk (>= 80%): Priority retention with dedicated VIP concierge.
    """
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Medium Risk"
    elif prob < 0.80:
        return "High Risk"
    else:
        return "Critical Risk"

def assign_value_tier(clv: float, clv_quantiles: Tuple[float, float]) -> str:
    """
    Assigns Value Tier based on CLV distribution quantiles.
    """
    q33, q66 = clv_quantiles
    if clv < q33:
        return "Bronze CLV"
    elif clv < q66:
        return "Silver CLV"
    else:
        return "Gold/Platinum CLV"

def segment_customers(df: pd.DataFrame, churn_probs: np.ndarray) -> pd.DataFrame:
    """
    Appends Risk Tier, Value Tier, and 2D Risk-Value Matrix Segment to customer dataframe.
    """
    df_segmented = df.copy()
    df_segmented['Churn_Probability'] = np.round(churn_probs, 4)
    
    # 4-Tier Risk Classification
    df_segmented['Risk_Tier'] = [assign_risk_tier(p) for p in churn_probs]
    
    # Value Tiers based on CLV quantiles
    q33 = df_segmented['CLV'].quantile(0.33)
    q66 = df_segmented['CLV'].quantile(0.66)
    df_segmented['Value_Tier'] = [assign_value_tier(clv, (q33, q66)) for clv in df_segmented['CLV']]
    
    # Combined 2D Risk-Value Segment
    df_segmented['Risk_Value_Segment'] = df_segmented['Risk_Tier'] + " | " + df_segmented['Value_Tier']
    
    return df_segmented
