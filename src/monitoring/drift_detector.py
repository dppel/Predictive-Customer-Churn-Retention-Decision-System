"""
Senior Enterprise Enhancement: Model Governance & Data Drift Monitoring
========================================================================
Calculates Population Stability Index (PSI) and feature distribution drift
between training baseline and production inference batches.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline and current distributions.
    
    PSI Rule of Thumb:
    - PSI < 0.10: No significant distribution change (Stable)
    - 0.10 <= PSI < 0.25: Moderate distribution shift (Warning / Monitor)
    - PSI >= 0.25: Significant distribution shift (Action Required / Retrain Model)
    """
    baseline = np.asarray(baseline)
    current = np.asarray(current)
    
    quantiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(baseline, quantiles)
    buckets[0] = -np.inf
    buckets[-1] = np.inf
    
    baseline_counts = np.histogram(baseline, bins=buckets)[0]
    current_counts = np.histogram(current, bins=buckets)[0]
    
    eps = 1e-4
    baseline_pct = (baseline_counts / len(baseline)) + eps
    current_pct = (current_counts / len(current)) + eps
    
    baseline_pct /= np.sum(baseline_pct)
    current_pct /= np.sum(current_pct)
    
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)

def monitor_feature_drift(
    df_baseline: pd.DataFrame, 
    df_current: pd.DataFrame, 
    features_to_monitor: List[str] = None
) -> Dict[str, Any]:
    """
    Monitors data drift across key features between reference training batch and current inference batch.
    """
    if features_to_monitor is None:
        features_to_monitor = ['MonthlyCharges', 'Tenure', 'SupportCallsCount', 'SatisfactionScore', 'CLV']
        
    drift_results = {}
    
    for feat in features_to_monitor:
        if feat in df_baseline.columns and feat in df_current.columns:
            psi_val = calculate_psi(df_baseline[feat].dropna().values, df_current[feat].dropna().values)
            
            if psi_val < 0.10:
                status = "STABLE"
            elif psi_val < 0.25:
                status = "WARNING SHIFT"
            else:
                status = "CRITICAL DRIFT (RETRAIN)"
                
            drift_results[feat] = {
                "psi": round(psi_val, 4),
                "status": status
            }
            
    print("\n" + "=" * 75)
    print("         PRODUCTION MODEL GOVERNANCE: DATA DRIFT MONITORING (PSI)       ")
    print("=" * 75)
    for feat, res in drift_results.items():
        print(f"- Feature '{feat:<22}': PSI = {res['psi']:.4f} -> Status: {res['status']}")
    print("=" * 75)
    
    return drift_results
